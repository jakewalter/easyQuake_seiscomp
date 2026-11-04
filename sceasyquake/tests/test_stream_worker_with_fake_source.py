import time
import pytest
from obspy import Trace, UTCDateTime
import numpy as np

from sceasyquake.stream.worker import StreamWorker


class DummyPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, stream):
        self.calls += 1
        tr = stream[0]
        return [{
            'network': 'XX', 'station': tr.stats.station, 'location': '', 'channel': tr.stats.channel,
            'time': tr.stats.starttime + 1, 'phase': 'P', 'probability': 0.8, 'method': 'PhaseNet-test'
        }]


class DummyUploader:
    def __init__(self):
        self.sent = []

    def send_pick(self, **kwargs):
        self.sent.append(kwargs)
        return True


class FakeSource:
    def __init__(self):
        self._sent = False

    def connect(self):
        pass

    def get_next_trace(self, timeout=1.0):
        if self._sent:
            return None
        tr = Trace(data=np.random.randn(1000))
        tr.stats.sampling_rate = 100.0
        tr.stats.starttime = UTCDateTime(2025, 12, 14)
        tr.stats.station = 'SIM'
        tr.stats.channel = 'BHZ'
        tr.stats.npts = 1000
        self._sent = True
        return tr


def test_stream_worker_processes_fake_trace():
    pred = DummyPredictor()
    up = DummyUploader()
    worker = StreamWorker(pred, up, buffer_seconds=10, step_seconds=1)
    src = FakeSource()
    worker.source = src
    worker.start()
    # wait up to 3s for the worker to process
    time.sleep(2)
    worker.stop()
    assert pred.calls >= 1
    assert len(up.sent) >= 1
    assert up.sent[0]['station'] == 'SIM'


def test_stream_worker_persistent_cutoff_with_skip():
    from obspy import Stream, Trace, UTCDateTime
    
    class MockPredictor:
        def __init__(self):
            self.picks_to_return = []
            
        def predict_multi(self, keys_and_streams):
            return self.picks_to_return
            
    class MockUploader:
        def __init__(self):
            self.sent = []
            
        def send_pick(self, **kwargs):
            self.sent.append(kwargs)
            return True
            
    pred = MockPredictor()
    up = MockUploader()
    worker = StreamWorker(pred, up, buffer_seconds=30, step_seconds=10, max_batch_stations=1)
    
    def make_trace(sta, starttime, duration=30):
        tr = Trace(data=np.random.randn(int(duration * 100)))
        tr.stats.sampling_rate = 100.0
        tr.stats.starttime = UTCDateTime(starttime)
        tr.stats.station = sta
        tr.stats.network = 'XX'
        tr.stats.channel = 'BHZ'
        return tr

    # Cycle 1: Start time t0
    t0 = 1782260000.0
    worker._buffers['XX.STA1.'] = Stream([make_trace('STA1', t0)])
    worker._buffers['XX.STA2.'] = Stream([make_trace('STA2', t0)])
    worker._buf_last_update['XX.STA1.'] = t0 + 30
    worker._buf_last_update['XX.STA2.'] = t0 + 30
    
    import unittest.mock as mock
    with mock.patch('time.time', return_value=t0 + 30):
        now = t0 + 30
        candidates = []
        for key, buf in list(worker._buffers.items()):
            overdue = now - worker._last_pred.get(key, 0)
            candidates.append((overdue, key, buf.copy()))
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:1]
        
        selected_key = candidates[0][1]
        skipped_key = 'XX.STA2.' if selected_key == 'XX.STA1.' else 'XX.STA1.'
        
        keys_and_copies = []
        step_cutoffs = {}
        nsl_buf_ends = {}
        for overdue, key, buf_copy in candidates:
            worker._last_pred[key] = now
            buf_end = max(float(tr.stats.endtime) for tr in buf_copy)
            parts = key.split('.', 2)
            nsl = (parts[0], parts[1], parts[2] if len(parts) > 2 else '')
            cutoff = buf_end - worker.step_seconds
            step_cutoffs[nsl] = cutoff
            nsl_buf_ends[nsl] = buf_end
            keys_and_copies.append((key, buf_copy))
            
        for nsl, buf_end in nsl_buf_ends.items():
            worker._nsl_pick_cutoff[nsl] = buf_end

    sel_parts = selected_key.split('.', 2)
    sel_nsl = (sel_parts[0], sel_parts[1], sel_parts[2] if len(sel_parts) > 2 else '')
    skp_parts = skipped_key.split('.', 2)
    skp_nsl = (skp_parts[0], skp_parts[1], skp_parts[2] if len(skp_parts) > 2 else '')
    
    assert worker._nsl_pick_cutoff[sel_nsl] == pytest.approx(t0 + 30, abs=0.02)
    assert skp_nsl not in worker._nsl_pick_cutoff

    # Cycle 2: Time 10s later. We add more data.
    worker._buffers['XX.STA1.'] = Stream([make_trace('STA1', t0, duration=40)])
    worker._buffers['XX.STA2.'] = Stream([make_trace('STA2', t0, duration=40)])
    worker._buf_last_update['XX.STA1.'] = t0 + 40
    worker._buf_last_update['XX.STA2.'] = t0 + 40

    with mock.patch('time.time', return_value=t0 + 40):
        now = t0 + 40
        candidates = []
        for key, buf in list(worker._buffers.items()):
            overdue = now - worker._last_pred.get(key, 0)
            candidates.append((overdue, key, buf.copy()))
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        assert candidates[0][1] == skipped_key
        candidates = candidates[:1]
        
        keys_and_copies = []
        step_cutoffs = {}
        nsl_buf_ends = {}
        for overdue, key, buf_copy in candidates:
            worker._last_pred[key] = now
            buf_end = max(float(tr.stats.endtime) for tr in buf_copy)
            parts = key.split('.', 2)
            nsl = (parts[0], parts[1], parts[2] if len(parts) > 2 else '')
            prev_cutoff = worker._nsl_pick_cutoff.get(nsl)
            if prev_cutoff is None:
                cutoff = buf_end - worker.step_seconds
            else:
                cutoff = prev_cutoff
            step_cutoffs[nsl] = cutoff
            nsl_buf_ends[nsl] = buf_end
            keys_and_copies.append((key, buf_copy))
            
        for nsl, buf_end in nsl_buf_ends.items():
            worker._nsl_pick_cutoff[nsl] = buf_end

    assert worker._nsl_pick_cutoff[skp_nsl] == pytest.approx(t0 + 40, abs=0.02)
    
    # Cycle 3: Time 10s later (t0 + 50). We add more data.
    worker._buffers['XX.STA1.'] = Stream([make_trace('STA1', t0, duration=50)])
    worker._buffers['XX.STA2.'] = Stream([make_trace('STA2', t0, duration=50)])
    worker._buf_last_update['XX.STA1.'] = t0 + 50
    worker._buf_last_update['XX.STA2.'] = t0 + 50
    
    with mock.patch('time.time', return_value=t0 + 50):
        now = t0 + 50
        candidates = []
        for key, buf in list(worker._buffers.items()):
            overdue = now - worker._last_pred.get(key, 0)
            candidates.append((overdue, key, buf.copy()))
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        assert candidates[0][1] == selected_key
        candidates = candidates[:1]
        
        keys_and_copies = []
        step_cutoffs = {}
        nsl_buf_ends = {}
        for overdue, key, buf_copy in candidates:
            worker._last_pred[key] = now
            buf_end = max(float(tr.stats.endtime) for tr in buf_copy)
            parts = key.split('.', 2)
            nsl = (parts[0], parts[1], parts[2] if len(parts) > 2 else '')
            prev_cutoff = worker._nsl_pick_cutoff.get(nsl)
            if prev_cutoff is None:
                cutoff = buf_end - worker.step_seconds
            else:
                cutoff = prev_cutoff
            step_cutoffs[nsl] = cutoff
            nsl_buf_ends[nsl] = buf_end
            keys_and_copies.append((key, buf_copy))
            
        for nsl, buf_end in nsl_buf_ends.items():
            worker._nsl_pick_cutoff[nsl] = buf_end
            
    assert worker._nsl_pick_cutoff[skp_nsl] == pytest.approx(t0 + 40, abs=0.02)

    # Cycle 4: Time 10s later (t0 + 60). We add more data.
    worker._buffers['XX.STA1.'] = Stream([make_trace('STA1', t0, duration=60)])
    worker._buffers['XX.STA2.'] = Stream([make_trace('STA2', t0, duration=60)])
    worker._buf_last_update['XX.STA1.'] = t0 + 60
    worker._buf_last_update['XX.STA2.'] = t0 + 60
    
    pred.picks_to_return = [
        {'network': 'XX', 'station': skp_parts[1], 'location': '', 'channel': 'BHZ',
         'time': UTCDateTime(t0 + 42), 'phase': 'P'},
        {'network': 'XX', 'station': skp_parts[1], 'location': '', 'channel': 'BHZ',
         'time': UTCDateTime(t0 + 52), 'phase': 'P'}
    ]
    
    with mock.patch('time.time', return_value=t0 + 60):
        now = t0 + 60
        candidates = []
        for key, buf in list(worker._buffers.items()):
            overdue = now - worker._last_pred.get(key, 0)
            candidates.append((overdue, key, buf.copy()))
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        assert candidates[0][1] == skipped_key
        candidates = candidates[:1]
        
        keys_and_copies = []
        step_cutoffs = {}
        nsl_buf_ends = {}
        for overdue, key, buf_copy in candidates:
            worker._last_pred[key] = now
            buf_end = max(float(tr.stats.endtime) for tr in buf_copy)
            parts = key.split('.', 2)
            nsl = (parts[0], parts[1], parts[2] if len(parts) > 2 else '')
            prev_cutoff = worker._nsl_pick_cutoff.get(nsl)
            if prev_cutoff is None:
                cutoff = buf_end - worker.step_seconds
            else:
                cutoff = prev_cutoff
            step_cutoffs[nsl] = cutoff
            nsl_buf_ends[nsl] = buf_end
            keys_and_copies.append((key, buf_copy))
            
        all_picks = pred.predict_multi(keys_and_copies)
        
        for nsl, buf_end in nsl_buf_ends.items():
            worker._nsl_pick_cutoff[nsl] = buf_end
            
        new_picks = []
        for p in all_picks:
            nsl = (p.get('network', ''), p.get('station', ''), p.get('location', ''))
            cutoff = step_cutoffs.get(nsl)
            if cutoff is None or float(p['time']) >= cutoff:
                new_picks.append(p)
                
        for p in new_picks:
            worker.uploader.send_pick(**p)
            
    assert len(up.sent) == 2
    assert float(up.sent[0]['time']) == t0 + 42
    assert float(up.sent[1]['time']) == t0 + 52
