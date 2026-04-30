import time
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
