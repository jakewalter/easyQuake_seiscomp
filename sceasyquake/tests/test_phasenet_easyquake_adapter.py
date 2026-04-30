import sys
import types
import numpy as np
from obspy import Trace, UTCDateTime
from sceasyquake.predictors.phasenet import PhaseNetPredictor


class DummyPhaseNetModule:
    @staticmethod
    def predict_trace(trace, model_path=None, device='cpu'):
        # return a probability array with a clear peak near the middle
        n = trace.stats.npts
        probs = np.zeros(n)
        mid = n // 2
        probs[mid-1:mid+2] = 1.0
        return probs


def test_easyquake_adapter_monkeypatch(monkeypatch):
    # create a fake easyQuake package with phase_net module
    dummy_pkg = types.SimpleNamespace()
    dummy_pkg.phase_net = DummyPhaseNetModule()
    monkeypatch.setitem(sys.modules, 'easyQuake', dummy_pkg)

    tr = Trace(data=np.random.randn(500))
    tr.stats.sampling_rate = 100.0
    tr.stats.starttime = UTCDateTime(2025, 12, 14)
    tr.stats.station = 'TEST'
    tr.stats.channel = 'BHZ'

    p = PhaseNetPredictor(backend='easyquake')
    picks = p.predict(tr)
    assert len(picks) >= 1
    assert picks[0]['station'] == 'TEST'
    assert picks[0]['phase'] == 'P'
    # time should be within trace start..end
    assert tr.stats.starttime <= picks[0]['time'] <= tr.stats.endtime
