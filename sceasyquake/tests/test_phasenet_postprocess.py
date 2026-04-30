from obspy import Trace
import numpy as np
from sceasyquake.predictors.phasenet import PhaseNetPredictor
from obspy import UTCDateTime


def make_fake_trace(sr=100.0, npts=1000):
    t = np.zeros(npts)
    # create two peaks in the probability domain (we'll feed them as probs)
    tr = Trace(data=np.random.randn(npts))
    tr.stats.sampling_rate = sr
    tr.stats.starttime = UTCDateTime(2025, 12, 14)
    tr.stats.station = 'FAKE'
    tr.stats.channel = 'BHZ'
    return tr


def test_stub_predictor_returns_simulated_pick():
    tr = make_fake_trace()
    p = PhaseNetPredictor(backend='stub')
    picks = p.predict(tr)
    assert len(picks) == 1
    assert picks[0]['station'] == 'FAKE'
    assert picks[0]['phase'] == 'P'
