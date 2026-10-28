"""EarthScope bulk fetch must isolate a station that returns corrupt miniSEED.

IRIS occasionally serves an undecodable record for one station; obspy then
raises from readMSEEDBuffer() and rejects the whole bulk response.  Before
the fix, _fetch_earthscope_batch() treated that as an "unexpected error" and
gave up on every gap in the batch -- on the 2026-10-03 backfill three bad
stations (4O.WB06, SC.WBN01, TX.PB25) knocked out 200 of 350 gaps.
"""
import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn.header import (
    FDSNInternalServerException,
    FDSNNoDataException,
    FDSNRequestTooLargeException,
    FDSNTimeoutException,
)
from obspy.io.mseed import InternalMSEEDError

import gap_recovery as gr

T0 = UTCDateTime("2026-10-03T15:00:00")
BAD = "WB06"


class FakeClient:
    """Serves synthetic data, but any request touching BAD fails to decode."""

    def __init__(self):
        self.calls = 0

    def get_waveforms_bulk(self, bulk, quality=None):
        self.calls += 1
        if any(sta == BAD for _, sta, *_ in bulk):
            raise InternalMSEEDError(
                "Encountered 2 error(s) during a call to readMSEEDBuffer():\n"
                "msr_unpack_data(): only decoded 0 samples of 4096 expected")
        st = Stream()
        for net, sta, loc, cha, t1, t2 in bulk:
            n = int((t2 - t1) * 100)
            st += Trace(np.zeros(n, dtype=np.int32), header=dict(
                network=net, station=sta, location="" if loc == "*" else loc,
                channel=cha, starttime=t1, sampling_rate=100.0))
        return st


def _fetch(gaps, client, archive, stats):
    return gr._fetch_earthscope_batch(
        gaps, client, archive, "IRIS", archive / "no-scart", stats, False,
        FDSNNoDataException, FDSNRequestTooLargeException,
        FDSNTimeoutException, FDSNInternalServerException)


def test_corrupt_station_does_not_sink_batch(tmp_path):
    stations = [f"S{i:02d}" for i in range(9)]
    stations.insert(4, BAD)
    gaps = [gr.Gap("TX", sta, "00", "HHZ", T0, T0 + 120, 120.0, "test")
            for sta in stations]
    stats = gr.RecoveryStats()
    client = FakeClient()

    unresolved = _fetch(gaps, client, tmp_path, stats)

    assert [g.station for g in unresolved] == [BAD]
    assert stats.gaps_earthscope == 9
    assert any(BAD in e for e in stats.errors)
    # Bisection, not one request per gap: well under 2N-1 = 19 calls.
    assert client.calls <= 10
