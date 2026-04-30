"""Tests for SeisCompStream.

The new SeisCompStream uses ObsPy SeedLink instead of seiscomp.Client.Application.
We verify the queue-based interface by injecting traces directly into the
internal queue, and also verify that FakeStream still works end-to-end.
"""

import threading
import numpy as np
from obspy import Trace, UTCDateTime
from obspy.core import Stats

from sceasyquake.stream.seiscomp_stream import SeisCompStream, FakeStream


def _make_trace(station='TEST', channel='HHZ', npts=500, sr=100.0):
    """Build a minimal ObsPy Trace for testing."""
    data = np.sin(np.linspace(0, 2 * np.pi, npts)).astype(np.float32)
    stats = Stats()
    stats.network = 'XX'
    stats.station = station
    stats.channel = channel
    stats.location = ''
    stats.sampling_rate = sr
    stats.starttime = UTCDateTime(2025, 1, 1)
    stats.npts = npts
    return Trace(data=data, header=stats)


def test_seiscomp_stream_queue_injection():
    """SeisCompStream returns traces placed directly in its internal queue."""
    tr = _make_trace()
    scs = SeisCompStream(seedlink_url='localhost:18000', stream_specs=['XX.TEST..HHZ'])

    # Inject a trace into the internal queue without starting SeedLink
    scs._queue.put_nowait(tr)

    # get_next_trace() should return it immediately
    result = scs.get_next_trace(timeout=0.5)
    assert result is not None
    assert result.stats.station == 'TEST'
    assert result.stats.channel == 'HHZ'
    assert result.stats.sampling_rate == 100.0


def test_seiscomp_stream_timeout_returns_none():
    """get_next_trace() returns None when the queue is empty and timeout elapses."""
    scs = SeisCompStream()
    result = scs.get_next_trace(timeout=0.05)
    assert result is None


def test_seiscomp_stream_multiple_traces():
    """Multiple injected traces are returned in order."""
    traces = [_make_trace(station=f'S{i}') for i in range(3)]
    scs = SeisCompStream()
    for tr in traces:
        scs._queue.put_nowait(tr)

    results = [scs.get_next_trace(timeout=0.1) for _ in range(3)]
    assert [r.stats.station for r in results] == ['S0', 'S1', 'S2']


def test_fake_stream_no_traces():
    """FakeStream with empty list returns None immediately."""
    fs = FakeStream([])
    fs.connect()
    assert fs.get_next_trace() is None


def test_fake_stream_with_traces():
    """FakeStream yields all supplied traces in order then returns None."""
    traces = [_make_trace(station=f'X{i}') for i in range(4)]
    fs = FakeStream(traces)
    fs.connect()

    returned = []
    for _ in range(5):
        t = fs.get_next_trace()
        returned.append(t)

    assert returned[:4] == traces
    assert returned[4] is None

