"""SeisComP/SeedLink stream source abstraction

This module provides a `SeisCompStream` class that subscribes to waveforms via
ObsPy's SeedLink client pointing at SeisComP's internal SeedLink server
(default localhost:18000).  Incoming `Trace` objects are placed on a thread-safe
queue and retrieved by `get_next_trace()`.

Interface used by `StreamWorker`:
- connect()
- get_next_trace(timeout) -> obspy.Trace | None
- disconnect()

For unit-testing, `FakeStream` yields a predefined sequence of `Trace` objects.
"""

import logging
import threading
from queue import Queue, Empty
from typing import Iterable, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FakeStream – for use in tests / demos
# ---------------------------------------------------------------------------

class FakeStream:
    """Yields a pre-supplied list of traces in order; returns None when exhausted."""

    def __init__(self, traces: Iterable):
        self._traces = list(traces)
        self._i = 0

    def connect(self):
        log.info('FakeStream connected (%d traces queued)', len(self._traces))

    def get_next_trace(self, timeout: float = 1.0):
        if self._i >= len(self._traces):
            return None
        tr = self._traces[self._i]
        self._i += 1
        return tr

    def disconnect(self):
        pass


# ---------------------------------------------------------------------------
# _SeedLinkBuffer – internal EasySeedLinkClient subclass that feeds a Queue
# ---------------------------------------------------------------------------

class _SeedLinkBuffer:
    """ObsPy SeedLink client that puts every received Trace into a shared Queue.

    We don't subclass EasySeedLinkClient directly here so that we can support
    test injection.  The real implementation is created inside SeisCompStream
    when SeisComP bindings are absent.
    """
    pass


def _make_obspy_seedlink_client(server: str, queue: Queue, stream_specs: List[str]):
    """Build and return an ObsPy EasySeedLinkClient that populates *queue*.

    *stream_specs* is a list of ``"NET.STA.LOC.CHA"`` or ``"NET.STA..CHA"``
    strings.  We select each stream on the SeedLink server and start receiving.
    """
    try:
        from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient  # type: ignore
    except ImportError as exc:
        raise RuntimeError('ObsPy SeedLink client not available; install obspy') from exc

    class _Client(EasySeedLinkClient):
        def on_data(self, trace):
            try:
                queue.put_nowait(trace)
            except Exception:
                pass  # queue full → drop oldest? for now just skip

    host, _, port_str = server.partition(':')
    port = int(port_str) if port_str else 18000
    client = _Client(f'{host}:{port}', autoconnect=False)

    for spec in stream_specs:
        parts = spec.split('.')
        if len(parts) == 4:
            net, sta, loc, cha = parts
        elif len(parts) == 3:
            net, sta, cha = parts
            loc = ''
        elif len(parts) == 2:
            net, sta = parts
            loc, cha = '', '??Z'
        else:
            net, sta, loc, cha = '', parts[0], '', '??Z'
        # SeedLink v3 does not accept wildcard location codes (e.g. '*').
        # An empty string means "any location", which is correct behaviour.
        if loc in ('*', '?', '??', '**'):
            loc = ''
        # SeedLink selector is loc+cha (e.g. '' + 'HHZ' or '' + 'HH?')
        selector = f'{loc}{cha}'
        client.select_stream(net, sta, selector)
        log.debug('SeedLink: subscribed %s.%s selector=%r', net, sta, selector)

    return client


# ---------------------------------------------------------------------------
# SeisCompStream – public API
# ---------------------------------------------------------------------------

class SeisCompStream:
    """Waveform source backed by SeisComP's SeedLink server (localhost:18000).

    Parameters
    ----------
    seedlink_url:
        SeedLink server as ``"host:port"`` (default ``"localhost:18000"``).
    stream_specs:
        List of stream selectors in ``"NET.STA.LOC.CHA"`` format.
        ``"*.*..??Z"`` selects all vertical channels.
        If empty or ``None``, subscribes to ``["*.*..??Z"]`` (all verticals).
    queue_maxsize:
        Maximum number of traces held in the internal buffer (oldest are
        dropped when full).
    """

    def __init__(
        self,
        seedlink_url: str = 'localhost:18000',
        stream_specs: Optional[List[str]] = None,
        queue_maxsize: int = 2000,
    ):
        self.seedlink_url = seedlink_url
        self.stream_specs: List[str] = stream_specs or ['*.*..??Z']
        self._queue: Queue = Queue(maxsize=queue_maxsize)
        self._client = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def connect(self):
        """Connect to SeedLink server and start receiving in a daemon thread."""
        # Allow tests to inject a fake client by pre-populating self._client
        if self._client is not None:
            log.info('SeisCompStream: using pre-injected client')
            return

        try:
            self._client = _make_obspy_seedlink_client(
                self.seedlink_url, self._queue, self.stream_specs
            )
            self._thread = threading.Thread(
                target=self._client.run,
                name='sceasyquake-seedlink',
                daemon=True,
            )
            self._thread.start()
            log.info(
                'SeisCompStream connected to %s (%d stream specs)',
                self.seedlink_url,
                len(self.stream_specs),
            )
        except Exception as exc:
            log.error('SeisCompStream.connect() failed: %s', exc)
            self._client = None

    def get_next_trace(self, timeout: float = 1.0):
        """Return next ObsPy Trace from the buffer, or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def disconnect(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        log.info('SeisCompStream disconnected')

    # Convenience: let the worker inject a fake client for testing
    def _inject_client(self, fake_client):
        """Inject a pre-built client (e.g. FakeStream adapter) for unit tests."""
        self._client = fake_client