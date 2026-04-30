"""Stream worker - buffers traces per-station and runs ML prediction on windows.

Architecture:
    _recv_thread: drains traces from source into per-station obspy.Stream buffers.
    _pred_thread: every step_seconds, runs prediction on stations with enough data.

This two-thread design ensures SeedLink packets never block prediction and that
ML models always receive a full-length continuously merged trace window (>= buffer_seconds),
which is required by SeisBench PhaseNet.
"""
import logging
import time
from threading import Thread, Event, Lock
from collections import defaultdict

log = logging.getLogger(__name__)

try:
    from obspy import Stream
    _HAS_OBSPY = True
except ImportError:
    _HAS_OBSPY = False


class PredictorBase:
    def predict(self, stream):
        raise NotImplementedError()


class StreamWorker:
    def __init__(self, predictor: PredictorBase, uploader, buffer_seconds=60, step_seconds=10):
        self.predictor = predictor
        self.uploader = uploader
        self.buffer_seconds = buffer_seconds
        self.step_seconds = step_seconds
        self._stop = Event()
        # Per-station waveform buffers: key = 'NET.STA.LOC'
        self._buffers: dict = defaultdict(Stream) if _HAS_OBSPY else defaultdict(list)
        self._buf_lock = Lock()   # protects _buffers mutations
        self._last_pred: dict = {}   # key -> unix time of last prediction
        # Per-(net,sta,loc) earliest acceptable pick time to avoid re-publishing
        # picks from the overlapping portion of successive sliding windows.
        self._nsl_pick_cutoff: dict = {}  # (net,sta,loc) -> float (epoch seconds)
        self.source = None

    def start(self):
        log.info('Starting stream worker (buffer=%ds, step=%ds)', self.buffer_seconds, self.step_seconds)
        self._recv_thread = Thread(target=self._recv_loop, daemon=True, name='sceasyquake-recv')
        self._pred_thread = Thread(target=self._pred_loop, daemon=True, name='sceasyquake-pred')
        self._recv_thread.start()
        self._pred_thread.start()

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    # Receive loop – drain SeedLink into per-station buffers
    # ------------------------------------------------------------------

    def _recv_loop(self):
        while not self._stop.is_set():
            trace = None
            if self.source is not None:
                trace = self.source.get_next_trace(timeout=1.0)
            else:
                trace = self.get_next_chunk()
            if trace is None:
                continue
            try:
                key = f'{trace.stats.network}.{trace.stats.station}.{trace.stats.location}'
                if _HAS_OBSPY:
                    with self._buf_lock:
                        self._buffers[key] += Stream(traces=[trace])
                        buf = self._buffers[key]
                        buf.merge(method=1, fill_value=0)
                        if len(buf) > 0:
                            latest_end = max(tr.stats.endtime for tr in buf)
                            clip_start = latest_end - self.buffer_seconds * 2
                            buf.trim(starttime=clip_start, nearest_sample=False)
            except Exception as exc:
                log.debug('Buffer update error for %s: %s', getattr(trace, 'id', '?'), exc)

    # ------------------------------------------------------------------
    # Prediction loop – fire ML model on buffered windows
    # ------------------------------------------------------------------

    def _pred_loop(self):
        while not self._stop.is_set():
            self._stop.wait(self.step_seconds)
            if self._stop.is_set():
                break
            now = time.time()
            with self._buf_lock:
                keys_and_copies = []
                # nsl_cutoffs: (net,sta,loc) -> float epoch of earliest acceptable pick.
                # Only picks at or after (buf_end - step_seconds) are new this step;
                # earlier picks were already published in a previous window pass.
                nsl_cutoffs = {}
                for key, buf in list(self._buffers.items()):
                    if not _HAS_OBSPY or len(buf) == 0:
                        continue
                    duration = max(
                        tr.stats.npts / tr.stats.sampling_rate
                        for tr in buf if tr.stats.sampling_rate > 0
                    )
                    if duration < self.buffer_seconds:
                        continue
                    if now - self._last_pred.get(key, 0) < self.step_seconds * 0.9:
                        continue
                    self._last_pred[key] = now
                    buf_copy = buf.copy()
                    # Record the cutoff for this (net, sta, loc): picks must be
                    # >= buf_end - step_seconds to be considered new.
                    try:
                        buf_end = max(float(tr.stats.endtime) for tr in buf_copy)
                        cutoff = buf_end - self.step_seconds
                        parts = key.split('.', 2)
                        nsl = (parts[0], parts[1], parts[2] if len(parts) > 2 else '')
                        # Use the most-recent cutoff seen for this station buffer
                        if nsl not in nsl_cutoffs or cutoff > nsl_cutoffs[nsl]:
                            nsl_cutoffs[nsl] = cutoff
                    except Exception:
                        pass
                    keys_and_copies.append((key, buf_copy))

            # Batch all ready station buffers into a single predict_multi() call so
            # the SeisBench backend can run one GPU annotate() pass for everything
            # rather than hundreds of sequential forward passes.
            if not keys_and_copies:
                continue
            try:
                if hasattr(self.predictor, 'predict_multi'):
                    all_picks = self.predictor.predict_multi(keys_and_copies)
                else:
                    all_picks = []
                    for key, work_stream in keys_and_copies:
                        all_picks.extend(self.predictor.predict(work_stream))
            except Exception as exc:
                log.exception('pred_loop batch error: %s', exc)
                continue

            # Filter out picks that fall in the already-published overlap region.
            # Any pick whose time < (buf_end - step_seconds) was already reported
            # in a previous window and must be suppressed.
            new_picks = []
            for p in all_picks:
                try:
                    nsl = (p.get('network', ''), p.get('station', ''), p.get('location', ''))
                    cutoff = nsl_cutoffs.get(nsl)
                    if cutoff is None or float(p['time']) >= cutoff:
                        new_picks.append(p)
                except Exception:
                    new_picks.append(p)
            if len(all_picks) != len(new_picks):
                log.debug('Suppressed %d duplicate picks from overlapping window',
                          len(all_picks) - len(new_picks))
            all_picks = new_picks

            # Distribute picks to uploader
            pick_counts: dict = {}
            for p in all_picks:
                key = f"{p.get('network')}.{p.get('station')}." \
                      f"{p.get('location')}.{p.get('channel')}"
                try:
                    self.uploader.send_pick(**p)
                    pick_counts[key] = pick_counts.get(key, 0) + 1
                except Exception as exc:
                    log.exception('Upload error for pick on %s: %s', key, exc)
            if pick_counts:
                total = sum(pick_counts.values())
                log.info('Published %d pick(s) across %d channel(s) this step',
                         total, len(pick_counts))

    def get_next_chunk(self):
        # Placeholder for test injection / custom sources
        return None
