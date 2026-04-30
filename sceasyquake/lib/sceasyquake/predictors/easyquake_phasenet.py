"""EasyQuake TF-based PhaseNet predictor.

Wraps the easyQuake phasenet_predict.py script as a single-subprocess batch
call.  easyQuake 2.0 requires TensorFlow ≥ 2.12 and Python ≥ 3.10.  The PhaseNet
TF1-checkpoint path uses ``tf.compat.v1`` (still supported in TF 2.12+) while the
TF2/Keras3 path uses native Keras.  Either way, running TF and PyTorch in the same
process risks conflicts, so a subprocess is the preferred approach.

Batch design
------------
Unlike the other predictors, ``predict(stream)`` for a single station is
intentionally deferred.  All streams are accumulated in ``_pending`` and
flushed when ``predict_batch()`` is called (or when ``predict()`` is called
with a stream whose station key already has a pending entry).

For the benchmark the caller should:
1. Call ``predictor.add_stream(key, stream)`` for every station.
2. Call ``predictor.run_batch()`` to get all picks in one subprocess pass.

``predict(stream)`` is also supported for compatibility — it flushes a single
station in its own subprocess call (slower, used for streaming).

Config parameters
-----------------
model_dir   : path to TF checkpoint dir (default: easyQuake bundled model)
python_cmd  : Python interpreter with TF (default: easyquake conda env)
threshold   : probability threshold for picks (applied post-hoc)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def _find_easyquake_python() -> str:
    """Return path to a Python interpreter that has easyQuake installed.

    Search order:
      1. Current interpreter — works when sceasyquake and easyQuake share the
         same env (e.g. a SeisComP venv that also has TF installed).
      2. Common conda/mamba environment names in standard base directories.
      3. ``python3`` on PATH as a last resort.

    No subprocesses are spawned; we only check file existence plus
    ``importlib.util.find_spec`` for the current interpreter.
    """
    import importlib.util as _ilu
    import shutil

    def _has_tf(python_path: str) -> bool:
        """True if ``python_path`` has both easyQuake and tensorflow."""
        try:
            prefix = os.path.dirname(os.path.dirname(python_path))
            for lib_dir in (
                os.path.join(prefix, 'lib'),
                os.path.join(prefix, 'lib', 'python3.7', 'site-packages'),
                os.path.join(prefix, 'lib', 'python3.8', 'site-packages'),
                os.path.join(prefix, 'lib', 'python3.9', 'site-packages'),
                os.path.join(prefix, 'lib', 'python3.10', 'site-packages'),
                os.path.join(prefix, 'lib', 'python3.11', 'site-packages'),
            ):
                if os.path.isdir(os.path.join(lib_dir, 'tensorflow')) or \
                   os.path.isdir(os.path.join(lib_dir, 'tensorflow_core')):
                    return True
            import glob
            for tf_dir in glob.glob(os.path.join(prefix, 'lib', 'python*', 'site-packages', 'tensorflow*')):
                if os.path.isdir(tf_dir):
                    return True
        except Exception:
            pass
        return False

    # 1. Dedicated conda / mamba env named after easyquake (preferred).
    for base in (
        os.path.expanduser('~/anaconda3'),
        os.path.expanduser('~/miniconda3'),
        os.path.expanduser('~/mambaforge'),
        os.path.expanduser('~/miniforge3'),
        '/opt/conda',
    ):
        for env_name in ('easyquake', 'easyQuake', 'seisml', 'tf', 'seismology'):
            candidate = os.path.join(base, 'envs', env_name, 'bin', 'python')
            if os.path.isfile(candidate) and _has_tf(candidate):
                return candidate

    # 2. Current interpreter — only if it also has TF
    if _ilu.find_spec('easyQuake') is not None and _ilu.find_spec('tensorflow') is not None:
        return sys.executable

    # 3. Any conda env candidate that at least exists
    for base in (
        os.path.expanduser('~/anaconda3'),
        os.path.expanduser('~/miniconda3'),
        os.path.expanduser('~/mambaforge'),
        os.path.expanduser('~/miniforge3'),
        '/opt/conda',
    ):
        for env_name in ('easyquake', 'easyQuake', 'seisml', 'tf', 'seismology'):
            candidate = os.path.join(base, 'envs', env_name, 'bin', 'python')
            if os.path.isfile(candidate):
                return candidate

    # 4. Fall back to python3 on PATH
    return shutil.which('python3') or sys.executable


_DEFAULT_PYTHON = _find_easyquake_python()


def _resolve_phasenet_paths(python_cmd: str):
    """Return (phasenet_predict_path, model_dir) by querying the target interpreter.

    Prefers the repo-local patched script under sceasyquake/share/scripts/ so
    the repository is self-contained and deployable without modifying the conda
    env.  Falls back to the installed easyQuake package when the local copy
    is absent.

    The *installed* easyQuake package (site-packages) has plain absolute imports
    and can be run as a script.  The git-checkout version uses relative imports
    and cannot.  We therefore always resolve model paths from the target Python.
    """
    # 1. Repo-local patched scripts (preferred — self-contained deployment)
    # Use .resolve() so symlinks from $SEISCOMP_ROOT/lib/python/ are followed
    # before computing parents; otherwise parents[3] lands in SeisComP's lib/.
    _local_scripts = Path(__file__).resolve().parents[3] / 'share' / 'scripts'
    _local_pn = _local_scripts / 'phasenet_predict.py'
    if _local_pn.exists():
        # Model weights still live inside the easyQuake package.
        # Import top-level easyQuake package only — easyQuake.phasenet has TF
        # module-level side-effects that cause bare imports to fail.
        probe = (
            'import os, easyQuake;'
            'print(os.path.join(os.path.dirname(easyQuake.__file__), "phasenet"))'
        )
        try:
            result = subprocess.run(
                [python_cmd, '-c', probe],
                capture_output=True, text=True, timeout=15,
            )
            # The easyQuake package prints its banner to stdout; take last line only
            phasenet_dir = Path(result.stdout.strip().splitlines()[-1])
            model = phasenet_dir / 'model' / '190703-214543'
            if phasenet_dir.is_dir():
                log.debug('_resolve_phasenet_paths: using repo-local script %s', _local_pn)
                return _local_pn, model
        except Exception as exc:
            log.debug('_resolve_phasenet_paths model probe failed: %s', exc)

    # 2. Installed easyQuake package
    probe = (
        'import os, easyQuake;'
        'print(os.path.join(os.path.dirname(easyQuake.__file__), "phasenet"))'
    )
    try:
        result = subprocess.run(
            [python_cmd, '-c', probe],
            capture_output=True, text=True, timeout=15,
        )
        # The easyQuake package prints its banner to stdout; take last line only
        phasenet_dir = Path(result.stdout.strip().splitlines()[-1])
        script = phasenet_dir / 'phasenet_predict.py'
        model = phasenet_dir / 'model' / '190703-214543'
        if script.exists():
            return script, model
    except Exception as exc:
        log.debug('_resolve_phasenet_paths probe failed: %s', exc)
    # Fallback: installed site-packages path derived from python_cmd prefix
    prefix = Path(python_cmd).parent.parent
    for candidate in prefix.rglob('easyQuake/phasenet/phasenet_predict.py'):
        return candidate, candidate.parent / 'model' / '190703-214543'
    return None, None


class EasyQuakePhaseNetPredictor:
    """TF-based PhaseNet predictor using easyQuake's phasenet_predict.py."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        python_cmd: Optional[str] = None,
        threshold: float = 0.3,
        p_threshold: Optional[float] = None,
        s_threshold: Optional[float] = None,
        device: str = 'cpu',  # TF manages GPU internally
    ):
        self.python_cmd = python_cmd or _DEFAULT_PYTHON
        self.threshold = threshold
        self.p_threshold = p_threshold if p_threshold is not None else threshold
        self.s_threshold = s_threshold if s_threshold is not None else self.p_threshold
        self.device = device
        self._loaded = False
        # Resolved at load_model() time
        self._phasenet_predict: Optional[Path] = None
        self._model_dir: Optional[Path] = Path(model_dir) if model_dir else None
        # Accumulated (net,sta,loc) -> obspy.Stream for batch prediction
        self._pending: Dict[Tuple[str, str, str], object] = {}

    @property
    def model_dir(self) -> Optional[Path]:
        return self._model_dir

    @property
    def _author_tag(self) -> str:
        name = self._model_dir.name if self._model_dir else 'unknown'
        return f'sceasyquake/EasyQuakePhaseNet:{name}'

    def load_model(self) -> bool:
        """Resolve phasenet_predict.py path from the target interpreter and validate."""
        if not Path(self.python_cmd).exists():
            log.error('EasyQuakePhaseNet: python not found at %s', self.python_cmd)
            return False

        script, default_model = _resolve_phasenet_paths(self.python_cmd)
        if script is None or not script.exists():
            log.error('EasyQuakePhaseNet: could not locate phasenet_predict.py '
                      'in the easyQuake package under %s', self.python_cmd)
            return False
        self._phasenet_predict = script

        if self._model_dir is None:
            self._model_dir = default_model
        if not self._model_dir or not self._model_dir.exists():
            log.error('EasyQuakePhaseNet: model dir not found at %s', self._model_dir)
            return False

        self._loaded = True
        log.info('EasyQuakePhaseNet: ready  script=%s  model=%s',
                 self._phasenet_predict, self._model_dir)
        return True

    # ------------------------------------------------------------------
    # Single-station predict (streaming API – one subprocess per call)
    # ------------------------------------------------------------------

    def predict(self, stream_or_trace) -> List[dict]:
        """Run prediction on a single ObsPy Stream/Trace.  Spawns one subprocess."""
        if not self._loaded:
            log.warning('EasyQuakePhaseNet: model not loaded; call load_model() first')
            return []
        try:
            from obspy import Stream as ObspyStream, Trace as ObspyTrace
            if isinstance(stream_or_trace, ObspyTrace):
                st = ObspyStream(traces=[stream_or_trace])
            else:
                st = stream_or_trace
        except ImportError:
            return []
        if len(st) == 0:
            return []
        net = st[0].stats.network
        sta = st[0].stats.station
        loc = st[0].stats.location
        return self._run_subprocess({(net, sta, loc): st})

    # ------------------------------------------------------------------
    # Batch API (used by benchmark for efficiency)
    # ------------------------------------------------------------------

    def add_stream(self, key: Tuple[str, str, str], stream) -> None:
        """Accumulate a stream for batch prediction."""
        self._pending[key] = stream

    def run_batch(self) -> List[dict]:
        """Process all accumulated streams in one subprocess call."""
        if not self._pending:
            return []
        picks = self._run_subprocess(dict(self._pending))
        self._pending.clear()
        return picks

    def predict_multi(self, keys_and_streams: list) -> List[dict]:
        """Batch inference (benchmark compatibility API)."""
        if not keys_and_streams:
            return []
        streams = {key: st for key, st in keys_and_streams}
        return self._run_subprocess(streams)

    # ------------------------------------------------------------------
    # Internal: subprocess execution
    # ------------------------------------------------------------------

    # Map channel-suffix → component index (E=0, N=1, Z=2)
    _COMP_ORDER = {'E': 0, '2': 0, '3': 0, 'N': 1, '1': 1, 'Z': 2}

    # PhaseNet's fixed input window: 3000 samples at 100 Hz = 30 seconds
    _WINDOW_S = 30

    def _run_subprocess(self, streams: Dict[Tuple[str, str, str], object]) -> List[dict]:
        """Write 3-component MSEED files and run phasenet_predict once.

        data_reader.py expects the data_list to have one line per station with
        three space-separated absolute paths: ``E_path N_path Z_path``.
        ``read_mseed(fname)`` opens each file individually and sorts channels by
        their last character using comp2idx = {'E':0,'N':1,'Z':2,...}.
        station_id in the output comes from ``E_path.split('.')[1]``.

        Streams longer than 30 s are split into non-overlapping 30-second
        chunks so that PhaseNet covers the entire window (its DataReader
        truncates each input to 3000 samples at 100 Hz = 30 s).
        """
        if not streams:
            return []

        with tempfile.TemporaryDirectory(prefix='sceasyquake_pnet_') as tmpdir:
            tmp = Path(tmpdir)
            mseed_dir = tmp / 'mseed'
            mseed_dir.mkdir()
            result_dir = tmp / 'results'
            result_dir.mkdir()

            # data_list lines: "E_abs_path N_abs_path Z_abs_path"
            data_list_lines: List[str] = []

            for (net, sta, loc), st in streams.items():
                try:
                    from obspy import Stream as _OStream, UTCDateTime as _UTC
                    st = st.copy()
                    st.merge(method=1, fill_value=0)
                    if len(st) == 0:
                        continue

                    # Sort traces into component slots [E/2, N/1, Z]
                    comp_traces: List[Optional[object]] = [None, None, None]
                    for tr in st:
                        suffix = tr.stats.channel[-1].upper()
                        idx = self._COMP_ORDER.get(suffix)
                        if idx is None or comp_traces[idx] is not None:
                            continue
                        comp_traces[idx] = tr

                    z_tr = comp_traces[2]
                    if z_tr is None:
                        log.debug('EasyQuakePhaseNet: no Z component for %s.%s, skipping', net, sta)
                        continue
                    # Fill missing components by duplicating Z
                    for i in range(3):
                        if comp_traces[i] is None:
                            comp_traces[i] = z_tr

                    # Split into 30-second chunks; each becomes one data_list entry
                    t_start = min(tr.stats.starttime for tr in comp_traces)
                    t_end   = max(tr.stats.endtime   for tr in comp_traces)
                    chunk_idx = 0
                    t = t_start
                    while t < t_end:
                        t_chunk_end = t + self._WINDOW_S
                        chunk_comp_paths: List[Optional[Path]] = [None, None, None]
                        for i, tr in enumerate(comp_traces):
                            sl = _OStream([tr]).slice(starttime=t, endtime=t_chunk_end)
                            if not sl or len(sl[0].data) < 10:
                                continue
                            ch = tr.stats.channel
                            fname = f'{net}.{sta}.{loc}.{ch}.chunk{chunk_idx:04d}.mseed'
                            fpath = mseed_dir / fname
                            sl.write(str(fpath), format='MSEED')
                            chunk_comp_paths[i] = fpath

                        z_path = chunk_comp_paths[2]
                        if z_path is not None:
                            for i in range(3):
                                if chunk_comp_paths[i] is None:
                                    chunk_comp_paths[i] = z_path
                            data_list_lines.append(
                                f'{chunk_comp_paths[0]} {chunk_comp_paths[1]} {chunk_comp_paths[2]}'
                            )
                        t = t_chunk_end
                        chunk_idx += 1

                except Exception as exc:
                    log.debug('EasyQuakePhaseNet: failed writing MSEED for %s.%s: %s',
                              net, sta, exc)

            if not data_list_lines:
                return []

            data_list_path = tmp / 'data_list.txt'
            with open(data_list_path, 'w') as f:
                f.write('\n'.join(data_list_lines) + '\n')

            result_file = 'picks.csv'
            cmd = [
                self.python_cmd,
                str(self._phasenet_predict),
                '--model', str(self._model_dir),
                '--data_list', str(data_list_path),
                '--format', 'mseed',
                '--result_dir', str(result_dir),
                '--result_fname', result_file,
                '--min_p_prob', str(self.p_threshold),
                '--min_s_prob', str(self.s_threshold),
                '--mpd', '20',  # Minimum peak distance: 20 samples = 0.2s at 100 Hz (matches SeisBench default)
                '--amplitude',
            ]
            n_stations = len(data_list_lines)
            log.debug('EasyQuakePhaseNet: running subprocess for %d station(s)',
                      n_stations)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env={**os.environ,
                         'CUDA_VISIBLE_DEVICES': '-1'
                             if self.device == 'cpu'
                             else os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                         'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                         'TF_GPU_ALLOCATOR': 'cuda_malloc_async'},

                )
                if result.returncode != 0:
                    log.warning('EasyQuakePhaseNet: subprocess returned %d\nSTDERR: %s',
                                result.returncode, result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                log.warning('EasyQuakePhaseNet: subprocess timed out after 300s')
                return []
            except Exception as exc:
                log.warning('EasyQuakePhaseNet: subprocess error: %s', exc)
                return []

            # Parse the output CSV (space-separated, no header):
            # network  station_id  chan_pick  phase_type  phase_time
            picks_file = result_dir / result_file
            if not picks_file.exists():
                log.debug('EasyQuakePhaseNet: no picks file produced')
                return []

            picks = []
            try:
                with open(picks_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        network, station_id, chan_pick, phase_type, phase_time_str = parts[:5]
                        try:
                            # phase_time is ISO string: e.g. 2026-04-06T21:58:18.500
                            dt = datetime.fromisoformat(
                                phase_time_str.replace('Z', '+00:00'))
                            epoch = dt.replace(tzinfo=timezone.utc).timestamp()
                        except Exception:
                            continue
                        phase_type = phase_type.upper()
                        thresh = self.p_threshold if phase_type == 'P' else self.s_threshold
                        # Read per-pick probability from 6th column if available
                        try:
                            prob = float(parts[5]) if len(parts) >= 6 else thresh
                        except (ValueError, IndexError):
                            prob = thresh
                        picks.append({
                            'network': network,
                            'station': station_id,
                            'location': '',
                            'channel': chan_pick,
                            'time': epoch,
                            'phase': phase_type,
                            'probability': prob,
                            'author': self._author_tag,
                        })
            except Exception as exc:
                log.warning('EasyQuakePhaseNet: failed parsing picks: %s', exc)

            log.debug('EasyQuakePhaseNet: %d picks from %d station(s)',
                      len(picks), n_stations)
            return picks
