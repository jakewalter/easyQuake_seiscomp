"""EasyQuake TF-based GPD predictor.

Wraps easyQuake's gpd_predict.py as a subprocess call, identical in structure
to EasyQuakePhaseNetPredictor.  GPD uses Keras/TF (Ross et al. 2018) and
cannot run in-process alongside PyTorch.

easyQuake 2.0 uses Keras 3 model format (.keras / .h5); the patched
``share/scripts/gpd_predict.py`` loads models via ``keras.models.load_model()``
instead of the old ``model_from_json`` + HDF5-weights approach.

GPD differences from PhaseNet:
- CLI: ``-I infile -O outfile -F model_dir``  (no threshold flags)
- Threshold (min_proba=0.994) is hardcoded in gpd_predict.py
- GPD internally slides its own window over the full stream — no need to
  pre-chunk streams into 30-second slices.
- Input format is identical: data_list one line per station, three
  space-separated absolute MSEED paths (E N Z).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_DEFAULT_PYTHON = '/home/jwalter/anaconda3/envs/easyquake/bin/python'


def _resolve_gpd_paths(python_cmd: str):
    """Return (gpd_predict_path, model_dir) from the target interpreter.

    Prefers the repo-local patched script under sceasyquake/share/scripts/ so
    the repository is self-contained and deployable without modifying the conda
    env.  Falls back to the installed easyQuake package when the local copy
    is absent.
    """
    # 1. Repo-local patched script (preferred — self-contained deployment)
    # Use .resolve() so the symlink from $SEISCOMP_ROOT/lib/python/ is followed
    # before computing parents; otherwise parents[3] lands in SeisComP's lib/.
    _local_scripts = Path(__file__).resolve().parents[3] / 'share' / 'scripts'
    _local_gpd = _local_scripts / 'gpd_predict.py'
    if _local_gpd.exists():
        # Model weights still live inside the easyQuake package
        probe = (
            'import os, inspect;'
            'import easyQuake.gpd_predict.gpd_predict as _m;'
            'd = os.path.dirname(inspect.getfile(_m));'
            'print(d)'
        )
        try:
            result = subprocess.run(
                [python_cmd, '-c', probe],
                capture_output=True, text=True, timeout=15,
            )
            # The easyQuake package prints its banner to stdout; take last line only
            model_dir = Path(result.stdout.strip().splitlines()[-1])
            if model_dir.is_dir():
                log.debug('_resolve_gpd_paths: using repo-local script %s', _local_gpd)
                return _local_gpd, model_dir
        except Exception as exc:
            log.debug('_resolve_gpd_paths model probe failed: %s', exc)

    # 2. Installed easyQuake package
    probe = (
        'import os, inspect;'
        'import easyQuake.gpd_predict.gpd_predict as _m;'
        'd = os.path.dirname(inspect.getfile(_m));'
        'print(d)'
    )
    try:
        result = subprocess.run(
            [python_cmd, '-c', probe],
            capture_output=True, text=True, timeout=15,
        )
        gpd_dir = Path(result.stdout.strip().splitlines()[-1])
        script = gpd_dir / 'gpd_predict.py'
        if script.exists():
            return script, gpd_dir  # model files live in same dir
    except Exception as exc:
        log.debug('_resolve_gpd_paths probe failed: %s', exc)
    # Fallback via prefix scan
    prefix = Path(python_cmd).parent.parent
    for candidate in prefix.rglob('easyQuake/gpd_predict/gpd_predict.py'):
        return candidate, candidate.parent
    return None, None


class EasyQuakeGPDPredictor:
    """TF/Keras GPD predictor using easyQuake's gpd_predict.py."""

    # Map channel-suffix → component index (E=0, N=1, Z=2)
    _COMP_ORDER = {'E': 0, '2': 0, '3': 0, 'N': 1, '1': 1, 'Z': 2}

    def __init__(
        self,
        python_cmd: Optional[str] = None,
        threshold: float = 0.994,   # mirrors gpd_predict.py min_proba
        p_threshold: Optional[float] = None,
        s_threshold: Optional[float] = None,
        device: str = 'cpu',
    ):
        self.python_cmd = python_cmd or _DEFAULT_PYTHON
        self.threshold = threshold
        self.p_threshold = p_threshold if p_threshold is not None else threshold
        self.s_threshold = s_threshold if s_threshold is not None else self.p_threshold
        self.device = device
        self._loaded = False
        self._gpd_predict: Optional[Path] = None
        self._model_dir: Optional[Path] = None
        self._pending: Dict[Tuple[str, str, str], object] = {}

    @property
    def _author_tag(self) -> str:
        return 'sceasyquake/EasyQuakeGPD'

    def load_model(self) -> bool:
        if not Path(self.python_cmd).exists():
            log.error('EasyQuakeGPD: python not found at %s', self.python_cmd)
            return False
        script, model_dir = _resolve_gpd_paths(self.python_cmd)
        if script is None or not script.exists():
            log.error('EasyQuakeGPD: could not locate gpd_predict.py')
            return False
        self._gpd_predict = script
        self._model_dir = model_dir
        self._loaded = True
        log.info('EasyQuakeGPD: ready  script=%s  model=%s', script, model_dir)
        return True

    # ------------------------------------------------------------------
    # Public predict API (same interface as EasyQuakePhaseNetPredictor)
    # ------------------------------------------------------------------

    def predict(self, stream_or_trace) -> List[dict]:
        """Run GPD on a single ObsPy Stream/Trace — one subprocess call."""
        if not self._loaded:
            log.warning('EasyQuakeGPD: not loaded; call load_model() first')
            return []
        try:
            from obspy import Stream as ObspyStream, Trace as ObspyTrace
            if isinstance(stream_or_trace, ObspyTrace):
                st = ObspyStream(traces=[stream_or_trace])
            else:
                st = stream_or_trace
        except ImportError:
            return []
        if not st:
            return []
        net = st[0].stats.network
        sta = st[0].stats.station
        loc = st[0].stats.location
        return self._run_subprocess({(net, sta, loc): st})

    def add_stream(self, key: Tuple[str, str, str], stream) -> None:
        self._pending[key] = stream

    def run_batch(self) -> List[dict]:
        if not self._pending:
            return []
        picks = self._run_subprocess(dict(self._pending))
        self._pending.clear()
        return picks

    def predict_multi(self, keys_and_streams: list) -> List[dict]:
        if not keys_and_streams:
            return []
        return self._run_subprocess({k: s for k, s in keys_and_streams})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_subprocess(self, streams: Dict[Tuple[str, str, str], object]) -> List[dict]:
        """Write component MSEED files, build data_list, run gpd_predict once."""
        if not streams:
            return []

        with tempfile.TemporaryDirectory(prefix='sceasyquake_gpd_') as tmpdir:
            tmp = Path(tmpdir)
            mseed_dir = tmp / 'mseed'
            mseed_dir.mkdir()

            data_list_lines: List[str] = []

            for (net, sta, loc), st in streams.items():
                try:
                    from obspy import Stream as _OStream
                    st = st.copy()
                    st.merge(method=1, fill_value=0)
                    if not st:
                        continue

                    comp_traces: List[Optional[object]] = [None, None, None]
                    for tr in st:
                        suffix = tr.stats.channel[-1].upper()
                        idx = self._COMP_ORDER.get(suffix)
                        if idx is None or comp_traces[idx] is not None:
                            continue
                        comp_traces[idx] = tr

                    z_tr = comp_traces[2]
                    if z_tr is None:
                        log.debug('EasyQuakeGPD: no Z for %s.%s, skipping', net, sta)
                        continue
                    for i in range(3):
                        if comp_traces[i] is None:
                            comp_traces[i] = z_tr

                    comp_paths: List[Path] = []
                    for i, tr in enumerate(comp_traces):
                        ch = tr.stats.channel
                        fpath = mseed_dir / f'{net}.{sta}.{loc}.{ch}.mseed'
                        _OStream([tr]).write(str(fpath), format='MSEED')
                        comp_paths.append(fpath)

                    data_list_lines.append(
                        f'{comp_paths[0]} {comp_paths[1]} {comp_paths[2]}'
                    )
                except Exception as exc:
                    log.debug('EasyQuakeGPD: MSEED write error %s.%s: %s', net, sta, exc)

            if not data_list_lines:
                return []

            infile = tmp / 'data_list.txt'
            outfile = tmp / 'picks.out'
            infile.write_text('\n'.join(data_list_lines) + '\n')

            cmd = [
                self.python_cmd,
                str(self._gpd_predict),
                '-V',
                '-P',       # suppress plotting
                '-I', str(infile),
                '-O', str(outfile),
                '-F', str(self._model_dir),
            ]
            log.debug('EasyQuakeGPD: subprocess for %d station(s)', len(data_list_lines))
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True, timeout=600,
                    env={**os.environ,
                         'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                             if self.device != 'cpu' else '0',
                         'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                         'TF_GPU_ALLOCATOR': 'cuda_malloc_async'},

                )
                if result.returncode != 0:
                    log.warning('EasyQuakeGPD: subprocess exit %d\nSTDERR: %s',
                                result.returncode, result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                log.warning('EasyQuakeGPD: subprocess timed out')
                return []
            except Exception as exc:
                log.warning('EasyQuakeGPD: subprocess error: %s', exc)
                return []

            if not outfile.exists():
                log.debug('EasyQuakeGPD: no picks file produced')
                return []

            # Output format (same as PhaseNet): net sta chan phase time_iso
            picks = []
            try:
                for line in outfile.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    network, station, chan_pick, phase_type, phase_time_str = parts[:5]
                    try:
                        dt = datetime.fromisoformat(phase_time_str.replace('Z', '+00:00'))
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
                        'station': station,
                        'location': '',
                        'channel': chan_pick,
                        'time': epoch,
                        'phase': phase_type,
                        'probability': prob,
                        'author': self._author_tag,
                    })
            except Exception as exc:
                log.warning('EasyQuakeGPD: parse error: %s', exc)

            log.debug('EasyQuakeGPD: %d picks from %d station(s)',
                      len(picks), len(data_list_lines))
            return picks
