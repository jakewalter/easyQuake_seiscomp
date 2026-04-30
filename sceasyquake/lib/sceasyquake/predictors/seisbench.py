"""Generic SeisBench predictor plugin for sceasyquake.

A single entry point that can run **any** SeisBench WaveformModel architecture
(PhaseNet, GPD, EQTransformer, …) with either pretrained or custom weights.
Use this when you want to switch model architecture via config without changing
the picker-specific module, or when loading custom-trained weights.

Config parameters (sceasyquake.cfg):
    picker.backend    = seisbench
    picker.model      = PhaseNet       # PhaseNet | GPD | EQTransformer
    picker.pretrained = stead          # see below
    picker.model_path =                # optional path to custom weights

Role of ``picker.pretrained``
-----------------------------
This parameter serves **two different purposes** depending on ``model_path``:

* **No model_path** (hub weights): ``pretrained`` names the weight set
  downloaded from the SeisBench model hub and used directly for inference,
  e.g. ``stead``, ``original``, ``ethz``.

* **model_path = raw .pt/.pth state-dict**: ``pretrained`` names the base
  architecture template — the model is first constructed via
  ``from_pretrained(pretrained)`` to capture its norm mode, label layout,
  sampling rate, and component order, then the weights are immediately
  replaced by the file in ``model_path``.  The pretrained weights themselves
  are not used at inference time.  Set this to the weight set whose
  architecture settings match how your custom model was trained.

* **model_path = SeisBench directory (.pt + .json)**: ``pretrained`` is
  ignored entirely; all architecture metadata is read from the ``.json``.

Custom weight loading (``model_path`` resolution order)
-------------------------------------------------------
1. **Raw state-dict** – path ends in ``.pt``/``.pth`` and the file exists.
   ``from_pretrained(pretrained)`` builds the model shell; custom weights
   are then applied via ``load_state_dict``.

2. **SeisBench-format directory** – path is a directory (or a base-name with
   a sibling ``.json`` file).  Loaded via ``ModelClass.load(path)``.  This is
   the format produced by ``model.save(path)``.

3. **Pretrained alias** – anything else is passed directly to
   ``ModelClass.from_pretrained(picker.model_path)``, allowing you to use
   alternate pretrained weight names without changing ``picker.pretrained``.

Backend search order
--------------------
``easyQuake.seisbench`` is tried first (it is the bundled, tuned version used
by the rest of easyQuake).  ``seisbench.models`` is used as a fallback.

Supported models and typical pretrained names
---------------------------------------------
PhaseNet     : stead (default), diting, ethz, scedc, geofon, instance, neic
GPD          : original (default), scedc, geofon
EQTransformer: original (default), ethz, scedc, geofon, instance, stead
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from scipy.signal import find_peaks

log = logging.getLogger(__name__)

# Model class names exposed on the backend module
_SUPPORTED_MODELS = ('PhaseNet', 'GPD', 'EQTransformer')


class SeisBenchPredictor:
    def __init__(
        self,
        model: str = 'PhaseNet',
        pretrained: str = 'stead',
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        p_threshold: Optional[float] = None,
        s_threshold: Optional[float] = None,
        min_distance: float = 0.2,
        norm: Optional[str] = None,
        device: str = 'cpu',
        phases: Optional[str] = None,
        label_order: Optional[str] = None,
    ):
        """Create a SeisBenchPredictor.

        Parameters
        ----------
        model        : SeisBench model class name (PhaseNet, GPD, EQTransformer).
        pretrained   : pretrained weight set (e.g. 'stead', 'original', 'ethz').
                       Ignored when *model_path* is provided.
        model_path   : path to custom weights (raw .pt or seisbench directory).
                       When set, *pretrained* is not used.
        threshold    : default probability threshold for all phases.
        p_threshold  : P-pick threshold; overrides *threshold* for P if set.
        s_threshold  : S-pick threshold; overrides *threshold* for S if set.
        phases       : comma/space-separated list of phases to pick, e.g. 'P'
                       or 'P,S'.  When None or empty, all phases are picked.
        min_distance : minimum separation between picks in seconds.
        norm         : seisbench amplitude normalization mode ('std' or 'peak').
                       When None, uses the model's own default.  Custom weights
                       trained with sbg.Normalize(amp_norm_type='peak') need
                       norm='peak' here to obtain correct probabilities.
        label_order  : output channel order the model was trained with, e.g.
                       'PSN'.  When set and different from the backend's native
                       order (seisbench PhaseNet uses 'NPS'), the output layer
                       weights are permuted on load so channels map correctly.
                       Example: a model trained with [P, S, N] output order
                       needs label_order='PSN' when loaded into seisbench.
        device       : 'cpu' or 'cuda'.
        """
        self.model_name = model or 'PhaseNet'
        self.pretrained = pretrained or 'stead'
        self.model_path = model_path or None
        self.threshold = threshold
        self.p_threshold = p_threshold if p_threshold is not None else threshold
        self.s_threshold = s_threshold if s_threshold is not None else self.p_threshold
        self.min_distance = min_distance
        self.norm = norm
        self.device = device
        # Normalise phases into a frozenset of uppercase letters, or None for all
        if phases:
            self._phases = frozenset(p.strip().upper() for p in phases.replace(',', ' ').split() if p.strip())
        else:
            self._phases = None  # None means pick all phases
        self.label_order = label_order.upper().strip() if label_order else None
        self._loaded_model = None
        self._sbm_module = None

    @property
    def _author_tag(self) -> str:
        """Human-readable author string identifying the model and weights."""
        import os
        if self.model_path:
            weights = os.path.basename(self.model_path.rstrip('/\\'))
        else:
            weights = self.pretrained
        return f'sceasyquake/{self.model_name}:{weights}'

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load and cache the model onto the target device.

        Tries each available backend in order (easyQuake.seisbench, then
        seisbench.models).  This allows graceful fallback when the custom
        model's state-dict architecture only matches one of the backends.
        """
        import torch
        backends = self._available_backends()
        if not backends:
            raise RuntimeError(
                'No seisbench-compatible backend found. '
                'Install easyQuake or seisbench.'
            )

        last_exc: Exception | None = None
        for backend_name, sbm in backends:
            if not hasattr(sbm, self.model_name):
                log.debug('Backend %s has no class %s; skipping', backend_name, self.model_name)
                continue
            try:
                model = self._load_weights(sbm)
                model.eval()
                if self.device == 'cuda':
                    try:
                        model = model.cuda()
                        log.info('%s loaded on CUDA via %s', self.model_name, backend_name)
                    except Exception as exc:
                        log.warning('CUDA unavailable, falling back to CPU: %s', exc)
                    # Thread counts are set separately — can fail after parallel work started
                    try:
                        torch.set_num_threads(4)
                    except Exception:
                        pass
                    try:
                        torch.set_num_interop_threads(2)
                    except Exception:
                        pass
                self._loaded_model = model
                self._sbm_module = sbm
                log.info(
                    'SeisBench %s ready (weights=%s, backend=%s, device=%s)',
                    self.model_name,
                    self.model_path or self.pretrained,
                    backend_name,
                    self.device,
                )
                return True
            except Exception as exc:
                log.debug('Backend %s failed to load %s weights: %s; trying next',
                          backend_name, self.model_name, exc)
                last_exc = exc
                continue

        log.exception('Failed to load SeisBench %s with any backend: %s',
                      self.model_name, last_exc)
        self._loaded_model = None
        return False

    def _load_weights(self, sbm):
        """Resolve model_path / pretrained into a model instance."""
        import torch
        model_cls = getattr(sbm, self.model_name)

        if self.model_path:
            p = Path(self.model_path)

            # 1. Raw state-dict .pt / .pth file
            if p.suffix in ('.pt', '.pth') and p.is_file():
                log.info('%s: loading raw state dict from %s', self.model_name, p)
                # Use the pretrained base model to preserve all metadata
                # (norm, labels, sampling_rate, component_order, etc.) that
                # the model was trained against.  The reference inference
                # workflow (oklad_annotate_workflow) loads from_pretrained
                # then overwrites weights — we do the same here so that
                # seisbench's annotate_batch_pre uses the correct settings.
                try:
                    m = model_cls.from_pretrained(self.pretrained)
                    log.info('%s: using %s as base for custom weights', self.model_name, self.pretrained)
                except Exception:
                    # Fall back to bare constructor if pretrained is unavailable
                    init_norm = self.norm if self.norm is not None else 'peak'
                    try:
                        m = model_cls(norm=init_norm)
                    except TypeError:
                        m = model_cls()
                    log.info('%s: pretrained base unavailable; using bare constructor', self.model_name)
                state = torch.load(str(p), map_location='cpu')
                # Reorder output layer if the model's training label order differs
                # from the backend's native label order (seisbench PhaseNet: 'NPS').
                # e.g. a model trained with [P,S,N] outputs needs label_order='PSN'.
                native_labels = getattr(m, 'labels', None)
                if self.label_order and native_labels and self.label_order != native_labels:
                    try:
                        perm = [self.label_order.index(c) for c in native_labels]
                        import torch as _torch
                        state['out.weight'] = state['out.weight'][perm]
                        state['out.bias']   = state['out.bias'][perm]
                        log.info('%s: permuted output layer from %s to %s (perm=%s)',
                                 self.model_name, self.label_order, native_labels, perm)
                    except (ValueError, KeyError) as exc:
                        log.warning('%s: could not apply label_order permutation: %s',
                                    self.model_name, exc)
                m.load_state_dict(state)
                return m

            # 2. SeisBench-format directory or base path with .json sibling
            json_path = p.with_suffix('.json') if p.suffix != '.json' else p
            if p.is_dir() or json_path.exists():
                log.info('%s: loading seisbench format from %s', self.model_name, p)
                return model_cls.load(str(p))

            # 3. Treat as an alternate pretrained name
            log.info('%s: treating model_path as pretrained alias: %s',
                     self.model_name, self.model_path)
            return model_cls.from_pretrained(self.model_path)

        # No custom path – use named pretrained weights
        log.info('%s: from_pretrained(%s)', self.model_name, self.pretrained)
        return model_cls.from_pretrained(self.pretrained)

    @staticmethod
    def _available_backends() -> list:
        """Return an ordered list of (name, module) pairs for available backends."""
        backends = []
        for name, module_path in (
            ('easyQuake.seisbench', 'easyQuake.seisbench'),
            ('seisbench.models', 'seisbench.models'),
        ):
            try:
                if module_path == 'easyQuake.seisbench':
                    import easyQuake.seisbench as mod  # type: ignore
                else:
                    import seisbench.models as mod  # type: ignore
                backends.append((name, mod))
                log.debug('Backend available: %s', name)
            except Exception as exc:
                log.debug('Backend not available: %s (%s)', name, exc)
        return backends

    @staticmethod
    def _import_backend():
        """Try easyQuake.seisbench first, then system seisbench."""
        for module_path in ('easyQuake.seisbench', 'seisbench.models'):
            try:
                if module_path == 'easyQuake.seisbench':
                    import easyQuake.seisbench as mod  # type: ignore
                else:
                    import seisbench.models as mod  # type: ignore
                log.info('SeisBenchPredictor: using %s', module_path)
                return mod
            except Exception as exc:
                log.debug('Could not import %s: %s', module_path, exc)
        return None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, stream_or_trace) -> List[Dict[str, Any]]:
        """Run inference on an ObsPy Trace or Stream."""
        from obspy import Stream as ObspyStream, Trace as ObspyTrace
        if isinstance(stream_or_trace, ObspyTrace):
            st = ObspyStream(traces=[stream_or_trace])
        elif isinstance(stream_or_trace, ObspyStream):
            st = stream_or_trace
        else:
            st = ObspyStream(traces=[stream_or_trace])
        return self._annotate_picks(st)

    def predict_multi(self, keys_and_streams: list) -> List[Dict[str, Any]]:
        """Batch inference: merge all channel streams into one annotate() call."""
        if not keys_and_streams:
            return []
        from obspy import Stream as ObspyStream
        combined = ObspyStream()
        for _key, st in keys_and_streams:
            try:
                combined += st.copy()
            except Exception:
                pass
        if len(combined) == 0:
            return []
        return self._annotate_picks(combined)

    def _annotate_picks(self, st) -> List[Dict[str, Any]]:
        """Run annotate() and extract discrete picks via peak finding.

        Works for PhaseNet (_P / _S channels), GPD (GPD_P / GPD_S / GPD_N),
        and EQTransformer (_P / _S / _Detection) by checking the suffix of
        each annotation channel.  Detection/noise channels are skipped.
        """
        if self._loaded_model is None:
            log.warning('%s model not loaded; call load_model() first', self.model_name)
            return []
        import torch
        # One-time threshold diagnostic (only log first call)
        if not getattr(self, '_thresh_logged', False):
            log.info('[diag] _annotate_picks: p_threshold=%.4f s_threshold=%.4f '
                     'min_distance=%.3f norm=%s phases=%s',
                     self.p_threshold, self.s_threshold, self.min_distance, self.norm,
                     sorted(self._phases) if self._phases else 'all')
            self._thresh_logged = True
        try:
            st = st.copy()
            st.merge(method=1, fill_value=0)
        except Exception:
            pass

        # Resample every trace to the model's expected sampling rate so that
        # seisbench's annotate_batch_pre can apply its bandpass filter without
        # hitting a Nyquist violation.  The native TF/Keras scripts (gpd_predict.py,
        # phasenet_predict.py) always call st.resample(100.0) before running the
        # model; we must do the same here.  Traces whose sample rate is far below
        # the target (< target / 4) are dropped because upsampling from, e.g.,
        # 1 Hz LH data to 100 Hz carries no useful seismic information.
        try:
            _model_sr = getattr(self._loaded_model, 'sampling_rate', None)
            if _model_sr:
                _keep = []
                _drop_count = 0
                for _tr in st:
                    _sr = _tr.stats.sampling_rate
                    if abs(_sr - _model_sr) < 0.5:
                        _keep.append(_tr)          # already at target SR
                    elif _sr >= _model_sr / 4.0:
                        _tr = _tr.copy()
                        _tr.resample(_model_sr)    # upsample (e.g. 40 Hz → 100 Hz)
                        _keep.append(_tr)
                    else:
                        _drop_count += 1           # too low (e.g. 1 Hz LH) — drop
                if _drop_count:
                    log.debug('%s: dropped %d trace(s) below minimum SR (%.0f Hz)',
                              self.model_name, _drop_count, _model_sr / 4.0)
                from obspy import Stream as _ObspyStream
                st = _ObspyStream(traces=_keep)
                if not st:
                    return []
        except Exception as _exc:
            log.warning('%s: SR normalisation failed: %s', self.model_name, _exc)

        # Pre-process: detrend only.
        # Do NOT apply a global per-channel max-abs normalisation here.
        # seisbench's annotate() splits the stream into short windows (e.g. 60 s
        # for PhaseNet) and normalises each window independently, which is
        # identical to what the native TF/Keras scripts do.  A global normalise
        # over a 1-hour trace crushes the entire hour when a single large spike
        # (big earthquake, glitch) sets the maximum — only ~0.1 % of samples
        # would survive above half the peak, silently suppressing all other picks.
        try:
            st.detrend(type='linear')
            st.detrend(type='demean')
        except Exception as _exc:
            log.warning('%s: preprocessing failed: %s', self.model_name, _exc)

        orig_chan_map: dict = {}
        for tr in st:
            key = (tr.stats.network, tr.stats.station, tr.stats.location)
            cha = tr.stats.channel
            existing = orig_chan_map.get(key)
            # Prefer the vertical (Z) channel so picks are assigned to the
            # same stream scrttv and scautopick display (HHZ, not HHE/HHN).
            if existing is None:
                orig_chan_map[key] = cha
            elif cha.endswith('Z') and not existing.endswith('Z'):
                orig_chan_map[key] = cha

        # For any station where only horizontal channels arrived, synthesise the
        # Z channel code (e.g. HHE → HHZ, HH1 → HHZ) so scrttv can match picks.
        for key, cha in orig_chan_map.items():
            if not cha.endswith('Z'):
                orig_chan_map[key] = cha[:-1] + 'Z'

        with torch.no_grad():
            try:
                annotations = self._loaded_model.annotate(st)
            except Exception as exc:
                log.warning('%s annotate() failed: %s', self.model_name, exc)
                return []

        picks = []
        for ann_tr in annotations:
            chan = ann_tr.stats.channel
            if chan.endswith('_P'):
                phase = 'P'
                phase_threshold = self.p_threshold
            elif chan.endswith('_S'):
                phase = 'S'
                phase_threshold = self.s_threshold
            else:
                continue  # skip noise / detection channels
            if self._phases is not None and phase not in self._phases:
                continue  # phase excluded by picker.phases config

            sr = ann_tr.stats.sampling_rate or 1.0
            probs = ann_tr.data.astype(float)
            min_dist_samples = max(1, int(self.min_distance * sr))
            peaks, props = find_peaks(
                probs, height=phase_threshold, distance=min_dist_samples
            )

            nsl_key = (ann_tr.stats.network, ann_tr.stats.station, ann_tr.stats.location)
            orig_cha = orig_chan_map.get(nsl_key, chan[:-(len(phase) + 1)])

            for i, pk in enumerate(peaks):
                t = ann_tr.stats.starttime + (pk / sr)
                prob = float(props['peak_heights'][i])
                picks.append({
                    'network': ann_tr.stats.network,
                    'station': ann_tr.stats.station,
                    'location': ann_tr.stats.location or '',
                    'channel': orig_cha,
                    'time': t,
                    'phase': phase,
                    'probability': prob,
                    'method': f'SeisBench-{self.model_name}',
                    'author': self._author_tag,
                })
        return picks
