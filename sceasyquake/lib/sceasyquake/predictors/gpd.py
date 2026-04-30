"""GPD (Generalized Phase Detection) predictor plugin for sceasyquake.

Uses the GPD architecture (Ross et al. 2018) via:
- easyQuake.seisbench.GPD  (bundled, preferred)
- seisbench.models.GPD     (system seisbench, fallback)

The default detection threshold is **0.994**, matching the value used in the
original easyQuake pipeline (gpd_predict/gpd_predict.py, ``min_proba = 0.994``).

Pretrained weight sets (picker.pretrained):
    original   – original Ross et al. weights (default)
    ethz       – ETH Zürich network (if available in seisbench)
    scedc      – Southern California Seismic Data Center
    geofon     – GEOFON network

Custom weights (picker.model_path):
    Point to either:
    • A directory containing ``<name>.pt`` + ``<name>.json`` (seisbench format,
      produced by ``GPD.save()``).  Loaded via ``GPD.load(path)``.
    • A raw PyTorch state-dict file (``*.pt``).  Loaded via
      ``GPD().load_state_dict(torch.load(path))``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


class GPDPredictor:
    def __init__(
        self,
        pretrained: str = 'original',
        model_path: Optional[str] = None,
        threshold: float = 0.994,
        min_distance: float = 0.2,
        device: str = 'cpu',
    ):
        """Create a GPDPredictor.

        Parameters
        ----------
        pretrained   : pretrained weight set name (e.g. 'original', 'scedc')
        model_path   : path to custom weights (directory or .pt file).
                       Overrides *pretrained* when set.
        threshold    : probability threshold – default 0.994 matches easyQuake.
        min_distance : minimum separation between picks in seconds.
        device       : 'cpu' or 'cuda'.
        """
        self.pretrained = pretrained or 'original'
        self.model_path = model_path or None
        self.threshold = threshold
        self.min_distance = min_distance
        self.device = device
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
        return f'sceasyquake/GPD:{weights}'

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load and cache the GPD model onto the target device."""
        sbm = self._import_backend()
        if sbm is None:
            raise RuntimeError(
                'No seisbench-compatible backend found for GPD. '
                'Install easyQuake or seisbench.'
            )
        self._sbm_module = sbm
        try:
            import torch
            model = self._load_weights(sbm)
            model.eval()
            if self.device == 'cuda':
                try:
                    model = model.cuda()
                    torch.set_num_threads(4)
                    torch.set_num_interop_threads(2)
                    log.info('GPD loaded on CUDA')
                except Exception as exc:
                    log.warning('CUDA unavailable, falling back to CPU: %s', exc)
            self._loaded_model = model
            log.info('GPD ready (weights=%s, device=%s, threshold=%.4f)',
                     self.model_path or self.pretrained, self.device, self.threshold)
            return True
        except Exception as exc:
            log.exception('Failed to load GPD model: %s', exc)
            self._loaded_model = None
            return False

    def _load_weights(self, sbm):
        """Resolve model_path / pretrained into a GPD instance."""
        import torch
        if self.model_path:
            p = Path(self.model_path)
            if p.suffix == '.pt' and p.is_file():
                # Raw state-dict file (easyQuake style)
                log.info('GPD: loading state dict from %s', p)
                model = sbm.GPD()
                state = torch.load(str(p), map_location='cpu')
                model.load_state_dict(state)
                return model
            else:
                # Seisbench-format directory (has .pt + .json)
                log.info('GPD: loading seisbench weights from %s', p)
                return sbm.GPD.load(str(p))
        log.info('GPD: from_pretrained(%s)', self.pretrained)
        return sbm.GPD.from_pretrained(self.pretrained)

    @staticmethod
    def _import_backend():
        """Try easyQuake.seisbench first, then system seisbench."""
        for module_path in ('easyQuake.seisbench', 'seisbench.models'):
            try:
                if module_path == 'easyQuake.seisbench':
                    import easyQuake.seisbench as mod  # type: ignore
                else:
                    import seisbench.models as mod  # type: ignore
                if not hasattr(mod, 'GPD'):
                    continue
                log.info('GPDPredictor: using %s', module_path)
                return mod
            except Exception as exc:
                log.debug('Could not import %s: %s', module_path, exc)
        return None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, stream_or_trace) -> List[Dict[str, Any]]:
        """Run GPD on an ObsPy Trace or Stream."""
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
        """Run annotate() and convert probability traces into pick dicts.

        GPD uses a point-output model; annotate() still returns per-sample
        probability traces labelled ``GPD_P``, ``GPD_S`` (and ``GPD_N`` which
        we skip).  We apply find_peaks with the 0.994 threshold (or whatever
        the user set) to extract discrete picks.
        """
        if self._loaded_model is None:
            log.warning('GPD model not loaded; call load_model() first')
            return []
        import torch
        try:
            st = st.copy()
            st.merge(method=1, fill_value=0)
        except Exception:
            pass

        # Resample to model's expected SR before annotate() to avoid Nyquist
        # violations when the batch contains CH (40 Hz) or LH (1 Hz) traces.
        try:
            _model_sr = getattr(self._loaded_model, 'sampling_rate', None)
            if _model_sr:
                _keep = []
                _drop_count = 0
                for _tr in st:
                    _sr = _tr.stats.sampling_rate
                    if abs(_sr - _model_sr) < 0.5:
                        _keep.append(_tr)
                    elif _sr >= _model_sr / 4.0:
                        _tr = _tr.copy()
                        _tr.resample(_model_sr)
                        _keep.append(_tr)
                    else:
                        _drop_count += 1
                if _drop_count:
                    log.debug('GPD: dropped %d trace(s) below minimum SR (%.0f Hz)',
                              _drop_count, _model_sr / 4.0)
                from obspy import Stream as _ObspyStream
                st = _ObspyStream(traces=_keep)
                if not st:
                    return []
        except Exception as _exc:
            log.warning('GPD: SR normalisation failed: %s', _exc)

        orig_chan_map = {}
        for tr in st:
            key = (tr.stats.network, tr.stats.station, tr.stats.location)
            if key not in orig_chan_map:
                orig_chan_map[key] = tr.stats.channel

        with torch.no_grad():
            try:
                annotations = self._loaded_model.annotate(st)
            except Exception as exc:
                log.warning('GPD annotate() failed: %s', exc)
                return []

        picks = []
        for ann_tr in annotations:
            chan = ann_tr.stats.channel
            # GPD output channels: GPD_P, GPD_S, GPD_N
            if chan.endswith('_P'):
                phase = 'P'
            elif chan.endswith('_S'):
                phase = 'S'
            else:
                continue  # skip noise channel

            sr = ann_tr.stats.sampling_rate or 1.0
            probs = ann_tr.data.astype(float)
            min_dist_samples = max(1, int(self.min_distance * sr))
            peaks, props = find_peaks(
                probs, height=self.threshold, distance=min_dist_samples
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
                    'method': 'GPD',
                    'author': self._author_tag,
                })
        return picks
