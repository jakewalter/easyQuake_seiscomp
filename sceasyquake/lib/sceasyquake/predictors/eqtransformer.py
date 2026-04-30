"""EQTransformer predictor plugin for sceasyquake.

Uses the EQTransformer architecture (Mousavi et al. 2020) via:
- easyQuake.seisbench.EQTransformer  (bundled, preferred)
- seisbench.models.EQTransformer     (system seisbench, fallback)

Pretrained weight sets (picker.pretrained):
    original   – original Mousavi et al. weights (default)
    ethz       – ETH Zürich network
    scedc      – Southern California Seismic Data Center
    geofon     – GEOFON network
    instance   – INGV Italian network
    stead      – STEAD benchmark dataset

Custom weights: set picker.model_path to the directory produced by
seisbench's Model.save(), containing <name>.pt + <name>.json.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


class EQTransformerPredictor:
    def __init__(
        self,
        pretrained: str = 'original',
        model_path: Optional[str] = None,
        threshold: float = 0.3,
        min_distance: float = 0.2,
        device: str = 'cpu',
    ):
        """Create an EQTransformerPredictor.

        Parameters
        ----------
        pretrained  : pretrained weight set name (e.g. 'original', 'ethz')
        model_path  : path to custom weights directory (overrides pretrained)
        threshold   : probability threshold for accepting a pick
        min_distance: minimum separation between picks in seconds
        device      : 'cpu' or 'cuda'
        """
        self.pretrained = pretrained or 'original'
        self.model_path = model_path or None
        self.threshold = threshold
        self.min_distance = min_distance
        self.device = device
        self._loaded_model = None
        self._sbm_module = None  # seisbench module used to load model

    @property
    def _author_tag(self) -> str:
        """Human-readable author string identifying the model and weights."""
        import os
        if self.model_path:
            weights = os.path.basename(self.model_path.rstrip('/\\'))
        else:
            weights = self.pretrained
        return f'sceasyquake/EQTransformer:{weights}'

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load and cache the EQTransformer model onto the target device."""
        sbm = self._import_backend()
        if sbm is None:
            raise RuntimeError(
                'No seisbench-compatible backend found for EQTransformer. '
                'Install easyQuake or seisbench.'
            )
        self._sbm_module = sbm
        try:
            import torch
            model = self._load_from_backend(sbm)
            if model is None:
                raise RuntimeError('_load_from_backend returned None')
            model.eval()
            if self.device == 'cuda':
                try:
                    model = model.cuda()
                    log.info('EQTransformer loaded on CUDA')
                except Exception as exc:
                    log.warning('CUDA unavailable, falling back to CPU: %s', exc)
                # Set thread counts separately — these can fail if another PyTorch
                # model already ran in this process (interop threads are locked after
                # first use).  Failure here does not affect CUDA availability.
                try:
                    torch.set_num_threads(4)
                except Exception:
                    pass
                try:
                    torch.set_num_interop_threads(2)
                except Exception:
                    pass
            self._loaded_model = model
            log.info('EQTransformer ready (pretrained=%s, device=%s)',
                     self.model_path or self.pretrained, self.device)
            return True
        except Exception as exc:
            log.exception('Failed to load EQTransformer model: %s', exc)
            self._loaded_model = None
            return False

    def _load_from_backend(self, sbm):
        """Load the model, falling back to seisbench.models on network errors."""
        import requests
        import torch  # noqa: F401 – ensure torch is importable before we call cuda()
        def _try_load(mod):
            if self.model_path:
                log.info('Loading EQTransformer from custom path: %s', self.model_path)
                return mod.EQTransformer.load(self.model_path)
            else:
                log.info('Loading EQTransformer pretrained=%s', self.pretrained)
                return mod.EQTransformer.from_pretrained(self.pretrained)

        try:
            return _try_load(sbm)
        except (requests.exceptions.ConnectionError, OSError) as net_exc:
            log.warning(
                'EQTransformer load via %s failed with network error: %s — '
                'retrying with seisbench.models',
                self._sbm_module, net_exc,
            )
            # Try system seisbench as fallback.
            try:
                import seisbench.models as sbm_sys
                self._sbm_module = sbm_sys
                log.info('EQTransformerPredictor: falling back to seisbench.models')
                return _try_load(sbm_sys)
            except Exception as fb_exc:
                log.error('Fallback seisbench.models also failed: %s', fb_exc)
                raise fb_exc

    @staticmethod
    def _import_backend():
        """Try easyQuake.seisbench first, then system seisbench."""
        for module_path in ('easyQuake.seisbench', 'seisbench.models'):
            try:
                if module_path == 'easyQuake.seisbench':
                    import easyQuake.seisbench as mod  # type: ignore
                else:
                    import seisbench.models as mod  # type: ignore
                # Verify the class exists
                if not hasattr(mod, 'EQTransformer'):
                    continue
                log.info('EQTransformerPredictor: using %s', module_path)
                return mod
            except Exception as exc:
                log.debug('Could not import %s: %s', module_path, exc)
        return None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, stream_or_trace) -> List[Dict[str, Any]]:
        """Run EQTransformer on an ObsPy Trace or Stream."""
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
        """Run annotate() and convert probability traces into pick dicts."""
        if self._loaded_model is None:
            log.warning('EQTransformer model not loaded; call load_model() first')
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
                    log.debug('EQTransformer: dropped %d trace(s) below minimum SR (%.0f Hz)',
                              _drop_count, _model_sr / 4.0)
                from obspy import Stream as _ObspyStream
                st = _ObspyStream(traces=_keep)
                if not st:
                    return []
        except Exception as _exc:
            log.warning('EQTransformer: SR normalisation failed: %s', _exc)

        # Build original channel map before annotate() replaces channel names
        orig_chan_map = {}
        for tr in st:
            key = (tr.stats.network, tr.stats.station, tr.stats.location)
            if key not in orig_chan_map:
                orig_chan_map[key] = tr.stats.channel

        with torch.no_grad():
            try:
                annotations = self._loaded_model.annotate(st)
            except Exception as exc:
                log.warning('EQTransformer annotate() failed: %s', exc)
                return []

        picks = []
        for ann_tr in annotations:
            chan = ann_tr.stats.channel
            # EQTransformer produces: *_P, *_S, *_Detection
            if chan.endswith('_P'):
                phase = 'P'
            elif chan.endswith('_S'):
                phase = 'S'
            else:
                continue  # skip detection channel

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
                    'method': 'EQTransformer',
                    'author': self._author_tag,
                })
        return picks
