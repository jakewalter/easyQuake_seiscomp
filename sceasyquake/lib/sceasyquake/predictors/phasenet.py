"""PhaseNet predictor plugin

This module provides `PhaseNetPredictor`, which is a thin wrapper that:
- attempts to load a PhaseNet backend (PhaseNet pip package, easyQuake PhaseNet wrapper, or SeisBench PhaseNet)
- runs inference on a short waveform (ObsPy Trace) to produce a probability trace
- applies a configurable peak-finding & thresholding postprocessing to convert
  probability traces into pick times and probabilities

The backend loading is best-effort; if no supported backend is available the
predictor can run in a `stub` mode for testing and demonstration.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import numpy as np
from obspy import UTCDateTime, Trace
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


class PhaseNetPredictor:
    def __init__(self, backend: str = 'auto', model_path: Optional[str] = None, threshold: float = 0.5, p_threshold: Optional[float] = None, s_threshold: Optional[float] = None, min_distance: float = 0.2, device: str = 'cuda'):
        """
        - backend: 'auto', 'phasenet', 'easyquake', or 'stub'
        - model_path: path to model weights (backend-dependent)
        - threshold: default probability threshold for all phases
        - p_threshold: P-pick threshold; overrides threshold if set
        - s_threshold: S-pick threshold; overrides threshold if set
        - min_distance: minimum separation between picks in seconds
        - device: 'cuda' or 'cpu'
        """
        self.backend = backend
        self.model_path = model_path
        self.threshold = threshold
        self.p_threshold = p_threshold if p_threshold is not None else threshold
        self.s_threshold = s_threshold if s_threshold is not None else self.p_threshold
        self.min_distance = min_distance
        self.device = device
        self._model = None
        self._loaded_model = None
        if backend != 'stub':
            try:
                self._attempt_load_backend()
            except Exception as e:
                log.warning('Failed to load requested backend %s: %s. Falling back to stub.', backend, e)
                self._model = None

    @property
    def _author_tag(self) -> str:
        """Human-readable author string identifying the model and weights."""
        import os
        if self.model_path:
            weights = os.path.basename(self.model_path.rstrip('/\\'))
        elif self.backend in ('easyquake', 'phasenet'):
            weights = self.backend
        elif self.backend == 'stub':
            weights = 'stub'
        else:
            weights = 'stead'
        return f'sceasyquake/PhaseNet:{weights}'

    def load_model(self):
        """Load and cache the model into memory (GPU if requested). This
        should be called once on startup to keep the model resident on the GPU
        and avoid reloading per-inference.
        """
        if self._model is None:
            raise RuntimeError('No backend available to load model')
        name, obj = self._model
        log.info('Loading PhaseNet model for backend %s', name)
        try:
            if name == 'seisbench':
                sbm = obj
                model = sbm.PhaseNet.from_pretrained('diting' if self.model_path == 'diting' else 'stead')
                model.eval()
                if self.device == 'cuda':
                    try:
                        import torch
                        model = model.cuda()
                        # Limit CPU threads when GPU is active — PyTorch defaults
                        # to spawning one thread per CPU core, which wastes
                        # resources when inference runs on the GPU.
                        torch.set_num_threads(4)
                        torch.set_num_interop_threads(2)
                    except Exception:
                        pass
                self._loaded_model = model
                log.info('SeisBench PhaseNet loaded (pretrained=stead, device=%s)', self.device)
                return True
            if name == 'easyquake':
                pnmod = getattr(obj, 'phase_net', None)
                if pnmod is None:
                    raise RuntimeError('easyQuake phase_net module missing')
                # try load_model or similar functions, but be robust
                if hasattr(pnmod, 'load_model'):
                    self._loaded_model = pnmod.load_model(self.model_path, device=self.device)
                elif hasattr(pnmod, 'load'):
                    self._loaded_model = pnmod.load(self.model_path, device=self.device)
                else:
                    # Some easyQuake wrappers don't require explicit load; store pnmod
                    self._loaded_model = pnmod
                log.info('PhaseNet model loaded into memory')
                return True
            elif name == 'phasenet':
                # phasenet package may expose model loading
                if hasattr(obj, 'load_model'):
                    self._loaded_model = obj.load_model(self.model_path, device=self.device)
                    log.info('PhaseNet model loaded (phasenet package)')
                    return True
                self._loaded_model = obj
                log.info('Using phasenet package without explicit load')
                return True
            else:
                log.warning('No explicit model load supported for backend %s', name)
                self._loaded_model = obj
                return True
        except Exception as e:
            log.exception('Failed to load PhaseNet model: %s', e)
            self._loaded_model = None
            return False

    def _attempt_load_backend(self):
        """Try to import a PhaseNet backend. This is best-effort and may require
        adjusting based on the installed package. Supported attempts:
        - `phasenet` pip package (generic)
        - `easyQuake` PhaseNet wrapper if `easyQuake` package is installed
        - (future) SeisBench PhaseNet model
        """
        if self.backend == 'auto' or self.backend == 'phasenet':
            try:
                import phasenet as pn  # type: ignore
                log.info('Using phasenet package as backend')
                self._model = ('phasenet', pn)
                return
            except Exception:
                log.debug('phasenet package not available')
        if self.backend == 'auto' or self.backend == 'easyquake':
            try:
                import easyQuake  # type: ignore
                # Only use easyquake backend if phase_net submodule is present
                if getattr(easyQuake, 'phase_net', None) is not None:
                    log.info('Using easyQuake package as backend')
                    self._model = ('easyquake', easyQuake)
                    return
                else:
                    log.debug('easyQuake found but phase_net module missing; trying next backend')
            except Exception:
                log.debug('easyQuake package not available')
        if self.backend in ('auto', 'seisbench', 'phasenet'):
            try:
                import seisbench.models as sbm  # type: ignore
                log.info('Using SeisBench PhaseNet as backend')
                self._model = ('seisbench', sbm)
                return
            except Exception as e:
                log.debug('seisbench not available: %s', e)
        raise RuntimeError('No supported PhaseNet backend available')

    def _backend_predict_probs(self, trace: Trace) -> np.ndarray:
        """Run backend model to obtain per-sample probability vector for P picks.
        This method converts an ObsPy Trace into the input expected by the
        backend and returns a numpy 1D array of probabilities of same length.

        Backend adapters should be implemented as needed; the default behaviour
        raises if no backend is present.
        """
        if self._model is None:
            raise RuntimeError('No PhaseNet backend loaded')
        name, obj = self._model
        data = trace.data.astype(np.float32)
        sr = trace.stats.sampling_rate
        if name == 'phasenet':
            # Attempt typical usage: phasenet.predict or phasenet.core.predict
            try:
                # Some phasenet packages have a `predict` function that accepts raw waveform
                # Prefer using a loaded model's predict() method if available
                if self._loaded_model is not None and hasattr(self._loaded_model, 'predict'):
                    probs = self._loaded_model.predict(data, sr)
                else:
                    probs = obj.predict(data, sr)
                return np.asarray(probs, dtype=float)
            except Exception:
                # If interface differs, raise informative error
                raise RuntimeError('phasenet package present but no compatible predict() found')
        elif name == 'seisbench':
            # SeisBench PhaseNet: annotate() returns a stream with prob traces
            try:
                import torch
                from obspy import Stream
                # SeisBench needs a 3-component stream; we annotate single trace
                st = Stream(traces=[trace])
                with torch.no_grad():
                    annotations = self._loaded_model.annotate(st)
                # annotations is an obspy Stream; find the P-phase trace
                p_trace = None
                for ann in annotations:
                    if ann.stats.channel.endswith('_P'):
                        p_trace = ann
                        break
                if p_trace is None and len(annotations) > 0:
                    p_trace = annotations[0]
                if p_trace is None:
                    return np.array([], dtype=float)
                # Resample to match input trace length if needed
                probs = p_trace.data.astype(np.float64)
                return probs
            except Exception as e:
                raise RuntimeError(f'seisbench predict error: {e}')
        elif name == 'easyquake':
            # easyQuake likely has functions for PhaseNet prediction; attempt common names
            try:
                # Example: easyQuake.phase_net.predict_trace(trace, model_path=...)
                pnmod = getattr(obj, 'phase_net', None)
                if pnmod is None:
                    raise RuntimeError('easyQuake package found but phase_net module not available')

                # Try multiple common function names to be robust to API differences
                # Preferred: predict_trace(trace, model_path=..., device=...)
                if hasattr(pnmod, 'predict_trace'):
                    probs = pnmod.predict_trace(trace, model_path=self.model_path, device=self.device)
                    return np.asarray(probs, dtype=float)

                # Older/alternate names
                if hasattr(pnmod, 'predict'):
                    try:
                        probs = pnmod.predict(trace, model_path=self.model_path, device=self.device)
                        return np.asarray(probs, dtype=float)
                    except TypeError:
                        # Try calling with raw array and sampling rate
                        probs = pnmod.predict(trace.data.astype(np.float32), trace.stats.sampling_rate, model_path=self.model_path, device=self.device)
                        return np.asarray(probs, dtype=float)

                if hasattr(pnmod, 'predict_probs'):
                    probs = pnmod.predict_probs(trace)
                    return np.asarray(probs, dtype=float)

                # If none found, raise
                raise RuntimeError('easyQuake phase_net does not expose a compatible predict function')
            except Exception as e:
                raise RuntimeError(f'easyQuake PhaseNet predict error: {e}')
        else:
            raise RuntimeError(f'Unsupported backend name: {name}')

    def _postprocess_probs(self, probs: np.ndarray, trace: Trace) -> List[Dict[str, Any]]:
        """Convert probability trace to pick dicts via peak finding."""
        if probs.size == 0:
            return []
        sr = trace.stats.sampling_rate
        trace_start = trace.stats.starttime
        # smooth probabilities (optional) - keep simple for now
        # find peaks above threshold, with minimum distance
        distance_samples = max(1, int(self.min_distance * sr))
        peaks, props = find_peaks(probs, height=self.threshold, distance=distance_samples)
        picks = []
        for idx, pk in enumerate(peaks):
            time = trace_start + (pk / sr)
            prob = float(props['peak_heights'][idx]) if 'peak_heights' in props else float(probs[pk])
            picks.append({
                'network': getattr(trace, 'network', ''),
                'station': trace.stats.station,
                'location': trace.stats.location if 'location' in trace.stats and trace.stats.location is not None else '',
                'channel': trace.stats.channel,
                'time': time,
                'phase': 'P',
                'probability': prob,
                'method': 'PhaseNet',
                'author': self._author_tag,
            })
        return picks

    def predict(self, stream_or_trace) -> List[Dict[str, Any]]:
        """Run prediction on an ObsPy Trace or Stream and return pick dicts.

        The new worker passes a per-channel obspy.Stream (already buffered to
        buffer_seconds).  Legacy callers may pass a single Trace; both are
        handled.

        SeisBench backend: uses annotate() on the full stream so that the model
        receives a long-enough continuous window (PhaseNet needs >= 3001 samples
        at its native 100 Hz, i.e. ~30 s).

        Other backends (easyquake, phasenet): fall back to single-trace
        probability extraction via _backend_predict_probs() + _postprocess_probs().
        """
        from obspy import Stream as ObspyStream, Trace as ObspyTrace

        # Normalise input
        if isinstance(stream_or_trace, ObspyTrace):
            st = ObspyStream(traces=[stream_or_trace])
        elif isinstance(stream_or_trace, ObspyStream):
            st = stream_or_trace
        else:
            st = ObspyStream(traces=[stream_or_trace])

        if self._model is None and self.backend == 'stub':
            # Stub mode: return a simulated pick at the centre of the first trace
            picks = []
            for tr in st:
                sr = tr.stats.sampling_rate or 1.0
                mid = tr.stats.starttime + (tr.stats.npts / sr) / 2
                picks.append({
                    'network': tr.stats.network,
                    'station': tr.stats.station,
                    'location': tr.stats.location or '',
                    'channel': tr.stats.channel,
                    'time': mid,
                    'phase': 'P',
                    'probability': 0.9,
                    'method': 'PhaseNet-stub',
                    'author': self._author_tag,
                })
            return picks

        name, _ = (self._model or (None, None))

        if name == 'seisbench':
            return self._seisbench_picks(st)

        # For other backends: operate on the first trace only (legacy behaviour)
        if len(st) == 0:
            return []
        trace = st[0]
        probs = self._backend_predict_probs(trace)
        return self._postprocess_probs(probs, trace)

    def predict_multi(self, keys_and_streams: list) -> List[Dict[str, Any]]:
        """Run prediction on multiple channels in a single batch pass.

        Parameters
        ----------
        keys_and_streams : list of (str, obspy.Stream)
            Each element is a ``(channel_key, buffered_stream)`` pair as
            produced by ``StreamWorker._pred_loop``.

        Returns
        -------
        list of pick dicts (same format as ``predict()``).

        For the SeisBench backend all input streams are merged into one
        combined ObsPy Stream so that ``annotate()`` processes every channel
        in a **single GPU forward pass** instead of hundreds of sequential
        calls.  Non-SeisBench backends fall back to sequential ``predict()``
        calls (no regression vs the old per-channel loop).
        """
        if not keys_and_streams:
            return []

        name = self._model[0] if self._model else None

        if name == 'seisbench':
            # Merge all channel streams into one combined Stream for a single
            # annotate() call.  SeisBench handles multi-station/multi-channel
            # streams natively – one GPU pass covers everything.
            from obspy import Stream as ObspyStream
            combined = ObspyStream()
            for _key, st in keys_and_streams:
                try:
                    combined += st.copy()
                except Exception:
                    pass
            if len(combined) == 0:
                return []
            return self._seisbench_picks(combined)

        # Fallback: sequential predict() for non-SeisBench backends
        all_picks: List[Dict[str, Any]] = []
        for _key, st in keys_and_streams:
            try:
                all_picks.extend(self.predict(st))
            except Exception as exc:
                log.warning('predict_multi fallback error for %s: %s', _key, exc)
        return all_picks

    def _seisbench_picks(self, st) -> List[Dict[str, Any]]:
        """Run SeisBench PhaseNet annotate() on a buffered stream and return picks."""
        import torch
        # Merge gappy/overlapping traces before annotating
        try:
            st = st.copy()
            st.merge(method=1, fill_value=0)
        except Exception:
            pass

        # Build a lookup of (net,sta,loc) → original channel code for the input
        # so we can restore the correct channel name from the annotation.
        orig_chan_map = {}
        for tr in st:
            key = (tr.stats.network, tr.stats.station, tr.stats.location)
            orig_chan_map[key] = tr.stats.channel

        with torch.no_grad():
            try:
                annotations = self._loaded_model.annotate(st)
            except Exception as exc:
                log.warning('SeisBench annotate() failed: %s', exc)
                return []

        picks = []
        for ann_tr in annotations:
            chan = ann_tr.stats.channel
            if not (chan.endswith('_P') or chan.endswith('_S')):
                continue
            phase = 'P' if chan.endswith('_P') else 'S'
            sr = ann_tr.stats.sampling_rate or 1.0
            probs = ann_tr.data.astype(float)
            min_dist_samples = max(1, int(self.min_distance * sr))
            phase_threshold = self.p_threshold if phase == 'P' else self.s_threshold
            peaks, props = find_peaks(probs, height=phase_threshold, distance=min_dist_samples)

            # Use original channel from the input stream (SeisBench replaces it
            # with the model name, e.g. 'HH1_P' becomes 'PhaseNet_P')
            nsl_key = (ann_tr.stats.network, ann_tr.stats.station, ann_tr.stats.location)
            orig_cha = orig_chan_map.get(nsl_key, chan[: -(len(phase) + 1)])

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
                    'method': 'PhaseNet-SeisBench',
                    'author': self._author_tag,
                })
        return picks

