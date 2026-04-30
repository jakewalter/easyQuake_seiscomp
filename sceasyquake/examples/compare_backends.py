#!/usr/bin/env python3
"""Compare PhaseNet backends on identical SDS archive data.

Three backends tested:
  A. JGR custom weights, norm='peak', norm_detrend=False  (current live config)
  B. JGR custom weights, norm='peak', norm_detrend=True   (matches training preproc)
  C. PhaseNet pretrained 'stead'                          (reference baseline)

The JGR model was trained with:
  sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type='peak')
so norm_detrend=True (B) should match training exactly.

Usage
-----
    python compare_backends.py
    python compare_backends.py --stations 4O.AT01,4O.BB01,4O.CF01
    python compare_backends.py --model-path /path/to/weights.pth --threshold 0.5
    python compare_backends.py --duration 600
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
)
log = logging.getLogger('compare_backends')
# Quiet the chatty SeedLink client
logging.getLogger('obspy.clients.seedlink').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Defaults (mirror what the live sceasyquake process uses)
# ---------------------------------------------------------------------------
DEFAULT_SEEDLINK  = 'localhost:18000'
DEFAULT_SDS_ROOT  = '/home/jwalter/seiscomp/var/lib/archive'
DEFAULT_STATIONS  = [
    '4O.AT01', '4O.BB01', '4O.BP01', '4O.CF01',
    '4O.CT01', '4O.DB02', '4O.EE01', '4O.GV01',
    '4O.HP01', '4O.LWM1',
]
DEFAULT_MODEL_PATH = '/home/jwalter/JGR_PNet_BestModel_20241220.pth'
DEFAULT_THRESHOLD  = 0.5
DEFAULT_DURATION   = 20 * 60   # seconds


# ---------------------------------------------------------------------------
# Step 1: Fetch data from SDS archive (fast) or live SeedLink (fallback)
# ---------------------------------------------------------------------------

def fetch_from_sds(sds_root: str, stations: list[str],
                   duration_s: int = 1200) -> dict:
    """Read the last *duration_s* seconds of HH? from the SDS archive.

    Returns dict : NET.STA -> obspy.Stream (3-component merged)
    """
    from obspy import UTCDateTime, Stream
    from obspy.clients.filesystem.sds import Client as SDSClient

    client = SDSClient(sds_root)
    endtime   = UTCDateTime()
    starttime = endtime - duration_s

    result = {}
    for nsta in stations:
        parts = nsta.split('.')
        if len(parts) < 2:
            continue
        net, sta = parts[0], parts[1]
        try:
            st = client.get_waveforms(net, sta, '*', 'HH?', starttime, endtime)
            if len(st) == 0:
                log.warning('No SDS data for %s.%s', net, sta)
                continue
            st.merge(method=1, fill_value=0)
            key = f'{net}.{sta}'
            result[key] = st
            dur = max(tr.stats.npts / tr.stats.sampling_rate for tr in st)
            log.info('  %s: %d trace(s), %.0f s from SDS', key, len(st), dur)
        except Exception as exc:
            log.warning('SDS read failed for %s.%s: %s', net, sta, exc)

    log.info('Loaded %d/%d stations from SDS archive', len(result), len(stations))
    return result


# ---------------------------------------------------------------------------
# Step 2: Load predictors
# ---------------------------------------------------------------------------

def _make_seisbench(model_path: str, threshold: float,
                    norm_detrend: bool, label: str) -> object:
    """Return a loaded SeisBenchPredictor configured as requested."""
    import torch
    import seisbench.models as sbm

    init_norm = 'peak'
    m = sbm.PhaseNet(norm=init_norm, norm_detrend=norm_detrend)
    state = torch.load(model_path, map_location='cpu')
    m.load_state_dict(state)
    m.eval()
    try:
        m = m.cuda()
        log.info('%s: loaded on CUDA (norm=%s norm_detrend=%s)', label, init_norm, norm_detrend)
    except Exception:
        log.info('%s: loaded on CPU (norm=%s norm_detrend=%s)', label, init_norm, norm_detrend)

    # Wrap in a thin callable duck-type that exposes predict_multi
    class _Pred:
        def __init__(self, model, thr, label):
            self._m = model
            self.threshold = thr
            self.label = label

        def predict_multi(self, keys_and_streams):
            from obspy import Stream
            from scipy.signal import find_peaks
            import torch

            combined = Stream()
            for _key, st in keys_and_streams:
                try:
                    combined += st.copy()
                except Exception:
                    pass
            if not combined:
                return []
            combined.merge(method=1, fill_value=0)

            orig_chan_map = {}
            for tr in combined:
                key = (tr.stats.network, tr.stats.station, tr.stats.location)
                if key not in orig_chan_map:
                    orig_chan_map[key] = tr.stats.channel

            with torch.no_grad():
                try:
                    annotations = self._m.annotate(combined)
                except Exception as exc:
                    log.warning('%s annotate() failed: %s', self.label, exc)
                    return []

            picks = []
            for ann_tr in annotations:
                chan = ann_tr.stats.channel
                if chan.endswith('_P'):
                    phase = 'P'
                elif chan.endswith('_S'):
                    continue  # JGR is P-only
                else:
                    continue
                sr = ann_tr.stats.sampling_rate or 1.0
                probs = ann_tr.data.astype(float)
                min_dist = max(1, int(1.0 * sr))
                peaks_idx, props = find_peaks(probs, height=self.threshold, distance=min_dist)
                nsl = (ann_tr.stats.network, ann_tr.stats.station, ann_tr.stats.location)
                orig_cha = orig_chan_map.get(nsl, chan[:-2])
                for i, pk in enumerate(peaks_idx):
                    t = ann_tr.stats.starttime + pk / sr
                    picks.append({
                        'network': ann_tr.stats.network,
                        'station': ann_tr.stats.station,
                        'location': ann_tr.stats.location or '',
                        'channel': orig_cha,
                        'time': t,
                        'phase': phase,
                        'probability': float(props['peak_heights'][i]),
                    })
            return picks

    return _Pred(m, threshold, label)


def _make_stead(threshold: float) -> object:
    """Return a SeisBenchPredictor loaded with stead pretrained weights."""
    from sceasyquake.predictors.seisbench import SeisBenchPredictor
    p = SeisBenchPredictor(
        model='PhaseNet',
        pretrained='stead',
        threshold=threshold,
        p_threshold=threshold,
        s_threshold=threshold,
        min_distance=1.0,
        device='cpu',
    )
    ok = p.load_model()
    if not ok:
        log.error('stead model failed to load')
    return p


# ---------------------------------------------------------------------------
# Step 3: Print side-by-side comparison table
# ---------------------------------------------------------------------------

def _pick_counts_by_sta(picks):
    counts = defaultdict(int)
    for p in picks:
        counts[f"{p['network']}.{p['station']}"] += 1
    return counts


def compare_all(label_picks: list[tuple[str, list]], match_window_s: float = 1.0):
    """Print multi-column comparison table.

    label_picks: list of (label_str, picks_list) pairs.
    """
    labels  = [lp[0] for lp in label_picks]
    pickses = [lp[1] for lp in label_picks]

    all_stas = sorted({
        f"{p['network']}.{p['station']}"
        for picks in pickses for p in picks
    })

    col_w = 14
    header = f"{'Station':<18}" + ''.join(f"{l:>{col_w}}" for l in labels)
    print('\n' + '='*len(header))
    print(header)
    print('-'*len(header))

    totals = [0] * len(labels)
    for sta in all_stas:
        counts = [sum(1 for p in picks if f"{p['network']}.{p['station']}" == sta)
                  for picks in pickses]
        for i, c in enumerate(counts):
            totals[i] += c
        print(f"{sta:<18}" + ''.join(f"{c:>{col_w}}" for c in counts))

    print('='*len(header))
    print(f"{'TOTAL':<18}" + ''.join(f"{t:>{col_w}}" for t in totals))
    print()

    for label, picks in label_picks:
        if not picks:
            print(f'{label:<35}  (no picks)')
            continue
        probs = np.array([p['probability'] for p in picks])
        print(f'{label:<35}  n={len(picks):5d}  '
              f'prob mean={probs.mean():.3f}  median={np.median(probs):.3f}  '
              f'min={probs.min():.3f}  max={probs.max():.3f}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sds-root', default=DEFAULT_SDS_ROOT)
    parser.add_argument('--stations', default=','.join(DEFAULT_STATIONS),
                        help='Comma-separated NET.STA list')
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION,
                        help='Seconds of data (default 1200 = 20 min)')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH,
                        help='Path to custom .pth weights')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    stations = [s.strip() for s in args.stations.split(',') if s.strip()]

    # --- Fetch data ---
    log.info('Fetching %d min of data for %d stations from SDS ...',
             args.duration // 60, len(stations))
    buffers = fetch_from_sds(args.sds_root, stations, args.duration)
    if not buffers:
        log.error('No data from SDS archive; aborting')
        sys.exit(1)
    keys_and_streams = list(buffers.items())

    # --- A: JGR, norm=peak, norm_detrend=False (current live config) ---
    log.info('Loading A: JGR custom, norm=peak, norm_detrend=False ...')
    pred_a = _make_seisbench(args.model_path, args.threshold,
                             norm_detrend=False, label='A:JGR-no-detrend')

    # --- B: JGR, norm=peak, norm_detrend=True (matches training) ---
    log.info('Loading B: JGR custom, norm=peak, norm_detrend=True ...')
    pred_b = _make_seisbench(args.model_path, args.threshold,
                             norm_detrend=True,  label='B:JGR-detrend')

    # --- C: stead pretrained reference ---
    log.info('Loading C: stead pretrained ...')
    pred_c = _make_stead(args.threshold)

    # --- Run ---
    label_picks = []
    for label, pred in [
        ('A: JGR norm=peak  detrend=F', pred_a),
        ('B: JGR norm=peak  detrend=T', pred_b),
        ('C: stead (pretrained)',        pred_c),
    ]:
        t0 = time.time()
        picks = pred.predict_multi(keys_and_streams)
        elapsed = time.time() - t0
        log.info('%s → %d picks in %.1f s', label, len(picks), elapsed)
        label_picks.append((label, picks))

    # --- Print comparison ---
    print(f'\n=== Backend comparison: last {args.duration//60} min, '
          f'{len(keys_and_streams)} stations, threshold={args.threshold} ===')
    compare_all(label_picks)


if __name__ == '__main__':
    main()
