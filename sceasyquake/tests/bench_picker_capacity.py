#!/usr/bin/env python3
"""sceasyquake picker capacity benchmark.

Measures how many waveform streams PhaseNet (SeisBench) can process per
step on this machine, first on CPU then on GPU (if CUDA is available).
Monitors CPU, RAM, and GPU resource usage over 60 seconds of live inference
and prints a capacity recommendation.

Usage::

    cd sceasyquake
    python tests/bench_picker_capacity.py
    # or with more channels:
    python tests/bench_picker_capacity.py --channels 50 --step-seconds 5

Requirements:
    pip install psutil
    seisbench (or easyQuake) installed
"""

import argparse
import statistics
import sys
import time
import logging
import threading

import numpy as np

log = logging.getLogger('bench')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s  %(message)s')

BANNER = '=' * 70


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_stream(n_stations: int, sr: float = 100.0, duration: float = 65.0):
    """Create a synthetic 3-component stream for *n_stations* stations."""
    try:
        from obspy import Stream, Trace, UTCDateTime
    except ImportError:
        sys.exit('ERROR: obspy is not installed.  pip install obspy')

    npts = int(sr * duration)
    st = Stream()
    for i in range(n_stations):
        for cha in ('HHZ', 'HHN', 'HHE'):
            tr = Trace(data=np.random.randn(npts).astype(np.float32))
            tr.stats.network = 'TX'
            tr.stats.station = f'STA{i:03d}'
            tr.stats.location = '00'
            tr.stats.channel = cha
            tr.stats.sampling_rate = sr
            tr.stats.starttime = UTCDateTime(2026, 1, 1)
            st += tr
    return st


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Polls CPU / RAM / GPU every *interval* seconds in a background thread."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._stop = threading.Event()
        self.cpu_samples: list = []
        self.ram_mb_samples: list = []
        self.gpu_util_samples: list = []
        self.gpu_mem_mb_samples: list = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        try:
            import psutil
        except ImportError:
            log.warning('psutil not installed; CPU/RAM monitoring disabled.  pip install psutil')
            return

        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.ram_mb_samples.append(psutil.Process().memory_info().rss / 1024 / 1024)
            gpu_util, gpu_mem = _gpu_stats()
            if gpu_util is not None:
                self.gpu_util_samples.append(gpu_util)
                self.gpu_mem_mb_samples.append(gpu_mem)
            time.sleep(self.interval)

    def summary(self) -> dict:
        def _s(lst):
            return {'mean': statistics.mean(lst), 'max': max(lst)} if lst else {}
        return {
            'cpu': _s(self.cpu_samples),
            'ram_mb': _s(self.ram_mb_samples),
            'gpu_util': _s(self.gpu_util_samples),
            'gpu_mem_mb': _s(self.gpu_mem_mb_samples),
        }


def _gpu_stats():
    """Return (gpu_util_pct, used_mem_mb) or (None, None)."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            used = torch.cuda.memory_allocated(0) / 1024 / 1024
            return None, used
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

# Map backend name → SeisBench class name
_BACKEND_CLASS = {
    'phasenet': 'PhaseNet',
    'gpd': 'GPD',
    'eqtransformer': 'EQTransformer',
}

# Default pretrained weights per backend
_BACKEND_DEFAULT_PRETRAINED = {
    'phasenet': 'stead',
    'gpd': 'original',
    'eqtransformer': 'original',
}


def _load_model(backend: str, device: str, pretrained: str = ''):
    """Load a SeisBench model for *backend* onto *device*."""
    cls_name = _BACKEND_CLASS.get(backend)
    if cls_name is None:
        raise RuntimeError(f'Unknown backend: {backend!r}.  Choose: {list(_BACKEND_CLASS)}')

    pretrained = pretrained or _BACKEND_DEFAULT_PRETRAINED[backend]

    # Try easyQuake's bundled seisbench first, then system seisbench
    sbm = None
    for mp in ('easyQuake.seisbench', 'seisbench.models'):
        try:
            if mp == 'easyQuake.seisbench':
                import easyQuake.seisbench as _sbm  # type: ignore
            else:
                import seisbench.models as _sbm  # type: ignore
            if not hasattr(_sbm, cls_name):
                continue
            sbm = _sbm
            log.info('Using %s.%s', mp, cls_name)
            break
        except Exception as exc:
            log.debug('Skipping %s: %s', mp, exc)

    if sbm is None:
        raise RuntimeError(
            f'{cls_name} not available in easyQuake.seisbench or seisbench.models.\n'
            'Install one of:  pip install easyQuake   or   pip install seisbench'
        )

    import torch
    log.info('Loading %s (pretrained=%s) …', cls_name, pretrained)
    model_cls = getattr(sbm, cls_name)
    model = model_cls.from_pretrained(pretrained)
    model.eval()
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
        model = model.cuda()
        log.info('Model on CUDA device: %s', torch.cuda.get_device_name(0))
    else:
        log.info('Model on CPU')
    return model


def _run_single_annotate(model, stream) -> tuple:
    """Run annotate() and return (wall-clock seconds, pick_count)."""
    from scipy.signal import find_peaks
    import torch
    with torch.no_grad():
        t0 = time.perf_counter()
        annotations = model.annotate(stream)
        elapsed = time.perf_counter() - t0
    # Count peaks above 0.3 in P/S annotation channels
    n_picks = 0
    for ann_tr in annotations:
        ch = ann_tr.stats.channel
        if ch.endswith('_P') or ch.endswith('_S') or ch.endswith('_N'):
            if ch.endswith('_P') or ch.endswith('_S'):
                peaks, _ = find_peaks(ann_tr.data.astype(float), height=0.3, distance=10)
                n_picks += len(peaks)
    return elapsed, n_picks


def _benchmark_device(
    device: str,
    n_stations: int,
    sr: float,
    buffer_sec: float,
    step_sec: float,
    monitor_sec: float,
    pretrained: str,
    backend: str = 'phasenet',
) -> dict:
    """
    Benchmark one device.  Returns a result dict with timing and resource stats.
    """
    pretrained_label = pretrained or _BACKEND_DEFAULT_PRETRAINED.get(backend, '?')
    print(f'\n{BANNER}')
    print(f'  Backend: {backend}   Device: {device.upper()}   stations: {n_stations}   '
          f'buffer: {buffer_sec:.0f}s   pretrained: {pretrained_label}')
    print(BANNER)

    try:
        model = _load_model(backend, device, pretrained)
    except RuntimeError as exc:
        print(f'  SKIP: {exc}')
        return {'device': device, 'backend': backend, 'skipped': True, 'reason': str(exc)}

    # Check CUDA is actually in use
    if device == 'cuda':
        try:
            import torch
            p = next(model.parameters())
            if not p.is_cuda:
                print('  WARNING: Parameters are not on CUDA despite device=cuda')
        except Exception:
            pass

    stream = _make_stream(n_stations, sr=sr, duration=buffer_sec + 5)

    # Warm-up pass (not counted)
    log.info('Warm-up pass …')
    try:
        _run_single_annotate(model, stream.copy())
    except Exception as exc:
        print(f'  SKIP: annotate() warm-up failed: {exc}')
        return {'device': device, 'backend': backend, 'skipped': True, 'reason': str(exc)}

    # Live monitoring loop
    monitor = ResourceMonitor(interval=1.0)
    times: list = []
    total_picks: int = 0
    deadline = time.time() + monitor_sec

    log.info('Running for %.0f seconds …', monitor_sec)
    monitor.start()
    while time.time() < deadline:
        t, n_picks = _run_single_annotate(model, stream.copy())
        times.append(t)
        total_picks += n_picks
        log.info('  step %.3f s  picks: %d  (%.1f streams/s)',
                 t, n_picks, n_stations / t if t > 0 else 0)
    monitor.stop()

    res = monitor.summary()
    mean_t = statistics.mean(times)
    max_t = max(times)
    throughput = n_stations / mean_t  # streams per second

    print(f'\n  Annotate timing ({len(times)} passes):')
    print(f'    mean  : {mean_t*1000:.1f} ms')
    print(f'    max   : {max_t*1000:.1f} ms')
    print(f'    throughput (mean): {throughput:.1f} streams/s')
    picks_per_pass = total_picks / len(times) if times else 0
    print(f'    picks detected : {total_picks} total  ({picks_per_pass:.1f}/pass  on synthetic noise)')

    if res['cpu']:
        print(f'\n  CPU usage : mean {res["cpu"]["mean"]:.1f}%  max {res["cpu"]["max"]:.1f}%')
    if res['ram_mb']:
        print(f'  RAM usage : mean {res["ram_mb"]["mean"]:.0f} MiB  max {res["ram_mb"]["max"]:.0f} MiB')
    if res['gpu_util']:
        print(f'  GPU util  : mean {res["gpu_util"]["mean"]:.1f}%  max {res["gpu_util"]["max"]:.1f}%')
    if res['gpu_mem_mb']:
        print(f'  GPU memory: mean {res["gpu_mem_mb"]["mean"]:.0f} MiB  max {res["gpu_mem_mb"]["max"]:.0f} MiB')

    # Capacity estimate: how many streams can fit within one step_seconds window?
    # A new inference batch must finish before the next step fires.
    # streams_per_batch * (mean_t / n_stations) <= step_sec
    # → max_streams = step_sec / (mean_t / n_stations)
    max_streams_mean = int(step_sec / (mean_t / n_stations))
    max_streams_safe = int(step_sec * 0.8 / (mean_t / n_stations))  # 20% headroom

    return {
        'device': device,
        'backend': backend,
        'skipped': False,
        'n_stations': n_stations,
        'mean_t': mean_t,
        'max_t': max_t,
        'throughput': throughput,
        'total_picks': total_picks,
        'picks_per_pass': picks_per_pass,
        'max_streams_mean': max_streams_mean,
        'max_streams_safe': max_streams_safe,
        'resources': res,
        'passes': len(times),
    }


# ---------------------------------------------------------------------------
# Recommendation printer
# ---------------------------------------------------------------------------

def _print_recommendation(results: list, step_sec: float):
    print(f'\n{BANNER}')
    print('  CAPACITY RECOMMENDATION')
    print(BANNER)

    has_cuda = any(r['device'] == 'cuda' and not r.get('skipped') for r in results)
    has_cpu  = any(r['device'] == 'cpu'  and not r.get('skipped') for r in results)

    for r in results:
        backend_label = r.get('backend', '?').upper()
        if r.get('skipped'):
            print(f'\n  [{backend_label}/{r["device"].upper()}]  SKIPPED — {r.get("reason","")}')
            continue

        print(f'\n  [{backend_label}/{r["device"].upper()}]')
        print(f'    Benchmark used {r["n_stations"]} stations over {r["passes"]} passes.')
        print(f'    Picks on synthetic noise : {r["total_picks"]} total  ({r["picks_per_pass"]:.1f}/pass)')
        print(f'    Mean annotate time : {r["mean_t"]*1000:.1f} ms')
        print(f'    Time per station   : {r["mean_t"]/r["n_stations"]*1000:.2f} ms/station')
        print(f'    With step_seconds={step_sec:.0f}s and 20 %% headroom:')
        print(f'    → Recommended max streams : {r["max_streams_safe"]}')
        print(f'    → Theoretical max streams : {r["max_streams_mean"]}')

    # Compare CPU vs GPU within same backend (first backend found)
    backends_seen = list(dict.fromkeys(r.get('backend','?') for r in results if not r.get('skipped')))
    for bk in backends_seen:
        cpu_r  = next((r for r in results if r.get('backend') == bk and r['device'] == 'cpu'  and not r.get('skipped')), None)
        cuda_r = next((r for r in results if r.get('backend') == bk and r['device'] == 'cuda' and not r.get('skipped')), None)
        if cpu_r and cuda_r:
            speedup = cpu_r['mean_t'] / cuda_r['mean_t']
            print(f'\n  [{bk.upper()}] GPU is {speedup:.1f}\u00d7 faster than CPU for this batch size.')
            if speedup < 1.5:
                print('  NOTE: Small speedup — consider a larger batch (--channels) '
                      'to better saturate the GPU.')

    # Actionable config advice (use best CUDA result, or best CPU)
    cuda_results = [r for r in results if r['device'] == 'cuda' and not r.get('skipped')]
    cpu_results  = [r for r in results if r['device'] == 'cpu'  and not r.get('skipped')]
    best = min(cuda_results, key=lambda r: r['mean_t']) if cuda_results else \
           min(cpu_results,  key=lambda r: r['mean_t']) if cpu_results else None
    cuda_r = best if best and best['device'] == 'cuda' else None
    print(f'\n  Suggested sceasyquake.cfg settings (fastest backend: {best["backend"] if best else "?"}):') if best else None
    if best:
        suggested_step = max(5, int(best['mean_t'] * 1.3))
        print(f'    picker.backend        = {best["backend"]}')
        print(f'    picker.device         = {"cuda" if cuda_r else "cpu"}')
        print(f'    picker.step_seconds   = {suggested_step}')
        print(f'    # supports up to ~{best["max_streams_safe"]} streams at this step interval')

    if not has_cuda:
        print('\n  CUDA not available.  To enable GPU acceleration:')
        print('    1. Verify NVIDIA driver:  nvidia-smi')
        print('    2. Install CUDA-enabled PyTorch:')
        print('       pip install torch --index-url https://download.pytorch.org/whl/cu121')
        print('    3. Set picker.device = cuda in sceasyquake.cfg')

    print()


# ---------------------------------------------------------------------------
# CUDA system check
# ---------------------------------------------------------------------------

def _cuda_system_check():
    print(f'\n{BANNER}')
    print('  CUDA / SYSTEM CHECK')
    print(BANNER)
    try:
        import torch
        print(f'  PyTorch version : {torch.__version__}')
        print(f'  CUDA available  : {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'  CUDA version    : {torch.version.cuda}')
            print(f'  Device 0        : {torch.cuda.get_device_name(0)}')
            props = torch.cuda.get_device_properties(0)
            print(f'  Total VRAM      : {props.total_memory / 1024**2:.0f} MiB')
            try:
                import subprocess
                r = subprocess.run(['nvidia-smi', '--query-gpu=driver_version',
                                    '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=3)
                print(f'  Driver version  : {r.stdout.strip()}')
            except Exception:
                pass
        else:
            print('  NOTE: No CUDA GPU detected.  Benchmark will run CPU-only.')
    except ImportError:
        print('  PyTorch not installed!  pip install torch')

    try:
        import psutil
        print(f'\n  CPU cores  : {psutil.cpu_count(logical=False)} physical '
              f'/ {psutil.cpu_count(logical=True)} logical')
        ram_gb = psutil.virtual_memory().total / 1024**3
        print(f'  Total RAM  : {ram_gb:.1f} GiB')
    except ImportError:
        print('\n  psutil not installed; skipping CPU/RAM info.  pip install psutil')

    try:
        import seisbench
        print(f'\n  seisbench version: {seisbench.__version__}')
    except Exception:
        pass
    try:
        import easyQuake
        print(f'  easyQuake version: {easyQuake.__version__}')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='sceasyquake picker capacity benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--channels', type=int, default=20,
                        help='Number of stations to include in each benchmark batch')
    parser.add_argument('--step-seconds', type=float, default=5.0,
                        help='Target step interval (s) used to calculate capacity')
    parser.add_argument('--buffer-seconds', type=float, default=65.0,
                        help='Waveform buffer length fed to annotate()')
    parser.add_argument('--monitor-seconds', type=float, default=60.0,
                        help='How long to monitor resource usage per device test')
    parser.add_argument('--pretrained', default='',
                        help='SeisBench pretrained weight set (default varies by backend)')
    parser.add_argument('--backend', default='all',
                        help='Which picker backend(s) to benchmark: '
                             'phasenet, gpd, eqtransformer, or "all" (default: all)')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Skip GPU benchmark even if CUDA is available')
    parser.add_argument('--gpu-only', action='store_true',
                        help='Skip CPU benchmark')
    args = parser.parse_args()

    print(f'\n{BANNER}')
    print('  sceasyquake — Picker Capacity Benchmark')
    print(BANNER)

    _cuda_system_check()

    import torch
    # Set thread limits once here so repeated model loads don't trigger errors
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # already set from a previous run in this process

    cuda_ok = torch.cuda.is_available() and not args.cpu_only

    devices = []
    if not args.gpu_only:
        devices.append('cpu')
    if cuda_ok:
        devices.append('cuda')
    elif args.gpu_only and not cuda_ok:
        print('\nERROR: --gpu-only requested but CUDA is not available.')
        sys.exit(1)

    if args.backend == 'all':
        backends = list(_BACKEND_CLASS.keys())
    else:
        backends = [b.strip() for b in args.backend.split(',')]

    total = len(backends) * len(devices)
    print(f'\n  Running {total} benchmark(s): {backends} × {devices}')
    print(f'  Each run: {args.monitor_seconds:.0f} s of live inference\n')

    results = []
    for backend in backends:
        for device in devices:
            r = _benchmark_device(
                device=device,
                n_stations=args.channels,
                sr=100.0,
                buffer_sec=args.buffer_seconds,
                step_sec=args.step_seconds,
                monitor_sec=args.monitor_seconds,
                pretrained=args.pretrained,
                backend=backend,
            )
            results.append(r)

    _print_recommendation(results, step_sec=args.step_seconds)


if __name__ == '__main__':
    main()
