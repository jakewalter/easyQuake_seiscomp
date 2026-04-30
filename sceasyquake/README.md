# sceasyquake — SeisComP ML Phase Picker Module

A SeisComP 5 module that runs continuous ML-based phase picking (P and S)
using [easyQuake](https://github.com/jwalter/easyQuake) /
[SeisBench](https://github.com/seisbench/seisbench) models and publishes
`DataModel.Pick` objects to the SeisComP messaging bus — a drop-in companion
to `scautopick`.

---

## Supported Pickers

| `picker.backend` | Model | Default weights | Notes |
|---|---|---|---|
| `phasenet` | PhaseNet | `stead` | P + S picks |
| `gpd` | GPD | `original` | threshold default 0.994 (easyQuake) |
| `eqtransformer` | EQTransformer | `original` | P + S + detection |
| `seisbench` | any SeisBench model | configurable | use `picker.model` to choose |
| `auto` | PhaseNet (SeisBench) | `stead` | tries seisbench → phasenet pkg → stub |
| `stub` | built-in | — | no real picks; for testing |

---

## Installation

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| SeisComP ≥ 5.0 | with Python bindings (`seiscomp exec python3 -c "import seiscomp"`) |
| Anaconda / conda | recommended; the `base` env ships with Python 3.9 |
| NVIDIA driver ≥ 525 | optional, for GPU acceleration |
| git | for cloning the repositories |

### 2. Clone the repositories

```bash
# sceasyquake (this module)
git clone https://github.com/jwalter/easyQuake_seiscomp.git
cd easyQuake_seiscomp/sceasyquake

# easyQuake (optional — bundled weights, recommended)
git clone https://github.com/jwalter/easyQuake.git ~/easyQuake
```

### 3. Install Python dependencies

SeisComP's Python bindings run inside a specific interpreter — all ML deps
must be installed **into the same interpreter**.

```bash
# Export once; install.sh will use it automatically
export SC_PYTHON=$(seiscomp exec which python3)
echo "SeisComP python: $SC_PYTHON"

# Core deps (install.sh does NOT install these automatically)
$SC_PYTHON -m pip install obspy scipy numpy PyYAML watchdog psutil

# ML backend — choose one:
# Option A: SeisBench only (lighter)
$SC_PYTHON -m pip install seisbench

# Option B: Full easyQuake package (preferred — bundled weights + seisbench)
cd ~/easyQuake && $SC_PYTHON -m pip install -e . && cd -

# GPU support (skip if CPU-only)
$SC_PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu121
$SC_PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 4. Install sceasyquake

```bash
cd easyQuake_seiscomp/sceasyquake
bash install.sh          # auto-detects $SEISCOMP_ROOT or ~/seiscomp
```

`install.sh` (when `$SC_PYTHON` is exported) will:
- Install the Python package in **editable mode** via `$SC_PYTHON -m pip install -e .`
- Copy `bin/sceasyquake-stream.py` → `$SEISCOMP_ROOT/bin/sceasyquake`
- Copy `share/descriptions/sceasyquake.xml` → `$SEISCOMP_ROOT/etc/descriptions/`
- Copy `share/defaults/sceasyquake.cfg` → `$SEISCOMP_ROOT/etc/defaults/`
- Symlink `lib/sceasyquake` into `$SEISCOMP_ROOT/lib/python/` (fallback import path)
- Create `$SEISCOMP_ROOT/etc/sceasyquake.cfg` (user config) if absent

> **Note:** `install.sh` can also be run without setting `$SC_PYTHON` — it will
> auto-detect `seiscomp-python` inside `$SEISCOMP_ROOT/bin`. Set `SC_PYTHON`
> explicitly when SeisComP uses a non-standard Python (e.g. conda env).

### 5. Install the easyQuake TF environment (optional — for GPD and PhaseNet native backends)

> **Skip this step** if you are using `picker.backend = seisbench` — the SeisBench
> backend runs GPD, PhaseNet, and EQTransformer entirely within SeisComP's Python
> environment and does not need a separate conda env.

The `gpd` and `phasenet` (native easyQuake) backends invoke easyQuake's TF inference
scripts as a subprocess using a dedicated **Python 3.10+ / TensorFlow ≥ 2.12** conda
environment.  This environment does **not** need SeisComP bindings.

easyQuake 2.0 requires Python 3.10 or newer and TF 2.12+ (Keras 3 model format).

```bash
conda create -n easyquake python=3.10
conda activate easyquake
pip install git+https://github.com/jakewalter/easyQuake.git@development
conda deactivate
```

After creating the environment, set `picker.python_easyquake` in
`$SEISCOMP_ROOT/etc/sceasyquake.cfg` (or leave it at the default
`/home/$USER/anaconda3/envs/easyquake/bin/python`).

#### Repo-local script patches

`sceasyquake` ships patched versions of the two easyQuake TF inference scripts
under `share/scripts/`:

| File | Purpose |
|---|---|
| `share/scripts/gpd_predict.py` | GPD — removes 3–20 Hz bandpass filter so preprocessing matches SeisBench GPD (both apply max-normalisation inside the model); adds per-pick probability to output; uses Keras 3 model loading (easyQuake 2.0) |
| `share/scripts/phasenet_predict.py` | PhaseNet — updated for TF 2.12+ (no legacy shim injection; easyQuake 2.0 `postprocess.py` already includes `phase_prob`) |
| `share/scripts/postprocess.py` | PhaseNet postprocess — kept for compatibility with older easyQuake installs; superseded by easyQuake 2.0 `postprocess.py` which already includes `"phase_prob"` |

These files are **used automatically** — `_resolve_gpd_paths` and
`_resolve_phasenet_paths` in the predictor classes check for them at startup
and prefer them over the scripts installed in the conda env.  The conda env
originals are **never modified**.

On a fresh deployment no extra steps are needed: clone the repo and run
`install.sh`; the patched scripts are included in the checkout.

### 6. Enable and start

```bash
seiscomp enable sceasyquake
seiscomp update-config sceasyquake
seiscomp start sceasyquake
seiscomp status sceasyquake
```

---

## Configuration

Edit `$SEISCOMP_ROOT/etc/sceasyquake.cfg` directly **or** open `scconfig`
and navigate to *Processing → sceasyquake*.

### SeedLink source

```ini
seedlink.host = localhost
seedlink.port = 18000
```

### Stream selection

```ini
# Comma-separated NET.STA.LOC.CHA selectors (wildcards OK)
# Leave empty to subscribe to all vertical channels
streams.codes = TX.*.00.HHZ, TX.*.00.HHN, TX.*.00.HHE
```

### Picker settings

```ini
# Backend
picker.backend         = phasenet   # phasenet|gpd|eqtransformer|seisbench|auto

# Architecture — only used when backend = seisbench
picker.model           = PhaseNet   # PhaseNet | GPD | EQTransformer

# Pretrained weight set
# PhaseNet     : stead (default), diting, ethz, scedc, geofon, instance, neic
# GPD          : original (default), scedc, geofon
# EQTransformer: original (default), ethz, scedc, geofon, instance, stead
picker.pretrained      = stead

# Custom weights — overrides pretrained when set
# Accepts a raw *.pt state-dict OR a seisbench save() directory
# picker.model_path    = /path/to/weights.pt

# Thresholds
picker.threshold       = 0.5        # PhaseNet / EQTransformer / seisbench
picker.gpd_threshold   = 0.994      # GPD only (matches easyQuake default)

# Compute device
picker.device          = cuda       # cpu | cuda

# Sliding window
picker.buffer_seconds  = 60
picker.step_seconds    = 5
```

### Loading custom weights

**Raw PyTorch state-dict** (produced by `torch.save(model.state_dict(), path)`):
```ini
picker.model_path = /data/models/my_phasenet_finetuned.pt
```

**SeisBench save format** (produced by `model.save("/data/models/my_model")`):
```ini
picker.model_path = /data/models/my_model
# Expects: /data/models/my_model.pt + /data/models/my_model.json
```

When `picker.model_path` is set, `picker.pretrained` is ignored.

---

## Capacity Benchmark

Before deploying to production run the capacity benchmark to measure how many
streams this machine can pick within one step interval:

```bash
cd ~/easyQuake_seiscomp/sceasyquake

# Test 20 stations on CPU then GPU for 60 s each
python tests/bench_picker_capacity.py --channels 20 --step-seconds 5

# CPU-only machine
python tests/bench_picker_capacity.py --cpu-only --channels 50

# GPU only, larger batch
python tests/bench_picker_capacity.py --gpu-only --channels 100 --monitor-seconds 120
```

The script:
1. Checks CUDA / driver / PyTorch configuration and reports any issues
2. Loads each model (SeisBench) and runs `annotate()` repeatedly on synthetic data
3. Monitors CPU %, RAM, and GPU utilisation via `psutil` / `nvidia-smi`
4. Prints recommended `picker.step_seconds` and maximum safe stream count

Requires: `pip install psutil`

### Example results

Measured on:

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 3900X (12 physical / 24 logical cores) |
| GPU | NVIDIA GeForce RTX 2070 SUPER (8 GB VRAM) |
| Driver / CUDA | 525+ / 12.8 |
| PyTorch | 2.8.0+cu128 |
| seisbench | 0.10.2 |
| Stations in batch | 20 (3-component, 65 s buffer @ 100 Hz) |
| Step interval | 5 s |

**Throughput and capacity** (60 s run per backend/device):

| Backend | Device | Passes / 60 s | Mean annotate | Throughput | Max streams\* |
|---|---|---|---|---|---|
| PhaseNet | CPU | 516 | 108 ms | 185 streams/s | 739 |
| PhaseNet | **CUDA** | 964 | **57 ms** | **354 streams/s** | **1414** |
| GPD | CPU | 14 | 4,471 ms | 4.5 streams/s | 17 |
| GPD | **CUDA** | 97 | 614 ms | 33 streams/s | 130 |
| EQTransformer | CPU | 197 | 298 ms | 67 streams/s | 268 |
| EQTransformer | **CUDA** | 280 | 208 ms | 96 streams/s | 383 |

\* Max streams = streams that fit within one 5 s step with 20 % headroom.

**System resource usage:**

| Backend | Device | CPU % (mean/max) | RAM (mean) | GPU util (mean/max) | VRAM |
|---|---|---|---|---|---|
| PhaseNet | CPU | 14 / 19 % | 823 MiB | 1 / 7 % | 382 MiB |
| PhaseNet | CUDA | 9 / 13 % | 1,238 MiB | 7 / 13 % | 586 MiB |
| GPD | CPU | 20 / 25 % | 1,327 MiB | 1 / 7 % | 586 MiB |
| GPD | CUDA | 15 / 20 % | 1,335 MiB | 29 / 37 % | 687 MiB |
| EQTransformer | CPU | 16 / 19 % | 1,454 MiB | 1 / 7 % | 690 MiB |
| EQTransformer | CUDA | 8 / 13 % | 1,579 MiB | 13 / 20 % | 692 MiB |

**GPU speedup over CPU:** PhaseNet 1.9×, GPD 7.3×, EQTransformer 1.4×

> GPU utilisation is low at 20 stations — the bottleneck is data prep, not GPU
> compute. Run `--channels 100` to push utilisation higher and get tighter
> capacity estimates for large networks.

Benchmark-backed recommended config for this hardware:
```ini
picker.backend      = phasenet
picker.device       = cuda
picker.step_seconds = 5
# comfortably handles 1000+ simultaneous streams on RTX 2070 Super
```

---

## Unit Tests

```bash
cd ~/easyQuake_seiscomp/sceasyquake
pip install pytest

# All tests (no model weights or GPU required — uses monkeypatching)
pytest tests/ -v

# Just picker tests
pytest tests/test_all_pickers.py -v
```

Tests cover:
- `PhaseNetPredictor` — stub mode, seisbench path, `predict_multi` batch count
- `GPDPredictor` — default 0.994 threshold, custom state-dict loading
- `EQTransformerPredictor` — P + S picks, single annotate call
- `SeisBenchPredictor` — all three architectures, custom weights (`.pt` + dir)
- `_make_predictor` factory — all backends route to correct class
- Pick schema validation — all backends return required fields

---

## Architecture

```
SeedLink ──► SeisCompStream ──► per-channel obspy.Stream buffers
                                         │
                                (every step_seconds)
                                         │
                                  predict_multi()   ◄── single GPU pass
                                         │
                                   PickUploader ──► SC messaging PICK group
                                                           │
                                                      scautoloc ──► scevent
```

`predict_multi()` merges all ready-channel streams into one combined
`ObsPy.Stream` and calls `model.annotate()` once per step — a single GPU
forward pass covers every station.

---

## Database Diagnostics

All queries run against the SeisComP MySQL database.  Replace credentials if
yours differ from the defaults (`sysop:sysop`, database `seiscomp`).

```bash
MYSQL="mysql -usysop -psysop seiscomp 2>/dev/null"
```

### Recent picks from sceasyquake

```bash
# Count and latest pick time from sceasyquake in the last 10 minutes
$MYSQL -BNe "
SELECT COUNT(*) AS picks_10min, MAX(time_value) AS latest
FROM Pick
WHERE creationInfo_agencyID = 'sceasyquake'
  AND time_value > DATE_SUB(UTC_TIMESTAMP(), INTERVAL 10 MINUTE);"

# Most recent 20 picks — station, phase, time, probability
$MYSQL -BNe "
SELECT p.time_value, p.waveformID_networkCode, p.waveformID_stationCode,
       p.phaseHint_code, p.creationInfo_agencyID
FROM Pick p
WHERE p.creationInfo_agencyID = 'sceasyquake'
ORDER BY p.time_value DESC LIMIT 20;"

# Pick rate per minute over the last hour (useful to spot gaps)
$MYSQL -BNe "
SELECT DATE_FORMAT(time_value, '%Y-%m-%d %H:%i') AS minute, COUNT(*) AS picks
FROM Pick
WHERE creationInfo_agencyID = 'sceasyquake'
  AND time_value > DATE_SUB(UTC_TIMESTAMP(), INTERVAL 60 MINUTE)
GROUP BY minute ORDER BY minute DESC LIMIT 60;"
```

### Pick–amplitude pairing (confirms sceasyquake SNR amplitudes reach scautoloc)

scautoloc requires an `Amplitude(type='snr')` per pick before it will process
it.  If the ratio drops below ~90 % picks will stall "waiting for amplitude".

```bash
$MYSQL -BNe "
SELECT
  total_picks,
  matched_amplitudes,
  ROUND(100.0 * matched_amplitudes / total_picks, 1) AS match_pct
FROM (
  SELECT COUNT(*) AS total_picks FROM Pick
  WHERE creationInfo_agencyID = 'sceasyquake'
    AND time_value > DATE_SUB(UTC_TIMESTAMP(), INTERVAL 10 MINUTE)
) p,
(
  SELECT COUNT(*) AS matched_amplitudes FROM Amplitude a
  JOIN Pick pk ON pk._oid = (SELECT _oid FROM PublicObject WHERE publicID = a.pickID LIMIT 1)
  WHERE pk.creationInfo_agencyID = 'sceasyquake'
    AND pk.time_value > DATE_SUB(UTC_TIMESTAMP(), INTERVAL 10 MINUTE)
) a;"
```

### Recent events and origins

```bash
# All origins with lat/lon, phase count, and the event they belong to
# (origins without an event_id are non-preferred solutions inside an event)
$MYSQL -BNe "
SELECT po.publicID AS origin_id,
       o.time_value,
       o.latitude_value,
       o.longitude_value,
       o.quality_usedPhaseCount   AS phases,
       o.quality_azimuthalGap     AS gap_deg,
       (SELECT ep.publicID
        FROM Event e
        JOIN PublicObject ep ON ep._oid = e._oid
        WHERE e.preferredOriginID = po.publicID LIMIT 1) AS preferred_in_event,
       (SELECT ep2.publicID
        FROM OriginReference orr
        JOIN PublicObject ep2 ON ep2._oid = orr._parent_oid
        WHERE orr.originID = po.publicID LIMIT 1) AS member_of_event
FROM Origin o
JOIN PublicObject po ON po._oid = o._oid
ORDER BY o.time_value DESC LIMIT 20;" | column -t

# Event summary: event ID, preferred origin time/location, magnitude, phase count
$MYSQL -BNe "
SELECT ep.publicID AS event_id,
       o.time_value,
       o.latitude_value  AS lat,
       o.longitude_value AS lon,
       o.quality_usedPhaseCount AS phases,
       m.magnitude_value AS mag,
       m.type AS mag_type
FROM Event e
JOIN PublicObject ep ON ep._oid = e._oid
LEFT JOIN PublicObject op ON op.publicID = e.preferredOriginID
LEFT JOIN Origin o ON o._oid = op._oid
LEFT JOIN PublicObject mp ON mp.publicID = e.preferredMagnitudeID
LEFT JOIN Magnitude m ON m._oid = mp._oid
ORDER BY o.time_value DESC LIMIT 20;" | column -t

# Total counts for a quick sanity check
$MYSQL -BNe "SELECT
  (SELECT COUNT(*) FROM Pick)        AS total_picks,
  (SELECT COUNT(*) FROM Amplitude)   AS total_amplitudes,
  (SELECT COUNT(*) FROM Origin)      AS total_origins,
  (SELECT COUNT(*) FROM Event)       AS total_events,
  (SELECT COUNT(*) FROM Magnitude)   AS total_magnitudes;"
```

### Cluster picks by event time (spot groups of simultaneous picks → real events)

```bash
# Stations with ≥4 P-picks in the same 60-second window — each row is a likely event
$MYSQL -BNe "
SELECT DATE_FORMAT(time_value, '%Y-%m-%d %H:%i') AS minute,
       COUNT(DISTINCT waveformID_stationCode) AS stations,
       COUNT(*) AS total_picks
FROM Pick
WHERE phaseHint_code = 'P'
  AND creationInfo_agencyID = 'sceasyquake'
  AND time_value > DATE_SUB(UTC_TIMESTAMP(), INTERVAL 2 HOUR)
GROUP BY minute
HAVING stations >= 4
ORDER BY stations DESC LIMIT 20;"
```

### scautoloc config quick-check

The single most common cause of zero origins is `autoloc.amplTypeAbs` not
matching the amplitude type sceasyquake publishes (`snr`):

```bash
grep -E "amplTypeAbs|amplTypeSNR|minPickSNR|locator.profile|networkType|minPhaseCount" \
  "$SEISCOMP_ROOT/etc/scautoloc.cfg"

# Expected output — all five lines must be present:
# autoloc.amplTypeAbs   = snr   ← CRITICAL: hasAmplitude() checks pick->amp
# autoloc.amplTypeSNR   = snr
# autoloc.minPickSNR    = 0
# locator.profile       = iasp91
# autoloc.networkType   = local
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Client name not unique` | Old process still running | `pkill -f sceasyquake-stream` then `seiscomp start sceasyquake` |
| GPU at 0 % utilisation | `picker.device = cpu` | Set `picker.device = cuda` and restart |
| No picks after 2 min | Buffer not full yet | Wait `buffer_seconds`; check `picker.threshold` |
| `seisbench not available` | Wrong Python env | `seiscomp exec python3 -c "import seisbench"` |
| `stationLocations` errors | scautoloc missing stations | Regenerate `station-locations.conf` from DB |
| Very high CPU with GPU | PyTorch thread count | Already limited to 4 threads in `load_model()` |
