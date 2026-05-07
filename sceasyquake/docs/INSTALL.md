# Installation

## Prerequisites

- SeisComP ≥ 5 with Python bindings
- git
- A GPU with CUDA ≥ 11.8 (optional — CPU mode works fine)

## 1. Clone the repositories

```bash
# This module
git clone https://github.com/jwalter/easyQuake_seiscomp.git
cd easyQuake_seiscomp/sceasyquake

# easyQuake (optional but recommended — bundled seisbench weights + GPD)
# Clone wherever you like; adjust the path in the pip install step below.
git clone https://github.com/jwalter/easyQuake.git /path/to/easyQuake
```

## 2. Identify the SeisComP Python interpreter

```bash
export SC_PYTHON=$(seiscomp exec which python3)
echo "SeisComP python: $SC_PYTHON"
```

All `pip install` commands below must use **this interpreter**, not the
system `/usr/bin/python3`. Export `SC_PYTHON` in your shell before running
`install.sh` — the script will use it automatically.

## 3. Install Python dependencies

> `install.sh` handles the sceasyquake package itself but does **not**
> install the ML backend or core deps — do that here first.

```bash
# Core runtime deps
$SC_PYTHON -m pip install obspy scipy numpy PyYAML watchdog psutil

# ML backend — choose one:
# Option A: SeisBench only (lighter, ships PhaseNet, GPD, EQTransformer)
$SC_PYTHON -m pip install seisbench

# Option B: Full easyQuake package (preferred — bundled weights)
cd /path/to/easyQuake && $SC_PYTHON -m pip install -e . && cd -

# GPU acceleration (skip for CPU-only deployments)
# Replace cu121 with the wheel tag matching your CUDA version:
#   CUDA 11.8 → cu118   CUDA 12.1 → cu121   CUDA 12.4 → cu124
# See https://pytorch.org/get-started/locally/ for the current selector.
$SC_PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu121
$SC_PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 4. Install sceasyquake

```bash
cd /path/to/easyQuake_seiscomp/sceasyquake
bash install.sh          # auto-detects $SEISCOMP_ROOT or ~/seiscomp
```

The script (with `$SC_PYTHON` exported) will:
1. Install the sceasyquake package via `$SC_PYTHON -m pip install -e .` (editable — source is live).
2. Copy `bin/sceasyquake-stream.py` → `$SEISCOMP_ROOT/bin/sceasyquake` with the correct shebang.
3. Copy `share/descriptions/sceasyquake.xml` → `$SEISCOMP_ROOT/etc/descriptions/` (for `scconfig` UI).
4. Copy `share/init/sceasyquake.py` → `$SEISCOMP_ROOT/etc/init/` (**required** for `seiscomp enable/start`).
5. Copy `share/defaults/sceasyquake.cfg` → `$SEISCOMP_ROOT/etc/defaults/` (factory defaults).
6. Symlink `lib/sceasyquake` into `$SEISCOMP_ROOT/lib/python/` (fallback import path for `seiscomp-python`).
7. Create `$SEISCOMP_ROOT/etc/sceasyquake.cfg` (user overrides) if absent.

To pass a non-standard SeisComP root:
```bash
bash install.sh /opt/seiscomp
```

## 5. Enable and start

```bash
seiscomp enable sceasyquake
seiscomp update-config sceasyquake
seiscomp start sceasyquake
seiscomp status sceasyquake
```

## Configuration reference

Edit `$SEISCOMP_ROOT/etc/sceasyquake.cfg` directly **or** open `scconfig`,
navigate to *Processing → sceasyquake*.

| Parameter | Description | Default |
|---|---|---|
| `seedlink.host` | SeedLink server host | `localhost` |
| `seedlink.port` | SeedLink server port | `18000` |
| `streams.codes` | `NET.STA.LOC.CHA` selectors (comma-separated) | *(all `??Z`)* |
| `picker.backend` | `auto` | `phasenet` | `gpd` | `eqtransformer` | `seisbench` | `stub` | `auto` |
| `picker.model` | Architecture for `seisbench` backend: `PhaseNet` | `GPD` | `EQTransformer` | `PhaseNet` |
| `picker.pretrained` | Pretrained weight set name (see table below) | `stead` |
| `picker.model_path` | Path to custom weights — raw `.pt` or seisbench dir | *(none)* |
| `picker.threshold` | Probability threshold 0–1 (PhaseNet / EQTransformer) | `0.5` |
| `picker.gpd_threshold` | Probability threshold for GPD picks | `0.994` |
| `picker.device` | `cpu` or `cuda` | `cpu` |
| `picker.buffer_seconds` | Sliding window length (s) | `60` |
| `picker.step_seconds` | Slide advance (s) | `5` |

### Pretrained weight sets by architecture

| Backend / model | Available `picker.pretrained` values |
|---|---|
| PhaseNet | `stead` *(default)*, `diting`, `ethz`, `scedc`, `geofon`, `instance`, `neic` |
| GPD | `original` *(default)*, `scedc`, `geofon` |
| EQTransformer | `original` *(default)*, `ethz`, `scedc`, `geofon`, `instance`, `stead` |

### Custom weights

```ini
# Raw PyTorch state-dict saved with torch.save(model.state_dict(), path)
picker.model_path = /data/models/finetuned.pt

# SeisBench save() directory (expects finetuned.pt + finetuned.json)
picker.model_path = /data/models/finetuned
```

When `picker.model_path` points to a raw `.pt`/`.pth` state-dict, the SeisBench
hub is never contacted — the model is built directly from the bare constructor
using `picker.norm`, and the state-dict fully replaces the weights. Set
`picker.norm` to match your model's training normalization (`peak` or `std`).
`picker.pretrained` is not used.

When `model_path` is a SeisBench save directory (`.pt` + `.json`), all
metadata comes from the `.json` and `picker.pretrained` is also not used.

## Capacity Benchmark

Run before deploying to production:

```bash
cd /path/to/easyQuake_seiscomp/sceasyquake
python tests/bench_picker_capacity.py --channels 20 --step-seconds 5
# CPU-only
python tests/bench_picker_capacity.py --cpu-only --channels 50
```

Reports recommended `picker.step_seconds` and maximum safe stream count.

## Test suite

```bash
cd /path/to/easyQuake_seiscomp/sceasyquake
$SC_PYTHON -m pip install pytest
pytest tests/ -v
```

## Standalone mode (no SeisComP bindings)

```bash
python3 bin/sceasyquake-stream.py \
    --seedlink localhost:18000 \
    --streams "CI.*.*.HHZ" \
    --backend phasenet \
    --device cuda \
    --pick-out ~/sceasyquake/picks
```

Picks are written as YAML files under `--pick-out`.

Run `seiscomp restart sceasyquake` after each config change.
