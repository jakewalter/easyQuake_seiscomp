# gap_recovery

Automated SeisComP archive gap detection, SeedLink back-fill, and ML-picker
playback for the easyQuake–SeisComP stack.

## Overview

```
SCAN  →  detect missing/incomplete waveform windows in ~/seiscomp/var/lib/archive
FETCH →  pull the missing data from the SeedLink ring-buffer (scart)
PICK  →  run a SeisBench ML model over every recovered window
UPLOAD→  push picks to SeisComP messaging → scautoloc/scevent → quakeportal
```

## Quick start

```bash
# From the workspace root — activate the conda env first
conda activate quakeportal

# Scan the last 24 hours and run the full pipeline
python gap_recovery/gap_recovery.py

# Scan only (no fetch, no picking)
python gap_recovery/gap_recovery.py --scan-only

# Dry-run: show what would happen without executing anything
python gap_recovery/gap_recovery.py --dry-run

# Longer lookback (e.g. recover 3 days while sceasyquake keeps running)
python gap_recovery/gap_recovery.py --hours 72 --device cpu

# Skip SeedLink fetch (use archive as-is)
python gap_recovery/gap_recovery.py --no-fetch
```

## Running alongside the live sceasyquake process

Gap recovery can run while `sceasyquake` is actively picking real-time data.
Both processes load a full PyTorch/PhaseNet model (~8 GB each), but with 62 GB
of RAM available this is not a problem.  The real risk is **messaging bus
saturation**: without throttling, gap_recovery publishes thousands of picks per
second, which can flood `scmaster` and destabilise `sceasyquake`'s connection.

The `--pick-rate` flag (default **20 picks/s**) prevents this.  At 20/s,
23,000 gap-recovery picks take ~20 minutes to publish — invisible to the live
pipeline.  The `sceasyquake` process always uses an unlimited rate (real-time
latency matters there), so the throttle only applies to gap_recovery.

```bash
# Default: 20 picks/s — safe to run while sceasyquake is live
python gap_recovery/gap_recovery.py --hours 72 --device cpu

# Faster if you want to finish sooner (test first that sceasyquake stays stable)
python gap_recovery/gap_recovery.py --hours 72 --device cpu --pick-rate 50

# Unlimited — only use if sceasyquake is stopped
python gap_recovery/gap_recovery.py --hours 72 --device cpu --pick-rate 0
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--archive` | `~/seiscomp/var/lib/archive` | SDS archive root |
| `--hours` | `24` | Lookback window in hours |
| `--min-gap` | `1.0` | Minimum gap to report (seconds) |
| `--seedlink-host` | `localhost` | SeedLink server |
| `--seedlink-port` | `18000` | SeedLink port |
| `--scart` | `~/seiscomp/bin/scart` | scart binary |
| `--device` | `cpu` | Torch device (`cpu` or `cuda`) |
| `--window` | `300` | Prediction chunk size in seconds |
| `--overlap` | `30` | Chunk overlap to avoid edge effects |
| `--sc-host` | `localhost` | SeisComP messaging host |
| `--pick-spool` | `~/sceasyquake/picks` | YAML fallback dir if SC messaging unreachable |
| `--pick-rate` | `20.0` | Max picks/s published to SeisComP messaging (0 = unlimited) |
| `--earthscope-url` | `IRIS` | EarthScope FDSN URL or alias |
| `--no-earthscope` | — | Skip EarthScope FDSN fallback |
| `--dry-run` | — | Scan and report only; no fetch, no picks |
| `--scan-only` | — | Print gaps and exit |
| `--no-fetch` | — | Skip SeedLink fetch |
| `--no-pick` | — | Skip ML picking |
| `--replay-spool` | — | Publish any YAML pick files from `--pick-spool` and exit |
| `-v` / `--verbose` | — | DEBUG logging |

## How it works

### SCAN

`obspy.clients.filesystem.sds.Client` reads each channel found in the archive
for the requested time window.  Gaps are detected in two ways:

1. **Missing file** — the returned Stream has no coverage at the window edges.
2. **Intra-file gap** — `Stream.get_gaps()` finds a data dropout inside an
   existing miniSEED day-file.

### FETCH

A scart stream-list file is written:

```
NET STA LOC CHA 2026-03-04T05:00:00 2026-03-04T06:00:00
```

then `scart -I slink://localhost:18000 --list streams.txt ARCHIVE` pulls from
the SeedLink ring-buffer.  How far back the ring-buffer reaches depends on the
`ringbuffer` size configured in `seedlink.cfg`.

### PICK

Recovered windows are read back from the SDS archive, split into
`--window`-second chunks with `--overlap`-second padding (to prevent picks
being cut off at chunk boundaries), and fed to `SeisBenchPredictor`.  Picks are
then sent via `PickUploader` to the live SeisComP messaging group where
`scautoloc` and `scevent` associate and locate them exactly as they would for
real-time picks.

Pick publishing is rate-limited to `--pick-rate` picks/second (default 20) to
avoid flooding `scmaster`.  Without throttling, rapid bulk publishing can
destabilise other connected clients such as the live `sceasyquake` process —
observed symptom is `sceasyquake` silently stopping 20–40 minutes after the
gap recovery pick phase completes.

If SeisComP messaging is unavailable the picks are spooled as YAML files to
`--pick-spool` for manual inspection or later import with `--replay-spool`.

### Cron integration

To run the recovery tool every hour, add to crontab:

```cron
15 * * * * /home/jwalter/anaconda3/envs/quakeportal/bin/python \
    /home/jwalter/easyQuake_seiscomp/gap_recovery/gap_recovery.py \
    --hours 2 --device cpu >> /var/log/gap_recovery.log 2>&1
```

`--hours 2` gives a 2-hour lookback so each hourly run overlaps with the
previous one, ensuring no window is ever missed.  The default `--pick-rate 20`
throttle keeps the messaging bus load well below the threshold that can
destabilise the live `sceasyquake` process.

## Dependencies

Install the extras if not present:

```bash
pip install obspy seisbench torch
```

The `sceasyquake` library is loaded automatically from `../sceasyquake/lib`.
