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
| `--earthscope-url` | `https://service.earthscope.org` | EarthScope FDSN URL or alias |
| `--no-earthscope` | — | Skip EarthScope FDSN fallback |
| `--dry-run` | — | Scan and report only; no fetch, no picks |
| `--scan-only` | — | Print gaps and exit |
| `--no-fetch` | — | Skip SeedLink fetch |
| `--no-pick` | — | Skip ML picking |
| `--replay-spool` | — | Publish any YAML pick files from `--pick-spool` and exit |
| `-v` / `--verbose` | — | DEBUG logging |

### Targeted event recovery

| Flag | Default | Description |
|------|---------|-------------|
| `--event-time` | — | ISO8601 origin time of a known event. When given, skips the archive-wide and origin-gap scans entirely and runs the targeted mode below instead. |
| `--event-lat` / `--event-lon` | — | Event location (degrees); required with `--event-time` |
| `--event-id` | — | Optional label used in logs and as pick provenance |
| `--event-radius-km` | `60` | Only consider stations within this distance of the event |
| `--event-window-before` | `20` | Seconds before `--event-time` to start the re-fetch/re-pick window |
| `--event-window-after` | `90` | Seconds after `--event-time` to end the window (covers P+S+coda out to `--event-radius-km`) |
| `--recovery-threshold` | `0.3` | Picker probability threshold for this pass only (live `sceasyquake.cfg` is untouched) |

Instead of scanning the whole archive, this mode finds the subscribed stations
within `--event-radius-km` of a known event (via the live SeisComP inventory,
not a static file), fetches/verifies just that tight time window, and re-picks
it with a lower, more sensitive threshold than the live picker uses. Picks are
published to the same PICK messaging group with `source=gap_recovery_targeted`
(distinct pick provenance -- `Pick/gap_recovery_targeted/...`), so they're
associated by the live scautoloc/scpyocto exactly like real-time picks.

```bash
# Try to recover a specific USGS-reported event our network didn't detect
python gap_recovery/gap_recovery.py \
    --event-time 2026-08-05T05:26:37.261 \
    --event-lat 32.101 --event-lon -101.943 \
    --event-id tx2026pgxqad \
    --device cpu
```

This is safe to run more aggressively than the live picker because the target
window is already known to contain a real, independently-confirmed event
(e.g. from USGS/ComCat) -- a lower threshold there doesn't raise the
network-wide false-pick rate the way permanently lowering `picker.threshold`
would. `quakeportal/sync/usgs_sync.py` drives this automatically; see its
docstring and the "USGS-only event recovery" section there.

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

## EarthScope authentication

EarthScope limits anonymous FDSN data downloads to 10 GiB/hour and 50 GiB/day
per IP, and **ends anonymous data access on 2026-12-15**
(<https://docs.earthscope.org/user-limits>).  With an account token the quota
is 200 GiB/hour and 1 TiB/day.

Log in once per host with the EarthScope CLI, installed outside the
SeisComP/anaconda environment (the SDK needs pydantic 2 and httpx):

```bash
pipx install earthscope-cli      # or: python3 -m venv ~/es-cli && ~/es-cli/bin/pip install earthscope-cli
es login                         # device flow; prints a URL to open in a browser
```

This writes `~/.earthscope/default/tokens.json`.  `gap_recovery.py` reads it,
refreshes the access token itself when it is near expiry, writes the refreshed
tokens back, and sends `Authorization: Bearer ...` on every dataselect request.
On a headless host, set `ES_OAUTH2__REFRESH_TOKEN` (from `es user get-refresh-token`)
instead.  The log states at the EarthScope step whether the run is authenticated.

HTTP 429 responses are waited out rather than dropped: 60 s, then until the
next hour boundary, then until 00:00 UTC (EarthScope's fixed quota windows).
