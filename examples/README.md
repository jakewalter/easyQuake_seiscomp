# examples/

Operational tooling for a running SeisComP + sceasyquake installation:

| Script | Purpose |
|---|---|
| [`status_check.py`](status_check.py) | Live health check, config snapshot/compare, and template-apply for standing up a new system |
| [`bbox_station_service.py`](bbox_station_service.py) | Web UI to select a geographic bounding box and auto-provision SeisComP station bindings/inventory for that region |

Both scripts read `SEISCOMP_ROOT` from the environment (default `/home/jwalter/seiscomp`) and operate on that installation's `etc/` tree, database, and running modules.

---

## status_check.py

```
python examples/status_check.py [check]
python examples/status_check.py snapshot [-o OUTPUT_DIR]
python examples/status_check.py compare SNAPSHOT_A SNAPSHOT_B [-o OUTPUT_FILE]
python examples/status_check.py apply SNAPSHOT_DIR [-y]
```

### `check` (default)

Runs a live audit of the pick → origin → event → magnitude pipeline: which modules are running, SeedLink/slarchive activity, station key-file validity, `global.cfg`/`scautoloc.cfg` pipeline requirements, recent pick/amplitude flow and pairing ratio, recent events, and pick latency. Plain-text output with inline fix suggestions. Safe to run anytime; read-only.

### `snapshot`

Archives a point-in-time copy of a system's configuration to a directory (default `seiscomp_snapshots/<hostname>_<timestamp>/`):

```
seiscomp_snapshots/home3_20260708_165424/
├── manifest.json     # hostname, timestamp, seiscomp version, DB station count,
│                     #   running/stopped modules, scfakequake + sceasyquake git/model info
├── modules.txt       # raw `seiscomp status` output
├── etc/              # full copy of $SEISCOMP_ROOT/etc (global.cfg, all module .cfg files,
│                     #   etc/key/* station bindings, scmaster.cfg with the DB password redacted)
└── share/
    └── scautoloc/    # station-locations.conf, grid.conf, station.conf
```

`scfakequake` is captured specifically because it lives outside the sceasyquake/SeisComP core: the snapshot records its config plus the git commit/dirty state of the external `FakeQuake` repo and the sha256/size/mtime of its RF model file, so you can tell whether a production box is running different code or a different model than another checkout. The git commit of this `easyQuake_seiscomp` checkout is recorded too (covers `sceasyquake`/`worker.py`).

The DB password in the copied `scmaster.cfg` is masked (`sysop:***REDACTED***@...`) — snapshots are meant to be shared/compared without leaking credentials.

### `compare`

Diffs two snapshot directories: manifest summary (hostname, SeisComP version, scfakequake/sceasyquake git commits, running-module set differences) followed by a unified diff for every config file that differs, plus files present in only one snapshot. Use `-o FILE` to write the report instead of printing it.

```
python examples/status_check.py compare seiscomp_snapshots/prod_20260701_090000 seiscomp_snapshots/dev_20260708_120000
```

### `apply`

Lays a snapshot's config down as a **template** on another (typically new) system. It classifies every file in the snapshot into three buckets:

- **copy as-is** — module tuning parameters and binding-profile templates (`scautoloc.cfg`, `scfakequake.cfg`, `etc/key/global`, `access`, `scautopick`, `scwfparam`, `slarchive`, `etc/defaults`, ...).
- **copy but flagged for manual review** — any file containing a redacted secret or a machine-specific absolute path (`/home/...`). This reliably catches `scmaster.cfg` (DB password) and things like `sceasyquake.cfg`'s `picker.model_path` or `scfakequake.cfg`'s `fakequake_root`/`model_path`/`waveform_source`.
- **skipped entirely** — `etc/key/station_*`, `etc/key/seedlink/profile_*`, `etc/inventory/*`, and `share/scautoloc/{station-locations,grid,station}.conf`. These describe a *specific* set of stations and are exactly what `bbox_station_service.py` (re)generates for a new deployment's region — `apply` never touches them, so it's safe to run `apply` before or after running the bbox service.

Safety:

- **Dry run by default.** `apply SNAPSHOT_DIR` only prints the plan (counts + the review/skip file lists); nothing is written until you pass `-y`/`--yes`.
- **Auto-backup before writing.** On execute, the target's current `etc/` is snapshotted first to `seiscomp_snapshots/pre-apply_<host>_<timestamp>/`, so an apply can be undone by re-applying the backup.
- **Version check.** Warns if the snapshot's SeisComP `Framework` version doesn't match the target's — config keys can shift across releases.

```
python examples/status_check.py apply seiscomp_snapshots/prod_20260701_090000          # preview
python examples/status_check.py apply seiscomp_snapshots/prod_20260701_090000 -y       # apply for real
```

---

## bbox_station_service.py

A small Flask web UI for provisioning stations by geographic region:

```
pip install flask obspy
python examples/bbox_station_service.py
```

Open `http://localhost:5000`, draw a bounding box on the basemap, and it will:

1. Query IRIS FDSN for stations in the box active in the last 24 h.
2. Check which are available on the primary/secondary SeedLink servers.
3. Write StationXML to `$SEISCOMP_ROOT/etc/inventory/`.
4. Write `etc/key/station_*` bindings and `etc/key/seedlink/profile_*` chain profiles.
5. Run `scinv sync` + `seiscomp update-config` + `seiscomp restart seedlink`, and regenerate `share/scautoloc/station-locations.conf`.

This is the tool responsible for everything `status_check.py apply` intentionally skips.

---

## Standing up a new system in minutes

The two tools are designed to compose: snapshot the config you already trust, let `bbox_station_service.py` provision the *new* region's stations, then apply the rest of the config template on top.

1. **Snapshot the reference system** (e.g. production):

   ```
   python examples/status_check.py snapshot -o seiscomp_snapshots/prod_reference
   ```

2. **Run `bbox_station_service.py` on the new system** to build its own station inventory/bindings for the new deployment's region:

   ```
   python examples/bbox_station_service.py
   # open http://localhost:5000, draw the new region's bounding box, apply
   ```

3. **Apply the snapshot for everything else** (order relative to step 2 doesn't matter — `apply` never overwrites what `bbox_station_service.py` owns):

   ```
   python examples/status_check.py apply seiscomp_snapshots/prod_reference     # preview first
   python examples/status_check.py apply seiscomp_snapshots/prod_reference -y  # then apply
   ```

4. **Fix the handful of flagged files** (`apply`'s dry-run output lists them explicitly — typically 4: `scmaster.cfg`, `sceasyquake.cfg`, `scfakequake.cfg`, `etc/defaults/scfakequake.cfg`):
   - Re-enter the real DB password in `scmaster.cfg`.
   - Point `sceasyquake.cfg`'s `picker.model_path` and `scfakequake.cfg`'s `fakequake_root`/`model_path`/`waveform_source` at the new machine's actual paths.

5. **Reload SeisComP:**

   ```
   seiscomp update-config && seiscomp restart
   ```

6. **Confirm the new system is healthy:**

   ```
   python examples/status_check.py check
   ```

If `check` reports issues, `compare` the new system's own snapshot against the reference to see exactly what still differs:

```
python examples/status_check.py snapshot -o seiscomp_snapshots/new_system
python examples/status_check.py compare seiscomp_snapshots/prod_reference seiscomp_snapshots/new_system
```
