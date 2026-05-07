# Integration into SeisComP workflow

`sceasyquake` replaces `scautopick` as the phase-picking module.  It connects
to the same messaging bus and publishes `DataModel.Pick` objects to the `PICK`
group, so all downstream modules (`scassoc`, `scamp`, `scmag`, `scevtlog`, …)
receive picks without any additional configuration.

## Architecture overview

```
SeedLink server (localhost:18000)
        │  ObsPy SeedLink subscription
        ▼
  SeisCompStream  ─── queue ───►  StreamWorker
                                        │
                                        │  PhaseNetPredictor
                                        │  (PhaseNet / easyQuake / SeisBench)
                                        │
                                        ▼
                                  PickUploader
                                        │
                    ┌───────────────────┴────────────────────┐
                    ▼  SeisComP bindings present?            ▼
           DataModel.Pick                            YAML fallback
           + Notifier → PICK group              ~/sceasyquake/picks/
```

## Enabling sceasyquake / disabling scautopick

```bash
# Install sceasyquake first (see INSTALL.md)
bash install.sh

# Stop and disable scautopick (avoids duplicate picks)
seiscomp stop scautopick
seiscomp disable scautopick

# Enable and start sceasyquake
seiscomp enable sceasyquake
seiscomp update-config sceasyquake
seiscomp start sceasyquake
```

## Verifying picks reach downstream modules

```bash
# Dump picks as they arrive:
seiscomp exec scdumppicks -H localhost

# Or watch in the map view:
scmv
```

`scassoc` associates picks from `sceasyquake` automatically because it
listens on the `PICK` group regardless of which module produced the picks.

## Stream selection

Set `streams.codes` in `$SEISCOMP_ROOT/etc/sceasyquake.cfg` to match the
streams configured in `scrttv`. Replace `NET` with your actual network code.

```ini
# All HH vertical channels — replace NET with your network code (e.g. GE, IU, CI)
streams.codes = NET.????.*.HH?

# Multiple channel types
streams.codes = NET.????.*.HHZ,NET.????.*.EHZ

# Explicit station list
streams.codes = NET.STA1..HHZ,NET.STA2..HHZ
```

**SeedLink wildcard notes** — not all wildcards work at every position:

| Field | Safe pattern | Notes |
|---|---|---|
| NET | explicit code only | `*` may be silently ignored by some servers |
| STA | `?` per character, e.g. `????` | `*` works on SeisComP SeedLink but is not in the protocol spec |
| LOC | empty string (no `*`) | The installer converts `*`/`?` to empty automatically |
| CHA | `?` freely, e.g. `HH?` or `??Z` | Works as expected |

## GPU acceleration

```ini
picker.device = cuda
```

Ensure CUDA / cuDNN are installed for the Python environment used by
`seiscomp-python`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No picks in scmv | Module not started | `seiscomp status sceasyquake` |
| `import sceasyquake` fails | Package not installed | `$SC_PYTHON -m pip install -e /path/to/easyQuake_seiscomp/sceasyquake` |
| SeedLink connection refused | SeedLink not running | `seiscomp start seedlink` |
| YAML files accumulate in `~/sceasyquake/picks/` | SeisComP bindings absent | Ensure SC ≥5 Python bindings importable |
| Very slow inference | GPU not in use | Set `picker.device = cuda` |
