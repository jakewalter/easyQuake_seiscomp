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
streams configured in `scrttv`:

```ini
# All HH and EH vertical channels in network CI
streams.codes = CI.*.*.HHZ,CI.*.*.EHZ
```

Wildcards follow SeedLink `NET.STA.LOC.CHA` conventions.

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
| `import sceasyquake` fails | Package not installed | `pip3 install -e .` |
| SeedLink connection refused | SeedLink not running | `seiscomp start seedlink` |
| YAML files accumulate in `~/sceasyquake/picks/` | SeisComP bindings absent | Ensure SC ≥5 Python bindings importable |
| Very slow inference | GPU not in use | Set `picker.device = cuda` |
