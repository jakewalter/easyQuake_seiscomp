scPhasePaPy
===========

SeisComP module wrapper for easyQuake/phasepapy associator.

Config keys (in `scphasepapy.cfg`):

- `phasepapy.db_dir`
- `phasepapy.db_assoc`
- `phasepapy.db_tt`
- `phasepapy.max_km`
- `phasepapy.aggregation`
- `phasepapy.aggr_norm`
- `phasepapy.assoc_ot_uncert`
- `phasepapy.nsta_declare`
- `phasepapy.cutoff_outlier`
- `phasepapy.loc_uncert_thresh`
- `phasepapy.cycle_seconds`
- `phasepapy.max_pick_age`

Run as SeisComP module:

```
cp scphasepapy.py $(seiscomp exec --path)/scphasepapy
seiscomp enable scphasepapy
seiscomp start scphasepapy
```
