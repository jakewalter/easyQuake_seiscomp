#!/usr/bin/env python3
"""Republish corrected locations for scpyocto-preferred-origin events in the
last 30 days, using the now-fixed octree gridding parameters (see
scpyocto.cfg / scpyocto.py: min_node_size, min_node_size_location,
location_split_depth, location_split_return, refinement_iterations).

For each event, re-associates that origin's exact original pick set with a
freshly-built PyOcto associator (matching the live, fixed scpyocto.cfg) and
publishes a new Origin with the corrected location/time, reusing the SAME
Arrival/pick references. Publishing follows the same pattern as scpyocto's
own _publish_event() and screloc's send(): create + Notifier + send to
LOCATION group. Preferred-origin selection is left to scevent's own normal
processing, exactly as for any other origin in this system -- no origin is
force-set as preferred here.

Rate-limited to avoid flooding scmaster (same concern documented in
gap_recovery's README).
"""
import os
import sys
import time
import traceback

import pymysql
import pandas as pd
import pyocto

import seiscomp.client as sc_client
import seiscomp.core as sc_core
import seiscomp.datamodel as dm
import seiscomp.logging as sc_log
import seiscomp.math as sc_math

MYSQL = dict(host="127.0.0.1", port=3306, user="sysop", password="sysop", database="seiscomp",
             cursorclass=pymysql.cursors.DictCursor)

# Matches the deployed ~/seiscomp/etc/scpyocto.cfg (confirmed via its own
# restart log: "vp=6.00 vs=3.46 tol=2.50 n_picks=4 n_p=3 n_s=0 n_ps=0
# match_tol=2.00" and "gridding min_node_size=10.00km
# min_node_size_location=0.50km split_depth=8 split_return=4
# refinement_iterations=4").
CFG = dict(
    lat_min=28.0, lat_max=35.0, lon_min=-109.0, lon_max=-99.0,
    depth_min=0.0, depth_max=50.0,
    vp=6.0, vs=3.46, tolerance=2.5,
    time_before=60.0,
    n_picks=4, n_p_picks=3, n_s_picks=0, n_p_and_s_picks=0,
    pick_match_tolerance=2.0, min_interevent_time=3.0, time_slicing=1200.0,
    min_node_size=10.0, min_node_size_location=0.5,
    location_split_depth=8, location_split_return=4, refinement_iterations=4,
)

# 2026-08-06: was 10.0. That rate, combined with the cron-based
# relocate_new_origins.py wrapper reacting to each new origin with its own
# screloc subprocess+connection every 2 min, drove scmaster to crash outright
# (full messaging-bus outage, not just one client stalling). Dropped
# significantly; also pause the cron wrapper for the duration of any bulk run
# (see nonlinloc/README or the crontab entry) rather than relying on rate
# alone to avoid a repeat.
RATE_PER_SEC = 2.0  # origins published per second -- keep scmaster happy


def build_associator():
    conn = pymysql.connect(**MYSQL)
    cur = conn.cursor()
    cur.execute("""
        SELECT CONCAT(n.code, '.', s.code) AS id, s.latitude, s.longitude, s.elevation
        FROM Station s JOIN Network n ON n._oid = s._parent_oid
        WHERE s.latitude IS NOT NULL AND s.longitude IS NOT NULL
    """)
    stations_df = pd.DataFrame(cur.fetchall()).drop_duplicates(subset="id").reset_index(drop=True)
    cur.close()
    conn.close()

    velocity_model = pyocto.VelocityModel0D(CFG["vp"], CFG["vs"], CFG["tolerance"])
    assoc = pyocto.OctoAssociator.from_area(
        lat=(CFG["lat_min"], CFG["lat_max"]),
        lon=(CFG["lon_min"], CFG["lon_max"]),
        zlim=(CFG["depth_min"], CFG["depth_max"]),
        velocity_model=velocity_model,
        time_before=CFG["time_before"],
        n_picks=CFG["n_picks"], n_p_picks=CFG["n_p_picks"],
        n_s_picks=CFG["n_s_picks"], n_p_and_s_picks=CFG["n_p_and_s_picks"],
        pick_match_tolerance=CFG["pick_match_tolerance"],
        min_interevent_time=CFG["min_interevent_time"],
        time_slicing=CFG["time_slicing"],
        min_node_size=CFG["min_node_size"],
        min_node_size_location=CFG["min_node_size_location"],
        location_split_depth=CFG["location_split_depth"],
        location_split_return=CFG["location_split_return"],
        refinement_iterations=CFG["refinement_iterations"],
    )
    assoc.transform_stations(stations_df)
    return assoc, stations_df


def fetch_targets(mysql_conn):
    """Resumable: only returns events that don't already have a
    nonlinloc_republish candidate, regardless of what's currently preferred
    (an event's preferred origin may have been hijacked by the screloc
    incident since the first pass, so this doesn't restrict to
    preferredOriginID being scpyocto-authored -- it looks at ANY
    scpyocto-authored origin referenced to the event)."""
    cur = mysql_conn.cursor()
    cur.execute("""
        SELECT DISTINCT po_e.publicID AS event_id, e._oid AS event_oid
        FROM Event e
        JOIN PublicObject po_e ON po_e._oid = e._oid
        JOIN OriginReference orf ON orf._parent_oid = e._oid
        JOIN PublicObject po_o ON po_o.publicID = orf.originID
        JOIN Origin o ON o._oid = po_o._oid
        WHERE o.creationInfo_author = 'scpyocto'
          AND o.time_value >= NOW() - INTERVAL 30 DAY
          AND NOT EXISTS (
              SELECT 1 FROM OriginReference orf2
              JOIN PublicObject po2 ON po2.publicID = orf2.originID
              JOIN Origin o2 ON o2._oid = po2._oid
              WHERE orf2._parent_oid = e._oid AND o2.creationInfo_author = 'nonlinloc_republish'
          )
    """)
    pending_events = cur.fetchall()
    print(f"(resumable query: {len(pending_events)} event(s) still need a republish pass)")

    # Second phase: pick the best-quality scpyocto origin per pending event
    # (avoids a GROUP BY/HAVING that MySQL's ONLY_FULL_GROUP_BY mode rejects).
    events = []
    for ev in pending_events:
        cur.execute("""
            SELECT po_o.publicID AS origin_id, o._oid AS origin_oid
            FROM OriginReference orf
            JOIN PublicObject po_o ON po_o.publicID = orf.originID
            JOIN Origin o ON o._oid = po_o._oid
            WHERE orf._parent_oid = %s AND o.creationInfo_author = 'scpyocto'
            ORDER BY o.quality_usedPhaseCount DESC
            LIMIT 1
        """, (ev["event_oid"],))
        best = cur.fetchone()
        if best:
            events.append({"event_id": ev["event_id"], "event_oid": ev["event_oid"],
                           "origin_id": best["origin_id"], "origin_oid": best["origin_oid"]})
    cur.close()

    origin_oids = [e["origin_oid"] for e in events]
    by_origin = {}
    BATCH = 500
    for i in range(0, len(origin_oids), BATCH):
        batch = origin_oids[i:i + BATCH]
        fmt = ",".join(str(x) for x in batch)
        cur = mysql_conn.cursor()
        cur.execute(f"""
            SELECT a._parent_oid AS origin_oid, a.pickID, a.phase_code,
                   pk.waveformID_networkCode AS net, pk.waveformID_stationCode AS sta
            FROM Arrival a
            JOIN PublicObject po_pk ON po_pk.publicID = a.pickID
            JOIN Pick pk ON pk._oid = po_pk._oid
            WHERE a._parent_oid IN ({fmt})
        """)
        for r in cur.fetchall():
            by_origin.setdefault(r["origin_oid"], []).append(r)
        cur.close()

    return events, by_origin


class RepublishApp(sc_client.Application):
    def __init__(self, events, by_origin, assoc, stations_df, station_lat_lon):
        _name = f"nll_republish_{os.getpid()}"
        _argv = [_name, "--console=1"]
        sc_client.Application.__init__(self, len(_argv), _argv)
        self.setMessagingEnabled(True)
        self.setDatabaseEnabled(False, False)
        self.setPrimaryMessagingGroup("LOCATION")
        self._events = events
        self._by_origin = by_origin
        self._assoc = assoc
        self._stations_df = stations_df
        self._station_lat_lon = station_lat_lon
        self._published = 0
        self._skipped = 0
        self._errors = 0

    def run(self):
        min_interval = 1.0 / RATE_PER_SEC
        for ev in self._events:
            t0 = time.time()
            try:
                self._process_one(ev)
            except Exception as exc:
                self._errors += 1
                sc_log.error(f"republish: {ev['event_id']}: {exc}\n{traceback.format_exc()}")
            elapsed = time.time() - t0
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            # Periodic cool-down: give scmaster breathing room every 500
            # published, independent of the steady-state rate above. This is
            # extra insurance after the 2026-08-06 scmaster crash -- a
            # constant rate alone wasn't enough once the cron relocate
            # wrapper's own connection churn compounded on top of it.
            if self._published > 0 and self._published % 500 == 0:
                sc_log.info(f"republish: cool-down pause at {self._published} published")
                time.sleep(15.0)

        sc_log.info(f"republish: done -- published={self._published} "
                    f"skipped={self._skipped} errors={self._errors}")
        print(f"DONE published={self._published} skipped={self._skipped} errors={self._errors}",
              flush=True)
        return True

    def _process_one(self, ev):
        arr = self._by_origin.get(ev["origin_oid"], [])
        picks_data = []
        for a in arr:
            sid = f'{a["net"]}.{a["sta"]}'
            if sid not in self._station_lat_lon:
                continue
            phase = a["phase_code"] if a["phase_code"] in ("P", "S") else "P"
            picks_data.append({"pickID": a["pickID"], "station": sid, "phase": phase})

        if len(picks_data) < 3:
            self._skipped += 1
            return

        # Need original pick TIMES for the associator input -- fetch once
        # more here (kept separate from the bulk fetch to avoid holding a
        # second huge dict in memory across 16k events).
        pick_ids = [p["pickID"] for p in picks_data]
        times = self._fetch_pick_times(pick_ids)

        rows = []
        for p in picks_data:
            t = times.get(p["pickID"])
            if t is None:
                continue
            rows.append({"time": t, "station": p["station"], "phase": p["phase"]})
        if len(rows) < 3:
            self._skipped += 1
            return

        picks_df = pd.DataFrame(rows)
        try:
            out_events, assignments = self._assoc.associate(picks_df, self._stations_df)
        except Exception:
            self._skipped += 1
            return
        if out_events is None or len(out_events) == 0:
            self._skipped += 1
            return
        self._assoc.transform_events(out_events)

        best_idx = out_events["idx"].map(
            lambda idx: (assignments["event_idx"] == idx).sum()
        ).idxmax()
        best = out_events.loc[best_idx]

        self._publish(ev, picks_data, best)

    _pick_time_cache_conn = None

    def _fetch_pick_times(self, pick_ids):
        if self._pick_time_cache_conn is None:
            RepublishApp._pick_time_cache_conn = pymysql.connect(**MYSQL)
        conn = self._pick_time_cache_conn
        cur = conn.cursor()
        fmt = ",".join(f"'{pid}'" for pid in pick_ids)
        cur.execute(f"""
            SELECT po.publicID, pk.time_value, pk.time_value_ms
            FROM Pick pk JOIN PublicObject po ON po._oid = pk._oid
            WHERE po.publicID IN ({fmt})
        """)
        out = {}
        for r in cur.fetchall():
            out[r["publicID"]] = r["time_value"].timestamp() + r["time_value_ms"] / 1e6
        cur.close()
        return out

    def _publish(self, ev, picks_data, best):
        pub_id = f"pyocto_reloc.{int(time.time()*1e6)}.{self._published:06d}"
        origin = dm.Origin.Create(pub_id)
        origin.setLatitude(dm.RealQuantity(float(best["latitude"])))
        origin.setLongitude(dm.RealQuantity(float(best["longitude"])))
        origin.setDepth(dm.RealQuantity(float(best.get("depth", 0.0))))
        origin.setTime(dm.TimeQuantity(sc_core.Time(float(best["time"]))))
        origin.setEvaluationMode(dm.AUTOMATIC)
        origin.setMethodID("pyocto_reloc")
        origin.setEarthModelID(f'0D_vp{CFG["vp"]:.2f}_vs{CFG["vs"]:.2f}_griddingfix')

        ci = dm.CreationInfo()
        ci.setAuthor("nonlinloc_republish")
        try:
            ci.setAgencyID(self.agencyID())
        except Exception:
            pass
        ci.setCreationTime(sc_core.Time.GMT())
        origin.setCreationInfo(ci)

        dm.Notifier.Enable()
        ep = dm.EventParameters()
        ep.add(origin)
        for p in picks_data:
            arrival = dm.Arrival()
            arrival.setPickID(p["pickID"])
            arrival.setPhase(dm.Phase(p["phase"]))
            arrival.setWeight(1.0)
            try:
                sta_coords = self._station_lat_lon.get(p["station"])
                if sta_coords:
                    dist_deg, az, _ = sc_math.delazi_wgs84(
                        float(best["latitude"]), float(best["longitude"]),
                        sta_coords[0], sta_coords[1],
                    )
                    arrival.setDistance(dist_deg)
                    arrival.setAzimuth(az)
            except Exception:
                pass
            origin.add(arrival)
        dm.Notifier.Disable()
        notifier_msg = dm.Notifier.GetMessage(True)

        if notifier_msg and notifier_msg.size() > 0:
            self.connection().send("LOCATION", notifier_msg)
            self._published += 1
            if self._published % 200 == 0:
                sc_log.info(f"republish: {self._published} published so far "
                            f"(latest: {ev['event_id']} -> {pub_id})")


def main():
    conn = pymysql.connect(**MYSQL)
    events, by_origin = fetch_targets(conn)
    conn.close()
    print(f"{len(events)} scpyocto-preferred events in last 30 days", flush=True)

    assoc, stations_df = build_associator()
    station_lat_lon = {
        row["id"]: (row["latitude"], row["longitude"])
        for _, row in stations_df.iterrows()
    }
    print(f"{len(stations_df)} stations loaded", flush=True)

    app = RepublishApp(events, by_origin, assoc, stations_df, station_lat_lon)
    app()


if __name__ == "__main__":
    main()
