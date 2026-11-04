#!/usr/bin/env python3
"""Scan the SeisComP MySQL database for long silences in event creation.

Simple standalone diagnostic: query ``Event.creationInfo_creationTime`` (when
each event was first created, not the estimated origin time -- so later
relocations/reprocessing of an old event never show up as "new" activity)
over the last N days, and report every silence between consecutive events
that exceeds a threshold. Useful for spotting periods when the pipeline
(scmaster, scautoloc/scpyocto, scevent, ...) was down or stalled, as opposed
to `gap_recovery.py`'s archive scan, which looks for missing waveform data.

Caveat: a gap here does not by itself prove an outage -- real seismicity is
not evenly spaced, and a quiet network can go hours without a new event
under completely normal operation. Treat gaps as candidates worth
cross-checking against module logs (`~/.seiscomp/log/`) rather than as
confirmed incidents.

Usage:
    python3 event_outage_scan.py                       # last 30 days, 2h threshold
    python3 event_outage_scan.py --days 7 --min-gap-hours 0.5
    python3 event_outage_scan.py --sc-root ~/seiscomp
"""
import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import pymysql
except ImportError:
    print("pymysql is required -- install it with: pip install pymysql", file=sys.stderr)
    sys.exit(1)


def db_credentials(sc_root: Path):
    """Read (user, password, host, database) out of scmaster.cfg.

    Falls back to the common sysop/sysop/localhost/seiscomp default (as used
    by scmaster.cfg out of the box) if the config can't be parsed, so this
    still works against a stock install.
    """
    cfg_path = sc_root / "etc" / "scmaster.cfg"
    try:
        for line in cfg_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("queues.production.processors.messages.dbstore.read"):
                db_url = line.split("=", 1)[1].strip()
                m = re.match(r"(?:mysql://)?([^:]+):([^@]+)@([^/]+)/(.+)", db_url)
                if m:
                    user, password, host, database = m.groups()
                    return user, password, host, database
    except OSError:
        pass
    return "sysop", "sysop", "localhost", "seiscomp"


def find_event_gaps(conn, lookback_days: float, min_gap_hours: float):
    """Return (total_events, gaps) where each gap is (start, end, duration_hours).

    SeisComP stores creationInfo_creationTime as a naive UTC DATETIME --
    bind naive UTC datetime strings, not epoch floats, or MySQL silently
    matches zero rows regardless of what's actually in the table.
    """
    now_dt = datetime.now(timezone.utc)
    start_dt = now_dt - timedelta(days=lookback_days)
    now_naive, start_naive = now_dt.replace(tzinfo=None), start_dt.replace(tzinfo=None)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT creationInfo_creationTime
        FROM   Event
        WHERE  creationInfo_creationTime >= %s
          AND  creationInfo_creationTime <= %s
        ORDER  BY creationInfo_creationTime ASC
        """,
        (start_naive, now_naive),
    )
    rows = cur.fetchall()
    cur.close()

    min_gap_s = min_gap_hours * 3600.0
    times = [r[0].replace(tzinfo=timezone.utc) for r in rows]

    if not times:
        return 0, [(start_dt, now_dt, lookback_days * 24.0)]

    gaps = []
    if (times[0] - start_dt).total_seconds() > min_gap_s:
        gaps.append((start_dt, times[0], (times[0] - start_dt).total_seconds() / 3600.0))

    for t_prev, t_next in zip(times, times[1:]):
        dur_s = (t_next - t_prev).total_seconds()
        if dur_s > min_gap_s:
            gaps.append((t_prev, t_next, dur_s / 3600.0))

    trailing_s = (now_dt - times[-1]).total_seconds()
    if trailing_s > min_gap_s:
        gaps.append((times[-1], now_dt, trailing_s / 3600.0))

    return len(times), gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=30.0, help="Lookback window in days (default: 30)")
    ap.add_argument("--min-gap-hours", type=float, default=2.0,
                     help="Report silences longer than this many hours (default: 2)")
    ap.add_argument("--sc-root", default=str(Path.home() / "seiscomp"),
                     help="SeisComP root, used to read DB credentials from etc/scmaster.cfg")
    ap.add_argument("--host", default=None, help="Override MySQL host")
    ap.add_argument("--port", type=int, default=3306, help="MySQL port (default: 3306)")
    args = ap.parse_args()

    user, password, host, database = db_credentials(Path(args.sc_root))
    if args.host:
        host = args.host

    conn = pymysql.connect(host=host, port=args.port, user=user, password=password,
                            database=database, cursorclass=pymysql.cursors.Cursor)
    try:
        total, gaps = find_event_gaps(conn, args.days, args.min_gap_hours)
    finally:
        conn.close()

    print(f"{total} event(s) in the last {args.days:.0f} day(s).")
    if not gaps:
        print(f"No silence longer than {args.min_gap_hours:g}h found.")
        return

    print(f"{len(gaps)} silence(s) longer than {args.min_gap_hours:g}h:")
    for start, end, hours in gaps:
        print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')} -> {end.strftime('%Y-%m-%d %H:%M:%S')}"
              f"  ({hours:.1f}h)")


if __name__ == "__main__":
    main()
