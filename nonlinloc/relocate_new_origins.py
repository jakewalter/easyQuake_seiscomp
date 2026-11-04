#!/usr/bin/env python3
"""Bounded, one-shot NonLinLoc relocation pass -- the safe replacement for
running screloc as a continuously-subscribed LOCATION-group daemon.

Background: screloc subscribes to the LOCATION message group and relocates
every origin it sees there. It has no guard against reprocessing its own
output (methodID='NonLinLoc', author='screloc@<host>'), which is also
AUTOMATIC/no-evaluationStatus just like everything else with
reloc.allowAnyStatus=true -- so its own relocated origin lands right back on
the same group it's listening to, gets relocated again, forever. This
produced 17,000+ spurious origins in ~90 minutes before being caught.

This script never subscribes to anything. Each run:
  1. Queries MySQL for origins authored by a real associator (scpyocto,
     scautoloc) whose event does not already have an NLL-authored sibling
     origin referencing it -- the self-loop is structurally impossible
     because screloc's own output (methodID='NonLinLoc') is never a
     candidate input.
  2. Invokes `screloc` ONCE per run with a repeated `-O <origin-id>` flag
     for every candidate (screloc's single-origin reprocess-and-send mode
     accepts `-O` more than once and relocates all of them over one
     messaging connection), capped at --max-per-run so one invocation can
     never touch an unbounded number of origins. Falls back to the old
     one-connection-per-origin behavior only for whatever the batch didn't
     resolve (see BATCHING below).
  3. Exits. Run this on a timer (cron/systemd timer), NOT as a persistent
     process -- see gap_recovery/README.md's cron pattern for the same
     philosophy applied there.

BATCHING (added 2026-08-08): the original design opened one messaging
connection per origin (up to --max-per-run per cron tick, both real-time and
backlog lines, every 1-2 minutes, for hours). That sustained per-origin
connect/disconnect churn was confirmed twice in one evening to crash
`scmaster` itself after a few hours of otherwise-normal operation (rapid
"Accepted new client"/"removed session" churn in scmaster.log immediately
preceding both crashes, with no graceful shutdown logged) -- a resource leak
under connection churn, not anything specific to this script's logic, but
this script is what generates the sustained churn. `screloc -O` can be
repeated in one invocation ("Keep N messages in the backlog" for N origins
relocated over a single connection), so a full batch of --max-per-run
origins now costs ONE connection instead of up to twenty. Success is
verified afterward by re-querying MySQL for which candidates still lack an
NLL-authored sibling (the exact same test that made them candidates in the
first place) rather than trusting screloc's overall exit code, which stays
0 even when some origins in the batch got "Empty location". Whatever is
still pending after the batch (a real failure, or the batch call itself
timing out) falls back to the original one-at-a-time loop, so per-origin
timeout/backoff/permanent-skip tracking is unchanged -- that path should
only ever cover a small remainder, not the whole candidate list.

Usage:
    python3 relocate_new_origins.py [--max-per-run 20] [--lookback-hours 2]
                                     [--order desc|asc] [--dry-run]

Suggested crontab -- TWO separate lines, not one with a wide --lookback-hours:
a single DESC-ordered query always prefers the newest un-relocated origins,
so as long as real-time events keep arriving they fill every --max-per-run
slot before the query ever reaches back into an older backlog (confirmed
2026-08-06: widening --lookback-hours alone left a 16k-event backlog
completely untouched for hours). The ASC-ordered line drains the backlog
oldest-first, independently and without starving the real-time line.

The two lines run on offset schedules (even vs. odd minutes), not both on
`*/2`: firing simultaneously creates a connection burst against scmaster,
which under load (2026-08-07: sustained load average ~70 on 24 cores from
unrelated processes) measurably worsened "Connection error: Failed to
connect" retries from screloc. Offsetting them spreads that load out.

    # real-time: newest-first, ~2 min behind live detection (even minutes)
    */2 * * * * /home/jwalter/anaconda3/bin/python3 \\
        /home/jwalter/easyQuake_seiscomp/nonlinloc/relocate_new_origins.py \\
        --max-per-run 20 >> /home/jwalter/easyQuake_seiscomp/nonlinloc/relocate.log 2>&1

    # backlog catchup: oldest-first, 31-day window (odd minutes)
    1-59/2 * * * * /home/jwalter/anaconda3/bin/python3 \\
        /home/jwalter/easyQuake_seiscomp/nonlinloc/relocate_new_origins.py \\
        --max-per-run 20 --lookback-hours 744 --order asc \\
        >> /home/jwalter/easyQuake_seiscomp/nonlinloc/relocate_backlog.log 2>&1

Origins that fail (timeout or non-zero exit) MAX_CONSECUTIVE_FAILURES times
are backed off for BACKOFF_SECONDS (state in relocate_failures.json next to
this script) so a handful of stuck origins can't monopolize the front of the
queue every cycle -- see _in_backoff().
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

import pymysql

MYSQL = dict(host="127.0.0.1", port=3306, user="sysop", password="sysop", database="seiscomp",
             cursorclass=pymysql.cursors.DictCursor)

LOCK_DIR = "/tmp"

# screloc has no cap on messaging-connection retries -- when scmaster is slow
# to accept new connections (observed 2026-08-07: sustained system load
# starving scmaster's ability to service new clients), `screloc -O` retries
# "Connection error: Failed to connect" every 2s forever rather than giving
# up. Confirmed by letting one run for 5 minutes uninterrupted: it never
# stopped retrying on its own. So the wrapper must fail fast (SCRELOC_TIMEOUT
# well above the ~10s a real relocation takes, well below minutes of
# connection retries) and must remember origins that keep failing so they
# stop monopolizing the front of the queue every single cycle.
SCRELOC_TIMEOUT = 20
FAILURE_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relocate_failures.json")
MAX_CONSECUTIVE_FAILURES = 3
BACKOFF_SECONDS = 1800  # 30 min cooldown once an origin hits the failure cap

# Timeout for the single batched screloc call that carries every candidate's
# -O flag over one connection (see BATCHING in the module docstring). Real
# relocations take ~0.25s each even under load (measured 2026-08-08), so this
# is generous headroom, not a tight estimate -- a batch that's genuinely this
# slow is already in trouble and should fall back to the per-origin path.
BATCH_TIMEOUT_BASE_S = 15
BATCH_TIMEOUT_PER_ORIGIN_S = 3


def _load_failure_state():
    try:
        with open(FAILURE_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_failure_state(state):
    tmp = FAILURE_STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, FAILURE_STATE_FILE)


def _in_backoff(state, origin_id, now):
    rec = state.get(origin_id)
    if not rec:
        return False
    if rec.get("permanent"):
        return True
    return rec["count"] >= MAX_CONSECUTIVE_FAILURES and (now - rec["last"]) < BACKOFF_SECONDS


# screloc logs "processing failed - Empty location" to stderr but still
# exits 0 when NLLoc finds no usable location for an origin's fixed arrival
# set (commonly: phase/station filtering drops it below LOCMETH's
# minNumberPhases, see run/screloc_nll.in) -- unlike a transient connection/
# timeout failure, this cannot change on retry. NLLoc's own diagnostic text
# ("too few observations to locate...") is unreliable to key off of: it's
# written by the NLLoc subprocess to stdout, which is not line-buffered
# under a non-tty pipe and was observed to go missing from a captured run
# entirely (confirmed 2026-08-08) even though the failure still happened.
# "Empty location" on stderr is emitted by screloc's own logging, not the
# NLLoc subprocess, and was present in every reproduction. Without this
# check the origin gets popped out of backoff tracking as if it had
# succeeded and requeued forever, permanently occupying a --max-per-run slot
# every cycle -- confirmed 2026-08-08 as why the oldest-first backlog line
# kept resurfacing the same candidates instead of draining.

# Invoked via `seiscomp exec` (not the raw binary) so LD_LIBRARY_PATH/
# PYTHONPATH get set correctly under cron's stripped-down environment --
# calling the binary directly works in an interactive shell (which sources
# the profile that happens to set these) but fails under cron with
# "error while loading shared libraries: libseiscomp_client.so.17: cannot
# open shared object file" (confirmed 2026-08-06: every cron-triggered
# relocation attempt since deployment had silently failed this way).
SEISCOMP_BIN = "/home/jwalter/seiscomp/bin/seiscomp"
SEISCOMP_ROOT = "/home/jwalter/seiscomp"

# Every author whose output is eligible input for NLL relocation. screloc's
# own output ('screloc@home3', methodID='NonLinLoc') must NEVER appear here --
# that is precisely the self-loop that caused the incident. The NOT EXISTS
# clause in find_candidates() is the actual guard against reprocessing NLL
# output; this tuple only controls which *un-relocated* origins are worth
# picking up.
#
# 'nonlinloc_republish' (PyOcto re-run with corrected octree gridding, see
# ../nonlinloc/republish_scpyocto.py) is included because it still exhibits a
# residual defect: a subset of events collapse onto a near-constant latitude
# regardless of true location (confirmed 2026-08-06 by comparing 6 such
# events against real NonLinLoc/DB1D relocations, which scattered them across
# a genuine ~22km spread). Without this, any event whose preferred origin was
# promoted to 'nonlinloc_republish' by the bulk republish became permanently
# ineligible here, since it no longer matched 'scpyocto'/'scautoloc@home3'.
REAL_ASSOCIATOR_AUTHORS = ("scpyocto", "scautoloc@home3", "nonlinloc_republish")


# "Already relocated" test shared by find_candidates() and
# check_still_pending(): does this origin have an NLL-methodID sibling that
# shares at least one of its arrivals' picks? A relocation reuses the exact
# same picks as its source origin, so this is a direct, race-free signal --
# unlike testing via Event.preferredOriginID / OriginReference, which
# depends on *scevent* choosing to associate/promote the new origin.
# scevent does not do that for a relocation NLLoc itself flags
# evaluationStatus='rejected' (e.g. "max prob location on grid boundary" for
# an origin right at the edge of LOCGRID) -- confirmed 2026-08-08: such an
# origin's OriginReference/preferredOriginID state never changes, no matter
# how long you wait, even though screloc successfully relocated it. Relying
# on OriginReference here meant these origins were silently reprocessed
# *every single cron cycle forever*, each time creating another redundant
# rejected NLL origin -- the same class of unbounded-origin-growth incident
# this whole script exists to prevent, just via a different mechanism.
ALREADY_RELOCATED_SQL = """
    NOT EXISTS (
        SELECT 1 FROM Arrival a1
        JOIN Arrival a2 ON a2.pickID = a1.pickID AND a2._parent_oid != o._oid
        JOIN Origin nll ON nll._oid = a2._parent_oid
        WHERE a1._parent_oid = o._oid AND nll.methodID = 'NonLinLoc'
    )
"""


def find_candidates(conn, lookback_hours, fetch_limit, order="desc"):
    # order='desc' (default): newest-first, for low-latency relocation of
    # fresh events -- this is what every routine cron invocation should use.
    # order='asc': oldest-first, for a *separate* backlog-catchup cron line.
    # A single DESC-ordered pass with a wide --lookback-hours cannot drain a
    # historical backlog: real-time events are always newer than backlog
    # events, so they refill all `max_per_run` slots every cycle and the
    # backlog is starved forever (observed 2026-08-06: widening lookback to
    # 744h alone did not touch the 16k-event nonlinloc_republish backlog).
    if order not in ("asc", "desc"):
        raise ValueError("order must be 'asc' or 'desc'")
    cur = conn.cursor()
    authors = ",".join(f"'{a}'" for a in REAL_ASSOCIATOR_AUTHORS)
    cur.execute(f"""
        SELECT po_o.publicID AS origin_id, po_e.publicID AS event_id,
               o.creationInfo_author AS author
        FROM Event e
        JOIN PublicObject po_e ON po_e._oid = e._oid
        JOIN PublicObject po_o ON po_o.publicID = e.preferredOriginID
        JOIN Origin o ON o._oid = po_o._oid
        WHERE o.creationInfo_author IN ({authors})
          AND o.time_value >= NOW() - INTERVAL {int(lookback_hours)} HOUR
          AND {ALREADY_RELOCATED_SQL}
        ORDER BY o.time_value {order.upper()}
        LIMIT {int(fetch_limit)}
    """)
    return cur.fetchall()


def check_still_pending(conn, origin_ids):
    """Return the subset of origin_ids that still lack an NLL sibling.

    Same test as find_candidates(), restricted to a specific set of origin
    IDs -- used to find out which candidates a batched screloc call
    actually resolved without trusting its overall exit code (which stays 0
    even when some origins in the batch got "Empty location").
    """
    if not origin_ids:
        return set()
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(origin_ids))
    cur.execute(f"""
        SELECT po_o.publicID AS origin_id
        FROM PublicObject po_o
        JOIN Origin o ON o._oid = po_o._oid
        WHERE po_o.publicID IN ({placeholders})
          AND {ALREADY_RELOCATED_SQL}
    """, tuple(origin_ids))
    pending = {r["origin_id"] for r in cur.fetchall()}
    cur.close()
    return pending


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-per-run", type=int, default=20,
                    help="Hard cap on origins relocated in one invocation (default: 20)")
    ap.add_argument("--lookback-hours", type=float, default=2.0,
                    help="Only consider origins with time_value in the last N hours (default: 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List candidates without invoking screloc")
    ap.add_argument("--order", choices=("asc", "desc"), default="desc",
                    help="'desc' (default): newest-first, for routine real-time relocation. "
                         "'asc': oldest-first, for a separate backlog-catchup cron line -- "
                         "see find_candidates() docstring for why this must be a separate pass.")
    args = ap.parse_args()

    # A candidate that hangs long enough to eat its full 60s subprocess
    # timeout, repeated across several of the --max-per-run candidates in one
    # invocation, can make a single run take longer than the 2-minute cron
    # interval. Without this guard the next cron tick starts a second,
    # overlapping invocation of the same line -- observed 2026-08-07: two
    # concurrent `--order asc` processes running at once, each competing for
    # the same CPU-starved system and making the underlying slowness worse.
    # flock is released automatically on process exit (crash or normal), so
    # there is no stale-lock cleanup to worry about.
    lock_path = os.path.join(LOCK_DIR, f"relocate_new_origins_{args.order}.lock")
    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another --order {args.order} invocation is still running, skipping this cycle.")
        return

    now = time.time()
    failure_state = _load_failure_state()

    # Fetch a wider pool than max_per_run so that origins currently in
    # backoff (see MAX_CONSECUTIVE_FAILURES/BACKOFF_SECONDS above) can be
    # filtered out in Python while still leaving enough real candidates to
    # fill the batch -- otherwise a handful of stuck origins at the front of
    # the ASC/DESC order would occupy every slot every cycle forever.
    conn = pymysql.connect(**MYSQL)
    fetched = find_candidates(conn, args.lookback_hours, args.max_per_run * 10, args.order)
    conn.close()

    skipped_backoff = [c for c in fetched if _in_backoff(failure_state, c["origin_id"], now)]
    candidates = [c for c in fetched if not _in_backoff(failure_state, c["origin_id"], now)][:args.max_per_run]

    if skipped_backoff:
        print(f"{len(skipped_backoff)} candidate(s) in backoff (>= {MAX_CONSECUTIVE_FAILURES} "
              f"consecutive failures, cooling down): "
              + ", ".join(c["origin_id"] for c in skipped_backoff[:5])
              + (" ..." if len(skipped_backoff) > 5 else ""))

    if not candidates:
        print("No un-relocated origins found (outside backoff).")
        return

    print(f"{len(candidates)} candidate(s) for NonLinLoc relocation "
          f"(capped at {args.max_per_run}):")
    for c in candidates:
        print(f"  {c['event_id']:12s} {c['origin_id']:40s} (author={c['author']})")

    if args.dry_run:
        print("[DRY-RUN] would invoke screloc -O for each of the above.")
        return

    # Distinct -u client name per order mode: the real-time (desc) and
    # backlog (asc) cron lines fire on the same */2 schedule, and without
    # this both invoke `screloc -O` with the same default client name
    # ('screloc') at once, which the messaging system rejects for the
    # second connection ("Client name not unique") -- observed 2026-08-06,
    # 7/20 failures in one cycle.
    client_name = f"screloc_{args.order}"
    ok, failed, timed_out, permanently_skipped = 0, 0, 0, 0
    origin_ids = [c["origin_id"] for c in candidates]

    # One connection for the whole batch instead of one per origin (see
    # BATCHING in the module docstring). A single slow/hung batch call must
    # never crash the script -- fall back to treating everything as pending
    # and let the per-origin loop below sort it out.
    batch_timeout = BATCH_TIMEOUT_BASE_S + BATCH_TIMEOUT_PER_ORIGIN_S * len(origin_ids)
    batch_cmd = [SEISCOMP_BIN, "exec", "bin/screloc"]
    for origin_id in origin_ids:
        batch_cmd += ["-O", origin_id]
    batch_cmd += ["-H", "localhost", "-u", client_name, "--console=1"]

    try:
        subprocess.run(batch_cmd, capture_output=True, text=True,
                        timeout=batch_timeout, cwd=SEISCOMP_ROOT)
        conn = pymysql.connect(**MYSQL)
        try:
            pending = check_still_pending(conn, origin_ids)
        finally:
            conn.close()
    except subprocess.TimeoutExpired:
        print(f"  BATCH TIMEOUT: {len(origin_ids)} origin(s) exceeded {batch_timeout}s as a "
              f"batch, falling back to individual calls", file=sys.stderr)
        pending = set(origin_ids)

    succeeded = [oid for oid in origin_ids if oid not in pending]
    ok += len(succeeded)
    for oid in succeeded:
        failure_state.pop(oid, None)

    # Stragglers: whatever the batch didn't resolve -- a real per-origin
    # failure/"Empty location", or the whole batch call timing out -- gets
    # the original one-connection-per-origin treatment, so per-origin
    # timeout/backoff/permanent-skip tracking is unchanged. This should only
    # ever cover a small remainder of `candidates`, not all of it.
    stragglers = [c for c in candidates if c["origin_id"] in pending]
    for c in stragglers:
        origin_id = c["origin_id"]
        # A single slow/hung screloc call must never abort the rest of the
        # batch -- subprocess.run(timeout=...) raises TimeoutExpired rather
        # than returning, and an uncaught exception here used to kill the
        # whole cron invocation mid-loop, silently dropping every remaining
        # candidate for that cycle (observed 2026-08-06/07: ~95% of backlog
        # cycles were lost this way, most progress limited to a handful of
        # origins per cycle instead of --max-per-run).
        try:
            result = subprocess.run(
                [SEISCOMP_BIN, "exec", "bin/screloc", "-O", origin_id,
                 "-H", "localhost", "-u", client_name, "--console=1"],
                capture_output=True, text=True, timeout=SCRELOC_TIMEOUT, cwd=SEISCOMP_ROOT,
            )
        except subprocess.TimeoutExpired:
            timed_out += 1
            print(f"  TIMEOUT {origin_id}: exceeded {SCRELOC_TIMEOUT}s, skipped", file=sys.stderr)
            rec = failure_state.setdefault(origin_id, {"count": 0, "last": 0})
            rec["count"] += 1
            rec["last"] = now
            continue
        if result.returncode == 0 and "Empty location" not in result.stderr:
            ok += 1
            failure_state.pop(origin_id, None)
        elif result.returncode == 0:
            # screloc reports "processing failed - Empty location" but still
            # exits 0 -- NLLoc found no usable location for this origin's
            # fixed arrival set (e.g. too few phases survive filtering). This
            # will not change on retry, so mark it permanent rather than
            # requeuing it every cycle forever (see the comment above
            # _in_backoff).
            permanently_skipped += 1
            failure_state[origin_id] = {"count": 0, "last": now, "permanent": True}
        else:
            failed += 1
            print(f"  FAILED {origin_id}: {result.stderr.strip()[-300:]}", file=sys.stderr)
            rec = failure_state.setdefault(origin_id, {"count": 0, "last": 0})
            rec["count"] += 1
            rec["last"] = now

    _save_failure_state(failure_state)
    print(f"Done: {ok} relocated ({len(origin_ids) - len(stragglers)} via 1 batched connection), "
          f"{failed} failed, {timed_out} timed out, "
          f"{permanently_skipped} permanently unlocatable, "
          f"{len(skipped_backoff)} in backoff.")


if __name__ == "__main__":
    main()
