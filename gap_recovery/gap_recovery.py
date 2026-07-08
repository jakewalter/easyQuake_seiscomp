#!/usr/bin/env python3
"""
gap_recovery.py  --  SeisComP archive gap detection and ML-picker playback.

Workflow
--------
1. SCAN   - Walk the SDS archive for the last --hours hours. For every
            NET/STA/LOC/CHA channel, compare expected coverage with what
            obspy's SDSClient actually reads.  Both missing day-files and
            intra-file gaps (data dropouts) are reported.

2. FETCH  - For each gap, request the missing data from the local SeedLink
            ring-buffer via ``scart -I slink://HOST:PORT --list``.  If the
            data is still in the ring-buffer it is written into the SDS archive
            so the next step can read it.

          Fallback: any gap still unresolved after SeedLink is requested from
            the EarthScope FDSN web service (service.iris.edu).  The returned
            miniSEED is written into the SDS archive via ``scart -I`` so the
            picker step sees it as ordinary archived data.

3. PICK   - Read waveforms for every recovered (and already-present) gap window
            from the SDS archive, split into overlapping chunks, run the
            configured SeisBench ML picker, and upload the resulting picks to
            the live SeisComP messaging bus via PickUploader.  SeisComP's
            standard daemons (scautoloc -> scevent) then create Origins and
            Events in the database exactly as they would for real-time picks.

4. REPORT - Print a summary of gaps found, data fetched, picks uploaded, and
            any gaps still unrecoverable from SeedLink.

Usage
-----
    python gap_recovery.py [options]

    # Typical invocation (run from any directory):
    python /path/to/gap_recovery.py --hours 24 --archive ~/seiscomp/var/lib/archive

    # Dry-run (scan and report only, no fetch, no picking):
    python gap_recovery.py --dry-run

    # Custom model / connection:
    python gap_recovery.py --model EQTransformer --pretrained ethz \\
                            --seedlink-host localhost --seedlink-port 18000 \\
                            --sc-host localhost --sc-port 4803

Requirements
------------
    obspy >= 1.4     (gap detection, miniSEED reading, SDS client, SeedLink)
    seisbench        (ML picking backend)
    torch            (via seisbench)

    The sceasyquake library at ../sceasyquake/lib is added to sys.path
    automatically when this script is run from the gap_recovery/ folder.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap sys.path so sceasyquake is importable without installation
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SC_LIB = _HERE.parent / "sceasyquake" / "lib"
if _SC_LIB.is_dir() and str(_SC_LIB) not in sys.path:
    sys.path.insert(0, str(_SC_LIB))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gap_recovery")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Gap:
    """A time interval where waveform data is missing for one NSLC channel."""
    network:   str
    station:   str
    location:  str
    channel:   str
    starttime: "UTCDateTime"   # obspy UTCDateTime
    endtime:   "UTCDateTime"
    duration_s: float
    source: str  # "missing_file" | "intra_file"

    @property
    def nslc(self) -> str:
        return f"{self.network}.{self.station}.{self.location}.{self.channel}"

    def __str__(self):
        return (
            f"{self.nslc}  {self.starttime.strftime('%Y-%m-%dT%H:%M:%S')} -> "
            f"{self.endtime.strftime('%Y-%m-%dT%H:%M:%S')}  "
            f"({self.duration_s:.0f}s  [{self.source}])"
        )


@dataclass
class OriginGap:
    """A time interval in which SeisComP created no origins (pipeline silence)."""
    start: datetime
    end:   datetime
    duration_s: float

    def __str__(self):
        h, rem = divmod(self.duration_s, 3600)
        m = rem // 60
        return (
            f"{self.start.strftime('%Y-%m-%dT%H:%M:%S')} -> "
            f"{self.end.strftime('%Y-%m-%dT%H:%M:%S')}  "
            f"({int(h):d}h {int(m):02d}m -- no origins)"
        )


@dataclass
class RecoveryStats:
    gaps_found:        int = 0
    gaps_fetched:      int = 0   # recovered via SeedLink
    gaps_earthscope:   int = 0   # recovered via EarthScope FDSN fallback
    gaps_unresolved:   int = 0   # still missing after both fetch attempts
    picks_uploaded:    int = 0
    origin_gaps_found: int = 0   # silent periods with no origins
    errors:            List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 0. ORIGIN GAP SCAN -- detect silent periods in the SeisComP origin stream
# ---------------------------------------------------------------------------

def scan_origin_gaps(
    lookback_seconds: float,
    min_gap_s: float = 1800.0,
    mysql_host: str = "localhost",
    mysql_port: int = 3306,
    mysql_user: str = "sysop",
    mysql_password: str = "sysop",
    mysql_db: str = "seiscomp",
) -> List[OriginGap]:
    """Query the SeisComP MySQL database for origins in the last ``lookback_seconds``
    and return every inter-origin silence that exceeds ``min_gap_s``.

    Also reports the leading silence (from window start to first origin) and
    the trailing silence (from last origin to now) if they exceed the threshold.
    If there are *no* origins at all in the window the entire window is returned
    as a single gap.

    SeisComP stores origin times as ``time_value`` (DOUBLE, Unix epoch seconds)
    in the ``Origin`` table.
    """
    # Try available MySQL drivers in order of preference
    _mysql_mod = None
    for _drv in ("mysql.connector", "pymysql", "MySQLdb"):
        try:
            import importlib as _importlib
            _mysql_mod = _importlib.import_module(_drv)
            break
        except ImportError:
            continue
    if _mysql_mod is None:
        log.warning("No MySQL driver found (tried mysql.connector, pymysql, MySQLdb) -- "
                    "skipping origin gap scan.  Install one with: pip install pymysql")
        return []

    now_dt   = datetime.now(timezone.utc)
    start_dt = datetime.fromtimestamp(now_dt.timestamp() - lookback_seconds,
                                      tz=timezone.utc)
    now_epoch   = now_dt.timestamp()
    start_epoch = start_dt.timestamp()

    QUERY = """
        SELECT time_value
        FROM   Origin
        WHERE  time_value >= %s
          AND  time_value <= %s
        ORDER  BY time_value ASC
    """

    try:
        # pymysql uses connect_timeout; mysql.connector uses connection_timeout
        _timeout_kwarg = ("connect_timeout"
                          if _mysql_mod.__name__ in ("pymysql", "MySQLdb")
                          else "connection_timeout")
        conn = _mysql_mod.connect(
            host=mysql_host, port=mysql_port,
            user=mysql_user, password=mysql_password,
            database=mysql_db, **{_timeout_kwarg: 10},
        )
        cursor = conn.cursor()
        try:
            cursor.execute(QUERY, (start_epoch, now_epoch))
            rows = cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
    except Exception as exc:
        log.warning("Could not query SeisComP MySQL for origin gaps: %s", exc)
        return []

    log.info("Origin scan: %d origin(s) in the last %.0f h",
             len(rows), lookback_seconds / 3600)

    gaps: List[OriginGap] = []

    if not rows:
        # No origins at all -- the entire window is a gap
        gaps.append(OriginGap(start_dt, now_dt, lookback_seconds))
        return gaps

    # Helper: epoch float -> UTC datetime
    def _dt(epoch_f: float) -> datetime:
        return datetime.fromtimestamp(float(epoch_f), tz=timezone.utc)

    times = [float(r[0]) for r in rows]

    # Leading silence (window start -> first origin)
    if times[0] - start_epoch > min_gap_s:
        gaps.append(OriginGap(start_dt, _dt(times[0]),
                               times[0] - start_epoch))

    # Inter-origin silences
    for t_prev, t_next in zip(times, times[1:]):
        dur = t_next - t_prev
        if dur > min_gap_s:
            gaps.append(OriginGap(_dt(t_prev), _dt(t_next), dur))

    # Trailing silence (last origin -> now)
    if now_epoch - times[-1] > min_gap_s:
        gaps.append(OriginGap(_dt(times[-1]), now_dt,
                               now_epoch - times[-1]))

    return gaps


# ---------------------------------------------------------------------------
# 1. SCAN -- archive gap detection
# ---------------------------------------------------------------------------

def discover_channels(archive_dir: Path) -> List[Tuple[str, str, str, str]]:
    """Walk the SDS directory tree and return unique (NET, STA, LOC, CHA) tuples.

    SDS layout: YEAR/NET/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY
    """
    channels = set()
    for chan_dir in archive_dir.rglob("*.D"):
        parts = chan_dir.parts
        # Expect ?/YEAR/NET/STA/CHA.D
        if len(parts) < 4:
            continue
        net = parts[-3]
        sta = parts[-2]
        cha = parts[-1].replace(".D", "")
        # Derive LOC from a sample filename inside the directory
        for fname in chan_dir.iterdir():
            if fname.is_file():
                tokens = fname.name.split(".")
                # NET.STA.LOC.CHA.D.YEAR.DOY   -> 7 tokens
                if len(tokens) >= 7:
                    loc = tokens[2]
                    channels.add((net, sta, loc, cha))
                    break
    return sorted(channels)


def scan_gaps(
    archive_dir: Path,
    lookback_seconds: float,
    min_gap_s: float = 1.0,
) -> List[Gap]:
    """Return every gap in the SDS archive over the last ``lookback_seconds``.

    For each NSLC channel we request the full window from obspy's SDSClient.
    If the returned Stream covers less than the requested window we report the
    uncovered spans as gaps.  We also call ``Stream.get_gaps()`` for intra-file
    dropouts.
    """
    try:
        from obspy import UTCDateTime
        from obspy.clients.filesystem.sds import Client as SDSClient
    except ImportError:
        log.error("obspy is required -- install it with:  pip install obspy")
        sys.exit(1)

    now   = UTCDateTime()
    start = now - lookback_seconds

    sds = SDSClient(str(archive_dir))
    channels = discover_channels(archive_dir)
    log.info("Scanning %d channels over the last %.0f hours ?",
             len(channels), lookback_seconds / 3600)

    gaps: List[Gap] = []

    for net, sta, loc, cha in channels:
        try:
            st = sds.get_waveforms(net, sta, loc, cha, start, now)
        except Exception as exc:
            log.debug("SDSClient error for %s.%s.%s.%s: %s", net, sta, loc, cha, exc)
            st = None

        if not st or len(st) == 0:
            # Entire window missing
            gaps.append(Gap(net, sta, loc, cha, start, now,
                            lookback_seconds, "missing_file"))
            continue

        st.merge(method=1, fill_value=None)

        # Check coverage at the window edges
        earliest_data = min(tr.stats.starttime for tr in st)
        latest_data   = max(tr.stats.endtime   for tr in st)

        if earliest_data > start + min_gap_s:
            gaps.append(Gap(net, sta, loc, cha, start, earliest_data,
                            float(earliest_data - start), "missing_file"))

        if latest_data < now - min_gap_s:
            gaps.append(Gap(net, sta, loc, cha, latest_data, now,
                            float(now - latest_data), "missing_file"))

        # Internal gaps
        for raw in st.get_gaps():
            # raw = [net, sta, loc, cha, gap_start, gap_end, duration, n_samples]
            g_start, g_end, g_dur = raw[4], raw[5], abs(raw[6])
            if g_dur < min_gap_s:
                continue
            # Clamp to our window
            gs = max(start, g_start)
            ge = min(now, g_end)
            if ge > gs:
                gaps.append(Gap(net, sta, loc, cha, gs, ge,
                                float(ge - gs), "intra_file"))

    log.info("Gap scan complete -- %d gap(s) found", len(gaps))
    return gaps


# ---------------------------------------------------------------------------
# 2. FETCH -- pull missing data from SeedLink ring-buffer
# ---------------------------------------------------------------------------

def _write_stream_list(gaps: List[Gap], path: Path):
    """Write a scart-compatible stream-list file for the given gaps.

    Format: NET STA LOC CHA STARTTIME ENDTIME
    (times as  YYYY-MM-DDTHH:MM:SS)
    """
    with open(path, "w") as fh:
        for g in gaps:
            # scart wants timestamps without fractional seconds
            s = g.starttime.strftime("%Y-%m-%dT%H:%M:%S")
            e = g.endtime.strftime(  "%Y-%m-%dT%H:%M:%S")
            loc = g.location if g.location else "  "   # scart uses two spaces for empty loc
            fh.write(f"{g.network} {g.station} {loc} {g.channel} {s} {e}\n")


def fetch_from_seedlink(
    gaps: List[Gap],
    archive_dir: Path,
    seedlink_host: str,
    seedlink_port: int,
    scart_bin: Path,
    stats: RecoveryStats,
    dry_run: bool = False,
) -> List[Gap]:
    """For each gap, attempt to retrieve data from the SeedLink ring-buffer.

    Uses ``scart -I slink://HOST:PORT --list ?`` to pull data and write it
    into the SDS archive where it can then be read by obspy.

    Returns the subset of gaps that could NOT be recovered (either scart
    returned no data or SeedLink doesn't hold that far back).
    """
    if not gaps:
        return []

    if not scart_bin.exists():
        log.warning("scart not found at %s -- skipping SeedLink fetch", scart_bin)
        stats.errors.append(f"scart not found at {scart_bin}")
        return gaps

    slink_url = f"slink://{seedlink_host}:{seedlink_port}"
    unresolved: List[Gap] = []

    with tempfile.TemporaryDirectory(prefix="gap_recovery_") as tmpdir:
        stream_list = Path(tmpdir) / "streams.txt"
        _write_stream_list(gaps, stream_list)

        cmd = [
            str(scart_bin),
            "-I", slink_url,
            "--list", str(stream_list),
            "-v",
            str(archive_dir),
        ]
        log.info("Fetching %d gap(s) from SeedLink %s ?", len(gaps), slink_url)
        if dry_run:
            log.info("[DRY-RUN] would run: %s", " ".join(cmd))
            return gaps   # dry-run: nothing fetched

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            log.debug("scart stdout:\n%s", result.stdout[:2000])
            if result.returncode != 0:
                stderr_snip = result.stderr[:500].strip()
                if result.returncode == 255:
                    # Exit 255 = SeedLink could not be reached or the ring
                    # buffer holds no data this old.  Expected for long --hours
                    # windows; log as a warning, not a hard error.
                    log.warning(
                        "scart: SeedLink ring buffer has no data for this window "
                        "(exit 255) -- gaps will fall through to EarthScope/archive"
                    )
                else:
                    log.warning("scart exited %d:\n%s", result.returncode, stderr_snip)
                    stats.errors.append(f"scart returned {result.returncode}")
                # Either way, treat all gaps as unresolved so they continue
                unresolved.extend(gaps)
            else:
                # Verify each gap by re-reading the archive
                from obspy import UTCDateTime
                from obspy.clients.filesystem.sds import Client as SDSClient
                sds = SDSClient(str(archive_dir))
                for g in gaps:
                    try:
                        st = sds.get_waveforms(
                            g.network, g.station, g.location, g.channel,
                            g.starttime, g.endtime
                        )
                        if st and len(st) > 0:
                            stats.gaps_fetched += 1
                            log.info("  [ok] recovered  %s", g)
                        else:
                            unresolved.append(g)
                            log.info("  [x] still missing after fetch: %s", g)
                    except Exception:
                        unresolved.append(g)
        except subprocess.TimeoutExpired:
            log.error("scart timed out fetching gaps")
            stats.errors.append("scart timed out")
            unresolved.extend(gaps)
        except FileNotFoundError:
            log.error("scart binary not found: %s", scart_bin)
            unresolved.extend(gaps)

    stats.gaps_unresolved += len(unresolved)
    return unresolved


# ---------------------------------------------------------------------------
# 2b. EarthScope FDSN fallback
# ---------------------------------------------------------------------------

def _write_to_sds(
    stream,
    archive_dir: Path,
    scart_bin: Path,
) -> bool:
    """Write an obspy Stream into the SDS archive.

    Preferred path: write a temp miniSEED then call
    ``scart -I /tmp/file.mseed ARCHIVE_DIR`` (consistent with the SeedLink
    fetch above).

    Fallback (when scart is absent or fails): manually compute the SDS day-file
    path for every trace and append-write using obspy.
    """
    import tempfile
    if not stream:
        return False

    # ?? preferred: scart import ????????????????????????????????????????????
    if scart_bin.exists():
        with tempfile.NamedTemporaryFile(suffix=".mseed", delete=False,
                                         prefix="gap_es_") as fh:
            tmp_path = Path(fh.name)
        try:
            stream.write(str(tmp_path), format="MSEED")
            result = subprocess.run(
                [str(scart_bin), "-I", str(tmp_path), str(archive_dir)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                return True
            log.debug("scart import returned %d: %s",
                      result.returncode, result.stderr[:300])
        except Exception as exc:
            log.debug("scart import failed: %s", exc)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ?? fallback: manual SDS write ?????????????????????????????????????????
    log.debug("Falling back to manual SDS write for %d trace(s)", len(stream))
    ok = False
    for tr in stream:
        try:
            s    = tr.stats
            year = s.starttime.year
            doy  = s.starttime.julday
            loc  = s.location if s.location else ""
            day_dir = archive_dir / str(year) / s.network / s.station / f"{s.channel}.D"
            day_dir.mkdir(parents=True, exist_ok=True)
            fname   = f"{s.network}.{s.station}.{loc}.{s.channel}.D.{year}.{doy:03d}"
            fpath   = day_dir / fname
            # Append if file already exists so we don't clobber prior data
            mode = "ab" if fpath.exists() else "wb"
            with open(fpath, mode) as fh:
                import io
                buf = io.BytesIO()
                from obspy import Stream as ObspyStream
                ObspyStream([tr]).write(buf, format="MSEED")
                fh.write(buf.getvalue())
            ok = True
        except Exception as exc:
            log.debug("Manual SDS write failed for %s: %s", tr.id, exc)
    return ok


# Maximum number of gap rows to include in a single FDSN bulk request.
# Large batches can cause TCP connection resets at the IRIS server;
# keeping this small (~50) makes each request fast and retryable.
_FDSN_BATCH_SIZE = 50


def _fetch_earthscope_batch(
    gaps: List[Gap],
    client,                       # obspy FDSNClient (already constructed)
    archive_dir: Path,
    earthscope_url: str,
    scart_bin: Path,
    stats: RecoveryStats,
    dry_run: bool,
    FDSNNoDataException,
    FDSNRequestTooLargeException,
    FDSNTimeoutException,
    FDSNInternalServerException,
) -> List[Gap]:
    """Send ONE bulk FDSN request for *gaps* and write the result to SDS.

    Splits recursively on HTTP-413 or TCP connection-reset errors so that
    a single oversized batch never silently discards all gaps.
    """
    from urllib.error import URLError

    if not gaps:
        return []

    if dry_run:
        for g in gaps:
            log.info("  [DRY-RUN] would request %s", g)
        return gaps

    bulk = [
        (
            g.network,
            g.station,
            g.location or "*",
            g.channel,
            g.starttime,
            g.endtime,
        )
        for g in gaps
    ]

    try:
        st = client.get_waveforms_bulk(bulk, quality="B")
    except FDSNNoDataException:
        log.debug("EarthScope batch: no data for %d gap(s)", len(gaps))
        # Don't increment gaps_unresolved here -- caller handles accounting
        return gaps
    except (FDSNRequestTooLargeException, URLError) as exc:
        if len(gaps) == 1:
            # Cannot split further -- log and give up on this single gap
            log.warning("EarthScope: cannot fetch single gap %s -- %s", gaps[0], exc)
            stats.errors.append(f"EarthScope single-gap fetch failed: {gaps[0].nslc}: {exc}")
            return gaps
        # Split in half and retry each half independently
        reason = "HTTP 413" if isinstance(exc, FDSNRequestTooLargeException) else f"connection error ({exc})"
        log.warning(
            "EarthScope batch of %d gap(s) failed (%s) -- retrying in two halves",
            len(gaps), reason,
        )
        mid = len(gaps) // 2
        unresolved_a = _fetch_earthscope_batch(
            gaps[:mid], client, archive_dir, earthscope_url, scart_bin, stats, dry_run,
            FDSNNoDataException, FDSNRequestTooLargeException,
            FDSNTimeoutException, FDSNInternalServerException,
        )
        unresolved_b = _fetch_earthscope_batch(
            gaps[mid:], client, archive_dir, earthscope_url, scart_bin, stats, dry_run,
            FDSNNoDataException, FDSNRequestTooLargeException,
            FDSNTimeoutException, FDSNInternalServerException,
        )
        return unresolved_a + unresolved_b
    except FDSNTimeoutException as exc:
        log.warning("EarthScope batch timed out (%d gap(s)) -- %s", len(gaps), exc)
        stats.errors.append(f"EarthScope timeout ({len(gaps)} gaps): {exc}")
        return gaps
    except FDSNInternalServerException as exc:
        log.warning("EarthScope server error for %d gap(s) -- %s", len(gaps), exc)
        stats.errors.append(f"EarthScope server error ({len(gaps)} gaps): {exc}")
        return gaps
    except Exception as exc:
        log.warning("EarthScope unexpected error for %d gap(s) -- %s", len(gaps), exc)
        stats.errors.append(f"EarthScope unexpected error ({len(gaps)} gaps): {exc}")
        return gaps

    if not st:
        return gaps

    # Write traces to SDS archive
    write_ok = _write_to_sds(st, archive_dir, scart_bin)
    if not write_ok:
        log.error("_write_to_sds failed for EarthScope batch of %d gap(s)", len(gaps))
        stats.errors.append(f"EarthScope SDS write failed ({len(gaps)} gaps)")
        return gaps

    # Verify each gap individually against the archive
    from obspy.clients.filesystem.sds import Client as SDSClient
    sds = SDSClient(str(archive_dir))
    returned_nslc = {
        f"{tr.stats.network}.{tr.stats.station}."
        f"{tr.stats.location}.{tr.stats.channel}"
        for tr in st
    }
    unresolved: List[Gap] = []
    for g in gaps:
        if g.nslc not in returned_nslc:
            unresolved.append(g)
            continue
        try:
            check = sds.get_waveforms(
                g.network, g.station, g.location, g.channel,
                g.starttime, g.endtime,
            )
            if check and len(check) > 0:
                stats.gaps_earthscope += 1
                log.info("  [ok] EarthScope filled  %s", g)
                continue
            log.warning("  [x] EarthScope had data but SDS re-read empty for %s", g)
            stats.errors.append(f"SDS re-read empty after EarthScope write: {g.nslc}")
        except Exception as exc:
            log.warning("  [x] SDS re-read error for %s: %s", g.nslc, exc)
            stats.errors.append(f"SDS re-read error after EarthScope: {g.nslc}: {exc}")
        unresolved.append(g)
    return unresolved


def fetch_from_earthscope(
    gaps: List[Gap],
    archive_dir: Path,
    earthscope_url: str,
    scart_bin: Path,
    stats: RecoveryStats,
    dry_run: bool = False,
) -> List[Gap]:
    """Attempt to fill remaining gaps via the EarthScope FDSN web service.

    Gaps are sent in batches of at most _FDSN_BATCH_SIZE entries so that a
    single bulk request never grows large enough to trigger a TCP connection
    reset at the IRIS server.  Each batch is retried with automatic recursive
    halving on HTTP-413 or connection-reset errors.

    Successfully downloaded miniSEED is written into the SDS archive (via
    scart or manual day-file write) so the ML picker step finds the data
    exactly as it would for real-time archived waveforms.

    Returns the subset of gaps still unresolvable even after EarthScope.
    """
    if not gaps:
        return []

    try:
        from obspy.clients.fdsn import Client as FDSNClient
        from obspy.clients.fdsn.header import (
            FDSNNoDataException,
            FDSNRequestTooLargeException,
            FDSNTimeoutException,
            FDSNInternalServerException,
        )
    except ImportError:
        log.warning("obspy.clients.fdsn unavailable -- skipping EarthScope fallback")
        return gaps

    log.info("EarthScope fallback: requesting %d gap(s) from %s in batches of %d ?",
             len(gaps), earthscope_url, _FDSN_BATCH_SIZE)

    if dry_run:
        log.info("[DRY-RUN] would send bulk FDSN requests to %s for:", earthscope_url)
        for g in gaps:
            log.info("  %s", g)
        return gaps

    try:
        client = FDSNClient(earthscope_url)
    except Exception as exc:
        log.warning("Could not create FDSN client for %s: %s", earthscope_url, exc)
        stats.errors.append(f"EarthScope client error: {exc}")
        return gaps

    # Split into fixed-size batches before any request is made.
    # This prevents the connection reset that occurs when sending all 1000+
    # gaps in a single HTTP POST to service.iris.edu.
    batches = [gaps[i:i + _FDSN_BATCH_SIZE]
               for i in range(0, len(gaps), _FDSN_BATCH_SIZE)]
    log.info("EarthScope: split %d gap(s) into %d batch(es)", len(gaps), len(batches))

    all_unresolved: List[Gap] = []
    for batch_num, batch in enumerate(batches, 1):
        log.info("EarthScope: batch %d/%d  (%d gap(s)) ?",
                 batch_num, len(batches), len(batch))
        unresolved = _fetch_earthscope_batch(
            batch, client, archive_dir, earthscope_url, scart_bin, stats, dry_run,
            FDSNNoDataException, FDSNRequestTooLargeException,
            FDSNTimeoutException, FDSNInternalServerException,
        )
        all_unresolved.extend(unresolved)

    filled = len(gaps) - len(all_unresolved)
    log.info("EarthScope filled %d gap(s); %d still unresolved",
             filled, len(all_unresolved))
    stats.gaps_unresolved += len(all_unresolved)
    return all_unresolved


# ---------------------------------------------------------------------------
# 3. PICK -- read waveforms for gap windows, pick with sceasyquake's own picker
# ---------------------------------------------------------------------------

def _read_sceasyquake_cfg(
    cfg_path: Optional[str] = None,
) -> dict:
    """Parse the deployed sceasyquake.cfg and return a settings dict.

    Tries (in order):
      1. ``cfg_path`` if explicitly provided
      2. ``~/seiscomp/etc/sceasyquake.cfg``   (deployed SeisComP module config)
      3. ``<workspace>/sceasyquake/etc/sceasyquake.conf``  (dev fallback)

    Returns a dict with keys matching the SeisComP config key names:
      picker.backend, picker.model, picker.pretrained, picker.model_path,
      picker.threshold, picker.gpd_threshold, picker.device,
      picker.buffer_seconds, picker.step_seconds
    """
    import configparser

    candidates = []
    if cfg_path:
        candidates.append(Path(cfg_path).expanduser())
    candidates += [
        Path(os.path.expanduser("~/seiscomp/etc/sceasyquake.cfg")),
        _HERE.parent / "sceasyquake" / "etc" / "sceasyquake.conf",
    ]

    raw: dict = {}
    for p in candidates:
        if not p.exists():
            continue
        # SeisComP cfg format: "key = value" (NOT ini-sections)
        with open(p, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    raw[k.strip()] = v.strip()
        log.info("Read sceasyquake config from %s", p)
        break
    else:
        log.warning("sceasyquake config not found -- using built-in defaults")

    return {
        "backend":         raw.get("picker.backend",        "phasenet"),
        "pretrained":      raw.get("picker.pretrained",      "stead"),
        "model_path":      raw.get("picker.model_path",      "") or None,
        "threshold":       float(raw.get("picker.threshold",     "0.5")),
        "gpd_threshold":   float(raw.get("picker.gpd_threshold", "0.994")),
        "device":          raw.get("picker.device",          "cpu"),
        "buffer_seconds":  int(raw.get("picker.buffer_seconds",  "60")),
        "step_seconds":    int(raw.get("picker.step_seconds",     "5")),
        "min_distance":    float(raw.get("picker.min_distance",   "0.2")),
        "norm":            raw.get("picker.norm",            "") or None,
        "phases":          raw.get("picker.phases",         "") or None,
        "label_order":     raw.get("picker.label_order",    "") or None,
    }


def _build_predictor(cfg: dict):
    """Instantiate a predictor using sceasyquake-stream's own factory function.

    Imports ``_make_predictor`` directly from ``sceasyquake-stream.py`` so the
    exact same backend/weights are used as in the running live process.
    """
    # Locate sceasyquake-stream.py relative to this file
    stream_script = _HERE.parent / "sceasyquake" / "bin" / "sceasyquake-stream.py"
    if stream_script.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("_sceasyquake_stream",
                                                      str(stream_script))
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            predictor = mod._make_predictor(
                backend       = cfg["backend"],
                pretrained    = cfg["pretrained"],
                model_path    = cfg["model_path"],
                threshold     = cfg["threshold"],
                gpd_threshold = cfg["gpd_threshold"],
                device        = cfg["device"],
                min_distance  = cfg.get("min_distance", 0.2),
                norm          = cfg.get("norm"),
                phases        = cfg.get("phases"),
                label_order   = cfg.get("label_order"),
            )
            try:
                predictor.load_model()
            except Exception:
                pass
            log.info("Predictor loaded via sceasyquake-stream factory: backend=%s device=%s",
                     cfg["backend"], cfg["device"])
            return predictor
        except Exception as exc:
            log.warning("Could not load predictor via sceasyquake-stream (%s) -- "
                        "falling back to direct import", exc)

    # Direct fallback
    try:
        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        p = PhaseNetPredictor(backend=cfg["backend"], threshold=cfg["threshold"],
                              device=cfg["device"])
        try:
            p.load_model()
        except Exception:
            pass
        return p
    except Exception as exc:
        log.error("Could not load predictor: %s", exc)
        return None


# ---------------------------------------------------------------------------
# GapPlaybackApp -- seiscomp.client.Application that picks archive windows
# ---------------------------------------------------------------------------

try:
    import seiscomp.client as _sc_client  # type: ignore
    _HAS_SC_CLIENT = True
except Exception:
    _HAS_SC_CLIENT = False


if _HAS_SC_CLIENT:
    class GapPlaybackApp(_sc_client.Application):
        """Minimal SeisComP Application that picks archived gap windows.

        Runs as a proper SC Application to obtain a real messaging connection
        so picks reach scautoloc on the PICK group exactly as the live
        sceasyquake process would send them.
        """

        def __init__(self, gaps: List[Gap], archive_dir: Path,
                     stats: RecoveryStats, window_s: float, overlap_s: float,
                     sceasyquake_cfg_path: Optional[str], pick_spool: Path,
                     sc_host: str = "localhost",
                     device_override: Optional[str] = None,
                     pick_rate: float = 20.0):
            # SC Application.__init__ takes argc/argv; pass a clean minimal argv
            # so it does not choke on gap_recovery's own argparse flags.
            # Use a PID-qualified name so concurrent or back-to-back runs don't
            # get a "Client name not unique" reject from scmaster.
            _name = f"gap_recovery_{os.getpid()}"
            _argv = [_name, "-H", sc_host, "--console=1"]
            _sc_client.Application.__init__(self, len(_argv), _argv)
            self.setMessagingEnabled(True)
            self.setDatabaseEnabled(False, False)
            self.setPrimaryMessagingGroup("PICK")
            # Store playback context
            self._gaps            = gaps
            self._archive_dir     = archive_dir
            self._stats           = stats
            self._window_s        = window_s
            self._overlap_s       = overlap_s
            self._cfg_path        = sceasyquake_cfg_path
            self._pick_spool      = pick_spool
            self._sc_host         = sc_host
            self._device_override = device_override
            self._pick_rate        = pick_rate

        def run(self) -> bool:
            from sceasyquake.uploader import PickUploader

            cfg = _read_sceasyquake_cfg(self._cfg_path)
            if self._device_override is not None:
                log.info("Overriding picker device: %s -> %s",
                         cfg.get("device", "cpu"), self._device_override)
                cfg["device"] = self._device_override
            predictor = _build_predictor(cfg)
            # Use the system agencyID (read from global.cfg via SeisComP bindings)
            # so picks look the same as live sceasyquake picks to scautoloc.
            try:
                _agency_id = self.agencyID()
            except Exception:
                _agency_id = None
            uploader  = PickUploader(
                connection = self.connection(),
                out_dir    = str(self._pick_spool),
                source     = "gap_recovery",
                agency_id  = _agency_id,
                pick_rate  = self._pick_rate,
            )

            if predictor is None:
                log.error("Predictor unavailable -- aborting pick step")
                return False

            _run_picking(self._gaps, self._archive_dir, predictor, uploader,
                         self._window_s, self._overlap_s, self._stats)
            return True


if _HAS_SC_CLIENT:
    class ReplaySpoolApp(_sc_client.Application):
        """SC Application that replays YAML pick spool files into the messaging bus.

        Used when a previous run fell back to YAML (e.g. because of a duplicate
        client-name error) and the picks need to be injected retrospectively.
        """

        def __init__(self, spool_dir: Path, sc_host: str = "localhost",
                     delete_after: bool = True):
            _name = f"gap_replay_{os.getpid()}"
            _argv = [_name, "-H", sc_host, "--console=1"]
            _sc_client.Application.__init__(self, len(_argv), _argv)
            self.setMessagingEnabled(True)
            self.setDatabaseEnabled(False, False)
            self.setPrimaryMessagingGroup("PICK")
            self._spool_dir    = spool_dir
            self._delete_after = delete_after
            self._published    = 0
            self._failed       = 0

        def run(self) -> bool:
            import yaml as _yaml
            from obspy import UTCDateTime
            from sceasyquake.uploader import PickUploader

            uploader = PickUploader(
                connection = self.connection(),
                out_dir    = str(self._spool_dir),
                source     = "gap_recovery",
            )

            pick_files = sorted(self._spool_dir.glob("pick_*.yml"))
            log.info("Replaying %d YAML pick file(s) from %s ?",
                     len(pick_files), self._spool_dir)

            for pf in pick_files:
                try:
                    with open(pf) as fh:
                        data = _yaml.safe_load(fh)
                    result = uploader.send_pick(
                        network     = data["network"],
                        station     = data["station"],
                        location    = data.get("location", ""),
                        channel     = data["channel"],
                        time        = UTCDateTime(data["time"]),
                        phase       = data.get("phase", "P"),
                        probability = data.get("probability"),
                        method      = data.get("method", "PhaseNet"),
                        author      = data.get("author", "gap_recovery"),
                    )
                    if result is True:
                        self._published += 1
                        if self._delete_after:
                            pf.unlink()
                    else:
                        # send_pick returned a path string (YAML fallback) -- still failed
                        self._failed += 1
                        log.warning("Pick replay fell back to YAML again for %s", pf.name)
                except Exception as exc:
                    self._failed += 1
                    log.warning("Failed to replay pick %s: %s", pf.name, exc)

            log.info("Replay complete -- published %d, failed %d",
                     self._published, self._failed)
            return True


def replay_yaml_spool(spool_dir: Path, sc_host: str, delete_after: bool = True):
    """Publish all YAML pick files in ``spool_dir`` to SC messaging.

    Picks were written there as a fallback when the messaging connection was
    unavailable during a previous run.  Successful publishes are deleted from
    the spool.
    """
    pick_files = list(spool_dir.glob("pick_*.yml"))
    if not pick_files:
        log.info("No YAML pick files found in %s -- nothing to replay", spool_dir)
        return

    log.info("Found %d YAML pick file(s) to replay from %s", len(pick_files), spool_dir)

    if not _HAS_SC_CLIENT:
        log.error("SeisComP Python bindings unavailable -- cannot replay picks to messaging")
        return

    try:
        app = ReplaySpoolApp(spool_dir=spool_dir, sc_host=sc_host,
                             delete_after=delete_after)
        app()
    except Exception as exc:
        log.error("ReplaySpoolApp failed: %s", exc)


def run_playback_picking(
    gaps: List[Gap],
    archive_dir: Path,
    window_s: float,
    overlap_s: float,
    stats: RecoveryStats,
    sc_host: str,
    sceasyquake_cfg_path: Optional[str],
    pick_spool: Path,
    dry_run: bool = False,
    device_override: Optional[str] = None,
    pick_rate: float = 20.0,
):
    """Orchestrate the pick step.

    When SeisComP Python bindings are available the waveforms are processed
    inside a ``seiscomp.client.Application`` so picks are delivered to the
    live PICK messaging group (-> scautoloc -> scevent -> scolv).

    Falls back to a YAML spool file when the bindings or messaging bus are
    unavailable.
    """
    if not gaps:
        return
    if dry_run:
        log.info("[DRY-RUN] would pick %d waveform gap window(s) -- skipping", len(gaps))
        return

    if _HAS_SC_CLIENT:
        log.info("Connecting to SeisComP messaging at %s ?", sc_host)
        # GapPlaybackApp passes -H directly in its own argv to the SC Application
        try:
            app = GapPlaybackApp(
                gaps                  = gaps,
                archive_dir           = archive_dir,
                stats                 = stats,
                window_s              = window_s,
                overlap_s             = overlap_s,
                sceasyquake_cfg_path  = sceasyquake_cfg_path,
                pick_spool            = pick_spool,
                sc_host               = sc_host,
                device_override       = device_override,
                pick_rate             = pick_rate,
            )
            app()
        except Exception as exc:
            log.error("GapPlaybackApp failed: %s", exc)
            stats.errors.append(f"SC playback app: {exc}")
    else:
        # No SC bindings -- fall back to YAML spool
        log.warning("SeisComP Python bindings unavailable -- picks will be "
                    "spooled as YAML to %s", pick_spool)
        cfg       = _read_sceasyquake_cfg(sceasyquake_cfg_path)
        if device_override is not None:
            log.info("Overriding picker device: %s -> %s",
                     cfg.get("device", "cpu"), device_override)
            cfg["device"] = device_override
        predictor = _build_predictor(cfg)
        if predictor is None:
            log.error("Predictor unavailable -- aborting pick step")
            return
        from sceasyquake.uploader import PickUploader
        pick_spool.mkdir(parents=True, exist_ok=True)
        uploader = PickUploader(connection=None, out_dir=str(pick_spool),
                                source="gap_recovery", pick_rate=pick_rate)
        _run_picking(gaps, archive_dir, predictor, uploader,
                     window_s, overlap_s, stats)


def _run_picking(
    gaps: List[Gap],
    archive_dir: Path,
    predictor,
    uploader,
    window_s: float,
    overlap_s: float,
    stats: RecoveryStats,
):
    """Inner worker: read archive waveforms, run predictor, upload picks."""
    from obspy.clients.filesystem.sds import Client as SDSClient
    from obspy import Stream, UTCDateTime

    sds = SDSClient(str(archive_dir))
    gaps_sorted   = sorted(gaps, key=lambda g: g.starttime.timestamp)
    merged_windows = _merge_gap_windows(gaps_sorted)

    for win_start, win_end, involved_gaps in merged_windows:
        log.info("Picking %s -> %s  (%.0f s, %d channel gap(s))",
                 win_start.strftime("%Y-%m-%dT%H:%M:%S"),
                 win_end.strftime("%Y-%m-%dT%H:%M:%S"),
                 float(win_end - win_start), len(involved_gaps))

        nslc_set = {(g.network, g.station, g.location, g.channel)
                    for g in involved_gaps}

        t = win_start
        while t < win_end:
            chunk_end = min(t + window_s, win_end + overlap_s)
            chunk_st  = Stream()
            for net, sta, loc, cha in nslc_set:
                try:
                    sub = sds.get_waveforms(net, sta, loc, cha,
                                            t - overlap_s, chunk_end + overlap_s)
                    if sub:
                        chunk_st += sub
                except Exception as exc:
                    log.debug("SDS read error %s.%s.%s.%s: %s", net, sta, loc, cha, exc)

            if not chunk_st:
                t += window_s - overlap_s
                continue

            chunk_st.merge(method=1, fill_value=0)
            chunk_st.detrend("demean")
            chunk_st.taper(max_percentage=0.05, max_length=2.0)

            try:
                if hasattr(predictor, "predict_multi"):
                    pairs = [
                        (f"{tr.stats.network}.{tr.stats.station}."
                         f"{tr.stats.location}.{tr.stats.channel}", Stream([tr]))
                        for tr in chunk_st
                    ]
                    picks = predictor.predict_multi(pairs)
                else:
                    picks = predictor.predict(chunk_st)
            except Exception as exc:
                log.warning("Prediction failed for chunk %s: %s", t, exc)
                t += window_s - overlap_s
                continue

            for pick in picks:
                try:
                    pick_time = pick.time if hasattr(pick, "time") else pick.get("time")
                    if pick_time is None:
                        continue
                    pt = UTCDateTime(pick_time)
                    if pt < t or pt > chunk_end:
                        continue  # outside chunk window
                    uploader.send_pick(
                        network     = pick.network  if hasattr(pick, "network")  else pick.get("network", ""),
                        station     = pick.station  if hasattr(pick, "station")  else pick.get("station", ""),
                        location    = pick.location if hasattr(pick, "location") else pick.get("location", ""),
                        channel     = pick.channel  if hasattr(pick, "channel")  else pick.get("channel", ""),
                        time        = pt,
                        phase       = pick.phase    if hasattr(pick, "phase")    else pick.get("phase", "P"),
                        probability = (pick.prob    if hasattr(pick, "prob")
                                       else pick.get("probability", pick.get("prob", None))),
                    )
                    stats.picks_uploaded += 1
                except Exception as exc:
                    log.debug("Pick upload error: %s  (%s)", exc, pick)

            t += window_s - overlap_s

    log.info("Picking complete -- %d pick(s) sent", stats.picks_uploaded)


def _merge_gap_windows(
    gaps: List[Gap],
    merge_tolerance_s: float = 60.0,
) -> List[Tuple["UTCDateTime", "UTCDateTime", List[Gap]]]:
    """Merge gaps whose time spans overlap (within tolerance) into unified windows.

    Returns list of (window_start, window_end, [contributing Gap objects]).
    Merging means all channels near-simultaneous time windows are picked in a
    single batch, giving the ML associator maximum phase context.
    """
    if not gaps:
        return []

    windows = []
    cur_start = gaps[0].starttime
    cur_end   = gaps[0].endtime
    cur_group = [gaps[0]]

    for g in gaps[1:]:
        if g.starttime <= cur_end + merge_tolerance_s:
            cur_end   = max(cur_end, g.endtime)
            cur_group.append(g)
        else:
            windows.append((cur_start, cur_end, cur_group))
            cur_start = g.starttime
            cur_end   = g.endtime
            cur_group = [g]

    windows.append((cur_start, cur_end, cur_group))
    return windows


# ---------------------------------------------------------------------------
# scautoloc buffer.pickKeep helpers
# ---------------------------------------------------------------------------

def _read_scautoloc_pick_keep(cfg_path: Path) -> Optional[int]:
    """Return the current buffer.pickKeep value from scautoloc.cfg (seconds)."""
    if not cfg_path.exists():
        return None
    with open(cfg_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "buffer.pickKeep":
                try:
                    return int(v.strip())
                except ValueError:
                    return None
    return None


def _patch_scautoloc_pick_keep(cfg_path: Path, new_value: int) -> Optional[int]:
    """Set buffer.pickKeep in scautoloc.cfg; return the original value (or None).

    If the key does not already exist it is appended.
    """
    lines = []
    original: Optional[int] = None
    if cfg_path.exists():
        with open(cfg_path) as fh:
            lines = fh.readlines()

    patched = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#") and "=" in stripped:
            k, _, v = stripped.partition("=")
            if k.strip() == "buffer.pickKeep":
                try:
                    original = int(v.strip())
                except ValueError:
                    pass
                new_lines.append(f"buffer.pickKeep = {new_value}\n")
                patched = True
                continue
        new_lines.append(line)

    if not patched:
        new_lines.append(f"buffer.pickKeep = {new_value}\n")

    with open(cfg_path, "w") as fh:
        fh.writelines(new_lines)

    log.info("Patched %s: buffer.pickKeep %s -> %d",
             cfg_path, original, new_value)
    return original


def _restart_scautoloc():
    """Restart scautoloc to pick up the new buffer.pickKeep value."""
    try:
        result = subprocess.run(
            ["seiscomp", "restart", "scautoloc"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            log.info("scautoloc restarted successfully")
        else:
            log.warning("seiscomp restart scautoloc returned %d:\n%s",
                        result.returncode, result.stderr[:300])
    except Exception as exc:
        log.warning("Could not restart scautoloc: %s -- "
                    "restart it manually to apply buffer.pickKeep change", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect archive gaps, re-fetch from SeedLink, and run ML picker playback.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Scan
    p.add_argument("--archive", default=os.path.expanduser("~/seiscomp/var/lib/archive"),
                   help="Path to the SDS archive root (default: ~/seiscomp/var/lib/archive)")
    p.add_argument("--hours", type=float, default=24.0,
                   help="How many hours back to scan for gaps (default: 24)")
    p.add_argument("--min-gap", type=float, default=1.0,
                   help="Minimum gap duration in seconds to report (default: 1.0)")

    # Origin gap scan (SeisComP MySQL)
    p.add_argument("--sc-db-host", default="localhost",
                   help="SeisComP MySQL host (default: localhost)")
    p.add_argument("--sc-db-port", type=int, default=3306,
                   help="SeisComP MySQL port (default: 3306)")
    p.add_argument("--sc-db-user", default="sysop",
                   help="SeisComP MySQL user (default: sysop)")
    p.add_argument("--sc-db-pass", default="sysop",
                   help="SeisComP MySQL password (default: sysop)")
    p.add_argument("--sc-db-name", default="seiscomp",
                   help="SeisComP MySQL database name (default: seiscomp)")
    p.add_argument("--min-origin-gap", type=float, default=1800.0,
                   help="Minimum silence between origins to flag (seconds, default: 1800 = 30 min)")
    p.add_argument("--no-origin-scan", action="store_true",
                   help="Skip the SeisComP origin gap scan")

    # SeedLink fetch
    p.add_argument("--seedlink-host", default="localhost",
                   help="SeedLink server host (default: localhost)")
    p.add_argument("--seedlink-port", type=int, default=18000,
                   help="SeedLink server port (default: 18000)")
    p.add_argument("--scart", default=os.path.expanduser("~/seiscomp/bin/scart"),
                   help="Path to the scart binary (default: ~/seiscomp/bin/scart)")

    # EarthScope FDSN fallback
    p.add_argument("--earthscope-url", default="IRIS",
                   help="FDSN datacenter ID or URL used as SeedLink fallback "
                        "(default: IRIS  -- the EarthScope/IRIS DMC endpoint). "
                        "Any obspy-recognised FDSN short-code or full URL is accepted, e.g. "
                        "'IRIS', 'GEOFON', 'https://service.iris.edu'.")
    p.add_argument("--no-earthscope", action="store_true",
                   help="Skip the EarthScope FDSN fallback even when SeedLink cannot fill a gap")

    # Playback picking
    p.add_argument("--window", type=float, default=300.0,
                   help="Playback chunk size in seconds (default: 300)")
    p.add_argument("--overlap", type=float, default=30.0,
                   help="Chunk overlap in seconds to avoid edge effects (default: 30)")
    p.add_argument("--sceasyquake-cfg",
                   default=os.path.expanduser("~/seiscomp/etc/sceasyquake.cfg"),
                   help="Path to sceasyquake config (default: ~/seiscomp/etc/sceasyquake.cfg). "
                        "Picker backend, weights, device, and threshold are read from this "
                        "file so the same model configuration is used as the live process.")
    p.add_argument("--device", default=None,
                   help="Override the picker inference device (e.g. 'cpu' or 'cuda'). "
                        "Defaults to picker.device from sceasyquake.cfg (typically 'cuda'). "
                        "Pass '--device cpu' to leave the GPU free for the live sceasyquake "
                        "process while gap recovery runs.")

    # SeisComP messaging
    p.add_argument("--sc-host", default="localhost",
                   help="SeisComP scmaster host for publishing picks (default: localhost)")
    p.add_argument("--pick-spool", default=os.path.expanduser("~/sceasyquake/picks"),
                   help="YAML pick spool directory used when SC messaging is unavailable "
                        "(default: ~/sceasyquake/picks)")

    # Behaviour
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and report only -- do not fetch data or upload picks")
    p.add_argument("--scan-only", action="store_true",
                   help="Scan and print gaps then exit (skips fetch and picking)")
    p.add_argument("--no-scan", action="store_true",
                   help="Skip the archive waveform gap scan (step 1). Picking is driven "
                        "entirely by the origin-gap scan (step 0). Useful when the archive "
                        "is intact and you only want to re-pick silent origin periods.")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip SeedLink fetch step (use existing archive data only)")
    p.add_argument("--no-pick", action="store_true",
                   help="Skip ML picking step")
    p.add_argument("--replay-spool", action="store_true",
                   help="Publish any YAML pick files from the spool directory to SC "
                        "messaging and exit (use after a run that fell back to YAML)")
    p.add_argument("--pick-rate", type=float, default=20.0,
                   help="Maximum picks/second to publish to the SeisComP messaging bus "
                        "(default: 20).  Throttling prevents flooding scmaster which can "
                        "destabilize other connected clients such as the live sceasyquake "
                        "process.  Set to 0 to disable throttling.")
    p.add_argument("--sc-pick-keep-s", type=int, default=None,
                   help="Temporarily set scautoloc's buffer.pickKeep (seconds) for this run "
                        "so old picks are not discarded.  The original value is restored on "
                        "exit.  scautoloc is restarted to apply the change.  Example: "
                        "--sc-pick-keep-s 172800 for a 48-hour window.  Useful when --hours "
                        "exceeds the default 21600s (6h) buffer.pickKeep.")
    p.add_argument("--scautoloc-cfg",
                   default=os.path.expanduser("~/seiscomp/etc/scautoloc.cfg"),
                   help="Path to scautoloc.cfg -- used only when --sc-pick-keep-s is given "
                        "(default: ~/seiscomp/etc/scautoloc.cfg)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable DEBUG logging")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ?? REPLAY SPOOL (early exit) ?????????????????????????????????????????
    if args.replay_spool:
        pick_spool = Path(args.pick_spool).expanduser()
        log.info("=== REPLAY SPOOL: publishing YAML picks from %s ===", pick_spool)
        replay_yaml_spool(
            spool_dir    = pick_spool,
            sc_host      = args.sc_host,
            delete_after = True,
        )
        return

    archive_dir = Path(args.archive).expanduser()
    if not archive_dir.is_dir():
        log.error("Archive directory not found: %s", archive_dir)
        sys.exit(1)

    stats = RecoveryStats()

    # ?? buffer.pickKeep check / temporary override ????????????????????????
    # scautoloc silently discards picks whose timestamp is older than
    # buffer.pickKeep seconds relative to the current system clock.
    # For --hours windows larger than buffer.pickKeep the picks will never
    # reach the associator.  Warn loudly and, if --sc-pick-keep-s is given,
    # patch scautoloc.cfg and restart scautoloc automatically.
    hours_s = int(args.hours * 3600)
    scautoloc_cfg = Path(args.scautoloc_cfg).expanduser()
    current_pick_keep = _read_scautoloc_pick_keep(scautoloc_cfg)
    _original_pick_keep: Optional[int] = None  # saved for restore on exit

    if not args.no_pick and not args.scan_only and not args.dry_run:
        if args.sc_pick_keep_s is not None:
            # User explicitly requested a temporary override
            desired = max(args.sc_pick_keep_s, hours_s)
            _original_pick_keep = _patch_scautoloc_pick_keep(scautoloc_cfg, desired)
            log.info("Restarting scautoloc with buffer.pickKeep = %d s ?", desired)
            _restart_scautoloc()
            time.sleep(5)  # give scautoloc time to come up
        elif current_pick_keep is not None and hours_s > current_pick_keep:
            log.warning(
                "[!]  scautoloc buffer.pickKeep = %d s (%.1f h) but you are "
                "re-picking %.1f h of data.  Picks older than %.1f h will be "
                "SILENTLY DISCARDED by scautoloc and will never reach the "
                "associator.  To fix this, either:\n"
                "  1. Re-run with  --sc-pick-keep-s %d  (auto-patches and "
                "restarts scautoloc)\n"
                "  2. Manually set  buffer.pickKeep = %d  in %s then "
                "run  seiscomp restart scautoloc",
                current_pick_keep, current_pick_keep / 3600,
                args.hours, current_pick_keep / 3600,
                hours_s, hours_s, scautoloc_cfg,
            )

    import atexit

    def _restore_pick_keep():
        if _original_pick_keep is not None:
            log.info("Restoring scautoloc buffer.pickKeep -> %d s ?", _original_pick_keep)
            _patch_scautoloc_pick_keep(scautoloc_cfg, _original_pick_keep)
            _restart_scautoloc()

    atexit.register(_restore_pick_keep)

    # ?? 0. ORIGIN GAP SCAN ????????????????????????????????????????????????
    if not args.no_origin_scan:
        log.info("=== STEP 0: Scanning SeisComP origins (last %.0f h, min silence %.0f s) ===",
                 args.hours, args.min_origin_gap)
        origin_gaps = scan_origin_gaps(
            lookback_seconds=args.hours * 3600,
            min_gap_s=args.min_origin_gap,
            mysql_host=args.sc_db_host,
            mysql_port=args.sc_db_port,
            mysql_user=args.sc_db_user,
            mysql_password=args.sc_db_pass,
            mysql_db=args.sc_db_name,
        )
        stats.origin_gaps_found = len(origin_gaps)
        if origin_gaps:
            log.info("Origin silence periods detected (> %.0f s with no new origins):",
                     args.min_origin_gap)
            for og in origin_gaps:
                log.info("  [!]  %s", og)
        else:
            log.info("No significant origin silences found.")
    else:
        log.info("[--no-origin-scan] Skipping origin gap scan")
        origin_gaps = []

    # ?? 1. SCAN ???????????????????????????????????????????????????????????
    if args.no_scan:
        log.info("\n[--no-scan] Skipping archive waveform gap scan")
        gaps = []
        stats.gaps_found = 0
    else:
        log.info("\n=== STEP 1: Scanning archive %s (last %.0f h) ===", archive_dir, args.hours)
        gaps = scan_gaps(archive_dir, args.hours * 3600, min_gap_s=args.min_gap)
        stats.gaps_found = len(gaps)

    if not gaps:
        log.info("No waveform gaps found -- archive looks complete for the last %.0f hours.",
                 args.hours)
        if not origin_gaps:
            return
        # The archive is intact but no picks/origins were created during the
        # origin-gap periods (e.g. sceasyquake / scevent was down).  Synthesise
        # picking windows from those silent intervals so the ML picker
        # re-processes every channel and scautoloc can form the missing events.
        if not args.no_pick and not args.scan_only and not args.dry_run:
            log.info(
                "\nWaveform archive is complete but %d origin-gap period(s) detected -- "
                "re-picking all channels over those intervals.",
                len(origin_gaps),
            )
            gaps_from_origins = _origin_gaps_to_gap_list(origin_gaps, archive_dir)
            log.info(
                "Synthesised %d channel?window picking task(s) from %d origin gap(s)",
                len(gaps_from_origins), len(origin_gaps),
            )
            pick_spool = Path(args.pick_spool).expanduser()
            pick_spool.mkdir(parents=True, exist_ok=True)
            cfg_path = args.sceasyquake_cfg if Path(args.sceasyquake_cfg).exists() else None
            run_playback_picking(
                gaps                 = gaps_from_origins,
                archive_dir          = archive_dir,
                window_s             = args.window,
                overlap_s            = args.overlap,
                stats                = stats,
                sc_host              = args.sc_host,
                sceasyquake_cfg_path = cfg_path,
                pick_spool           = pick_spool,
                device_override      = args.device,
                pick_rate            = args.pick_rate,
            )
        elif args.scan_only or args.dry_run:
            log.info("[scan-only / dry-run] Would re-pick %d origin-gap window(s) -- skipping.",
                     len(origin_gaps))
        _print_summary(stats)
        return

    log.info("\nWaveform gaps detected:")
    for g in gaps:
        log.info("  %s", g)

    if args.scan_only or args.dry_run:
        if args.dry_run:
            log.info("\n[DRY-RUN] Fetch and picking steps would follow. Exiting.")
        _print_summary(stats)
        return

    # ?? 2. FETCH ??????????????????????????????????????????????????????????
    unresolved = gaps
    if not args.no_fetch:
        log.info("\n=== STEP 2: Fetching gaps from SeedLink %s:%d ===",
                 args.seedlink_host, args.seedlink_port)
        unresolved = fetch_from_seedlink(
            gaps,
            archive_dir,
            seedlink_host=args.seedlink_host,
            seedlink_port=args.seedlink_port,
            scart_bin=Path(args.scart).expanduser(),
            stats=stats,
            dry_run=args.dry_run,
        )
        resolved = len(gaps) - len(unresolved)
        log.info("Fetched data for %d/%d gap(s); %d still missing from SeedLink",
                 resolved, len(gaps), len(unresolved))
    else:
        log.info("\n[--no-fetch] Skipping SeedLink fetch step")

    # ?? 2b. EARTHSCOPE FALLBACK ???????????????????????????????????????????
    if unresolved and not args.no_earthscope and not args.no_fetch:
        log.info("\n=== STEP 2b: EarthScope FDSN fallback for %d unresolved gap(s) ===",
                 len(unresolved))
        # Reset the unresolved counter; fetch_from_earthscope will re-add what
        # it cannot fill, so we don't double-count with what fetch_from_seedlink set.
        stats.gaps_unresolved -= len(unresolved)
        unresolved = fetch_from_earthscope(
            unresolved,
            archive_dir,
            earthscope_url=args.earthscope_url,
            scart_bin=Path(args.scart).expanduser(),
            stats=stats,
            dry_run=args.dry_run,
        )
        log.info("EarthScope filled %d gap(s); %d still unresolved",
                 stats.gaps_earthscope, len(unresolved))
    elif unresolved and (args.no_earthscope or args.no_fetch):
        log.info("\n[--no-earthscope / --no-fetch] Skipping EarthScope fallback")

    # ?? 3. PICK ???????????????????????????????????????????????????????????
    if not args.no_pick:
        log.info("\n=== STEP 3: Playback picking via sceasyquake config ===")
        if args.pick_rate > 0:
            log.info("Pick publish rate limited to %.0f picks/s to avoid flooding scmaster",
                     args.pick_rate)
        pick_spool = Path(args.pick_spool).expanduser()
        pick_spool.mkdir(parents=True, exist_ok=True)
        cfg_path = args.sceasyquake_cfg if Path(args.sceasyquake_cfg).exists() else None
        # Pick ALL gap windows (including still-unresolved ones that may have
        # partial data) so scautoloc gets as many phases as possible.
        run_playback_picking(
            gaps                 = gaps,
            archive_dir          = archive_dir,
            window_s             = args.window,
            overlap_s            = args.overlap,
            stats                = stats,
            sc_host              = args.sc_host,
            sceasyquake_cfg_path = cfg_path,
            pick_spool           = pick_spool,
            device_override      = args.device,
            pick_rate            = args.pick_rate,
        )
    else:
        log.info("\n[--no-pick] Skipping ML picking step")

    # ?? 4. REPORT ?????????????????????????????????????????????????????????
    _print_summary(stats)


def _origin_gaps_to_gap_list(
    origin_gaps: List[OriginGap],
    archive_dir: Path,
) -> List[Gap]:
    """Convert origin-pipeline silence periods into synthetic Gap records.

    For each origin gap period we create one Gap entry per channel found in the
    archive so that ``run_playback_picking`` will re-run the ML picker over every
    channel during the silent interval.  The waveforms themselves are intact in
    the archive -- we just need to (re)pick them.
    """
    from obspy import UTCDateTime

    channels = discover_channels(archive_dir)
    synthetic: List[Gap] = []
    for og in origin_gaps:
        t0 = UTCDateTime(og.start.timestamp())
        t1 = UTCDateTime(og.end.timestamp())
        dur = float(t1 - t0)
        for net, sta, loc, cha in channels:
            synthetic.append(Gap(
                network   = net,
                station   = sta,
                location  = loc,
                channel   = cha,
                starttime = t0,
                endtime   = t1,
                duration_s = dur,
                source    = "origin_gap",
            ))
    return synthetic


def _print_summary(stats: RecoveryStats):
    log.info("")
    log.info("======================== SUMMARY ========================")
    log.info("  Origin silence gaps : %d", stats.origin_gaps_found)
    log.info("  Waveform gaps found : %d", stats.gaps_found)
    log.info("  Gaps via SeedLink   : %d", stats.gaps_fetched)
    log.info("  Gaps via EarthScope : %d", stats.gaps_earthscope)
    log.info("  Gaps still missing  : %d", stats.gaps_unresolved)
    log.info("  Picks uploaded      : %d", stats.picks_uploaded)
    if stats.errors:
        log.info("  Errors:")
        for e in stats.errors:
            log.info("    ? %s", e)
    log.info("=========================================================")


if __name__ == "__main__":
    main()
