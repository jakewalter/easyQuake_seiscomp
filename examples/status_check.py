#!/usr/bin/env python3
"""Quick audit of a running SeisComP installation.

This helper inspects the following aspects of a live SeisComP root and
reports issues that often prevent a real-time processing pipeline from
working correctly:

* which modules are enabled/running (``seiscomp status``).
* **scdb, scamp, scautoloc, sceasyquake, scevent, scmag** are all running
  – the complete pick→origin→event→magnitude chain.
* slarchive state directory: are there recent files for stations?  this is a
  proxy for "stations are actually streaming".
* key files under etc/key look sane (contain seedlink bindings).
* ``global.cfg`` has ``recordstream`` set (required by scamp for waveform
  access) and ``core.plugins = dbmysql``.
* "minimum configuration" values compared to a reference backup located at
  ~/proc_backup (global.cfg mainly).
* check that a pick/associate workflow is possible by ensuring the
  scautopick module is enabled and some basics of its config exist.
* **scautoloc.cfg** has the correct keys:

  - ``locator.profile`` set (detects deprecated ``autoloc.locator.profile``).
  - ``autoloc.minPickSNR = 0`` so ML picks are not silently discarded.
  - ``autoloc.amplTypeSNR`` annotation reminding that sceasyquake uploader.py
    must publish a companion SNR amplitude per pick (deadlock detection).
  - ``autoloc.amplTypeAbs = snr`` **critical**: scautoloc's ``hasAmplitude()``
    checks ``pick->amp > 0``, which is only set by ``amplTypeAbs``-type
    amplitudes (default ``mb``).  Since sceasyquake publishes ``type="snr"``
    amplitudes, ``pick->amp`` stays 0 forever unless ``amplTypeAbs`` is also
    set to ``snr``.  Without this every pick stalls "waiting for amplitude"
    and zero origins are ever produced.
  - ``buffer.pickKeep`` is properly set (replaces deprecated ``autoloc.maxAge``).
    If ``autoloc.maxAge = 0`` is set, scautoloc immediately discards all picks
    as "too old" and produces zero origins. Typical values: 1800-3600s (30-60 min)
    for real-time, 21600-86400s (6-24 hours) for gap recovery replay.
  - ``station-locations.conf`` exists, is reasonably fresh (< 7 days), and
    covers at least as many stations as are in the DB.

* **recent picks and SNR amplitudes** in the DB (last 10 min) with pairing
  ratio ≥ 90 % — confirms the pick-amplitude flow that scautoloc requires.

The output is plain text with diagnostic suggestions when something looks
wrong.  The intent is to run this periodically or manually when you suspect
problems and get a quick checklist of things to fix.

In addition to the live health check, this script can archive a point-in-time
snapshot of a SeisComP installation's configuration (``etc/``, station
locations, module status, versions, and the ``scfakequake`` / ``sceasyquake``
module details) and later diff two such snapshots against each other -
handy for comparing a production system to a dev/test system, or to see what
changed on a system over time.

A snapshot can also be *applied* as a config template to a new system: it
copies over module tuning parameters and binding-profile templates while
skipping station/region-specific artifacts (station bindings, seedlink
source profiles, inventory XML, the autoloc station-locations/grid files) -
those are what ``bbox_station_service.py`` should generate for the new
deployment's region. Typical flow to stand up a new system in minutes:

1. ``status_check.py snapshot`` on the existing/reference system.
2. Run ``bbox_station_service.py`` on the new system to build its station
   inventory/bindings for the new region.
3. ``status_check.py apply <snapshot_dir>`` on the new system to lay down
   the rest of the config (order relative to step 2 doesn't matter - apply
   never touches what bbox_station_service owns).
4. Fix the handful of files flagged for manual review (DB password,
   machine-specific absolute paths), then ``seiscomp update-config &&
   seiscomp restart``.

Usage:
    python examples/status_check.py [check]
    python examples/status_check.py snapshot [-o OUTPUT_DIR]
    python examples/status_check.py compare SNAPSHOT_A SNAPSHOT_B [-o OUTPUT_FILE]
    python examples/status_check.py apply SNAPSHOT_DIR [-y]

"""

import argparse
import difflib
import fnmatch
import hashlib
import json
import os
import subprocess
import socket
import shutil
import sys
import glob
import time
import re
from datetime import datetime, timedelta

SEISCOMP_ROOT = os.environ.get('SEISCOMP_ROOT', '/home/jwalter/seiscomp')
BACKUP_GLOBAL = os.path.expanduser('~/proc_backup/etc/defaults/global.cfg')
EASYQUAKE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# helper functions ---------------------------------------------------------

def _run(cmd, **kw):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
        return r.returncode, r.stdout + r.stderr
    except Exception as exc:
        return -1, str(exc)


def status_modules():
    rc, out = _run(['seiscomp', 'status'])
    lines = out.strip().splitlines()
    running = []
    stopped = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            name, _, state = parts[:3]
            if state == 'running':
                running.append(name)
            else:
                stopped.append(name)
    return running, stopped, out


def check_slarchive_activity(max_age_minutes=10):
    # look at rc_* files modification time
    path = os.path.join(SEISCOMP_ROOT, 'var', 'lib', 'slarchive')
    recent = []
    old = []
    now = time.time()
    if not os.path.isdir(path):
        return None, None
    for fn in glob.glob(os.path.join(path, 'rc_*')):
        try:
            m = os.path.getmtime(fn)
        except OSError:
            continue
        age = now - m
        station = os.path.basename(fn)[3:]
        if age < max_age_minutes * 60:
            recent.append(station)
        else:
            old.append(station)
    return recent, old


def check_bindings():
    keydir = os.path.join(SEISCOMP_ROOT, 'etc', 'key')
    problems = []
    if not os.path.isdir(keydir):
        return ['key directory missing']
    for fn in glob.glob(os.path.join(keydir, 'station_*')):
        try:
            with open(fn) as fh:
                text = fh.read()
        except Exception:
            problems.append(f'cannot read {fn}')
            continue
        if 'seedlink:' not in text:
            problems.append(f'{os.path.basename(fn)} missing seedlink line')
    return problems


def compare_global():
    if not os.path.exists(BACKUP_GLOBAL):
        return None
    def parse_cfg(fname):
        cfg = {}
        with open(fname) as fh:
            for line in fh:
                line=line.strip()
                if not line or line.startswith('#') or '=' not in line: continue
                k,v = line.split('=',1)
                cfg[k.strip()] = v.strip()
        return cfg
    cur = parse_cfg(os.path.join(SEISCOMP_ROOT,'etc','defaults','global.cfg'))
    ref = parse_cfg(BACKUP_GLOBAL)
    diffs = []
    for key in ('recordstream','connection.server','core.plugins'):
        a = cur.get(key)
        b = ref.get(key)
        if a != b:
            diffs.append((key,a,b))
    return diffs


def check_scautopick_enabled():
    rc, out = _run(['seiscomp','list','modules'])
    enabled = []
    for line in out.splitlines():
        if 'scautopick is enabled' in line:
            enabled.append('scautopick')
    return bool(enabled), out


def run_scautopick_debug():
    """Run `scautopick --debug` once and return stdout/stderr.

    We capture the exit code and output so the caller can look for the
    "No stations added" message which indicates that the module has no
    configured streams/stations and therefore will not pick anything.
    """
    rc, out = _run(['scautopick', '--debug'])
    return rc, out


def check_scautopick_cfg():
    """Look for obvious omissions in the scautopick configuration file.

    Returns a list of strings describing missing items.  Right now we only
    inspect the `streams`/`streamlist` setting; if neither is present we
    warn the user (this is usually why the debug output complains about
    "No stations added").
    """
    cfg_path = os.path.join(SEISCOMP_ROOT, 'etc', 'defaults', 'scautopick.cfg')
    problems = []
    if not os.path.exists(cfg_path):
        problems.append('scautopick.cfg not found')
        return problems
    has_streams = False
    with open(cfg_path) as fh:
        for line in fh:
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k,v = line.split('=',1)
            k=k.strip()
            v=v.strip()
            if k in ('streams','streamlist') and v:
                has_streams = True
                break
    if not has_streams:
        problems.append('no `streams` or `streamlist` defined in scautopick.cfg')
    return problems


def check_scrttv_cfg():
    """Verify scrttv has a usable streams.codes configuration.

    Returns a list of diagnostic strings.
    """
    problems = []
    # Check etc/scrttv.cfg (instance-level, highest priority after ~/.seiscomp)
    cfg_path = os.path.join(SEISCOMP_ROOT, 'etc', 'scrttv.cfg')
    defaults_path = os.path.join(SEISCOMP_ROOT, 'etc', 'defaults', 'scrttv.cfg')

    def _parse(path):
        cfg = {}
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    cfg[k.strip()] = v.strip()
        except OSError:
            pass
        return cfg

    instance = _parse(cfg_path)
    defaults = _parse(defaults_path)
    codes = instance.get('streams.codes') or defaults.get('streams.codes', '')

    if not codes:
        problems.append('streams.codes not set in scrttv.cfg or its defaults')
    elif codes.strip() == 'default':
        # 'default' relies solely on global bindings; warn if key dir is empty
        keydir = os.path.join(SEISCOMP_ROOT, 'etc', 'key')
        sta_keys = [f for f in os.listdir(keydir) if f.startswith('station_')] if os.path.isdir(keydir) else []
        if not sta_keys:
            problems.append('streams.codes = default but no station key files found '
                            '(apply bbox bindings first)')
        else:
            # check that at least one key file has a global line
            has_global = False
            for fn in sta_keys[:20]:
                try:
                    text = open(os.path.join(keydir, fn)).read()
                    if 'global' in text:
                        has_global = True
                        break
                except OSError:
                    pass
            if not has_global:
                problems.append('streams.codes = default but none of the station key '
                                'files contain a `global` binding – scrttv will show '
                                'nothing; run `seiscomp update-config scrttv`')
    else:
        count = len([p for p in codes.split(',') if p.strip()])
        if count == 0:
            problems.append('streams.codes is set but empty')
        # report the count as info (not a problem)
        problems.append(f'INFO: streams.codes has {count} channel pattern(s)')
        # Check for 3-component wildcards – scrttv should show Z only
        wildcard_entries = [p.strip() for p in codes.split(',') if p.strip().endswith('?')]
        if wildcard_entries:
            problems.append(
                f'ISSUE: {len(wildcard_entries)} streams.codes entries use "?" wildcard '
                f'(e.g. {wildcard_entries[0]}) – scrttv will show 3 traces per station. '
                f'Replace trailing "?" with "Z" (e.g. HH? → HHZ) so only the vertical '
                f'component is displayed.'
            )

    return problems


def _parse_cfg(path):
    """Parse a simple key=value SeisComP config file into a dict."""
    cfg = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.strip()
    except OSError:
        pass
    return cfg


def _db_credentials():
    """Return (user, password, host, database) from scmaster.cfg, or None."""
    cfg_path = os.path.join(SEISCOMP_ROOT, 'etc', 'scmaster.cfg')
    try:
        with open(cfg_path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('queues.production.processors.messages.dbstore.read'):
                    db_url = line.split('=', 1)[1].strip()
                    # url is like sysop:sysop@localhost/seiscomp
                    m = re.match(r'(?:mysql://)?([^:]+):([^@]+)@([^/]+)/(.+)', db_url)
                    if m:
                        return m.groups()
    except OSError:
        pass
    return None


def _db_query(sql):
    """Run a MySQL query against the SeisComP database.

    Returns (returncode, output_text). On credential failure returns (-1, msg).
    """
    creds = _db_credentials()
    if creds is None:
        return -1, 'Cannot read DB credentials from scmaster.cfg'
    user, pw, host, db = creds
    return _run(['mysql', f'-u{user}', f'-p{pw}', '-h', host, db, '-e', sql])


def _db_station_count():
    """Return number of stations in the SeisComP MySQL database, or None."""
    rc, out = _db_query('SELECT COUNT(*) FROM Station;')
    if rc != 0:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def check_scautoloc_config():
    """Audit the scautoloc configuration and station-locations.conf file.

    Checks:
    - scautoloc.cfg exists with autoloc.stationLocations set
    - station-locations.conf file exists, is non-empty, and is < 7 days old
    - station-locations.conf coverage vs DB station count
    - locator.profile is set (detects use of deprecated autoloc.locator.profile key)
    - autoloc.networkType is 'local' for regional networks
    - autoloc.minPhaseCount is sensible for local events
    - autoloc.minPickSNR is 0 (so picks are not silently discarded)
    - autoloc.amplTypeSNR deadlock: warns if sceasyquake SNR amplitudes are required
    - buffer.pickKeep is set correctly (replaces deprecated autoloc.maxAge)
    - autoloc.maxAge=0 bug detection (causes all picks to be discarded)
    """
    problems = []
    infos = []

    cfg_path = os.path.join(SEISCOMP_ROOT, 'etc', 'scautoloc.cfg')
    defaults_path = os.path.join(SEISCOMP_ROOT, 'etc', 'defaults', 'scautoloc.cfg')
    cfg = {}
    cfg.update(_parse_cfg(defaults_path))
    cfg.update(_parse_cfg(cfg_path))

    if not os.path.exists(cfg_path):
        problems.append('etc/scautoloc.cfg does not exist – no user config applied')

    # --- station locations file ---
    sta_loc = cfg.get('autoloc.stationLocations', '')
    if not sta_loc:
        problems.append('autoloc.stationLocations not set; scautoloc will reject '
                        'all picks with "not found in station inventory"')
    else:
        # Expand @DATADIR@ macro
        datadir = os.path.join(SEISCOMP_ROOT, 'share')
        sta_loc_path = sta_loc.replace('@DATADIR@', datadir)
        if not os.path.exists(sta_loc_path):
            problems.append(f'autoloc.stationLocations file not found: {sta_loc_path}')
        else:
            with open(sta_loc_path) as fh:
                loc_lines = [l for l in fh if l.strip() and not l.startswith('#')]
            loc_count = len(loc_lines)
            infos.append(f'station-locations.conf has {loc_count} entries')

            # Freshness check: warn if older than 7 days
            mtime = os.path.getmtime(sta_loc_path)
            age_days = (time.time() - mtime) / 86400
            if age_days > 7:
                problems.append(
                    f'station-locations.conf is {age_days:.0f} days old – rebuild it '
                    f'after scinv sync so new stations are not rejected by scautoloc.  '
                    f'Call: bbox_station_service._update_scautoloc_station_locations()'
                )
            else:
                infos.append(f'station-locations.conf age: {age_days:.1f} days (fresh)')

            db_count = _db_station_count()
            if db_count is not None:
                missing = db_count - loc_count
                if missing > 0:
                    problems.append(
                        f'station-locations.conf has {loc_count} stations but DB has '
                        f'{db_count} – {missing} stations will be rejected by scautoloc. '
                        f'Call: bbox_station_service._update_scautoloc_station_locations()'
                    )
                else:
                    infos.append(f'station-locations.conf covers all {db_count} DB stations')

    # --- locator profile ---
    # SeisComP 5+ uses the top-level key "locator.profile"; the old
    # "autoloc.locator.profile" key is deprecated and ignored in newer builds.
    profile = cfg.get('locator.profile') or cfg.get('autoloc.locator.profile', '')
    deprecated_key_only = (not cfg.get('locator.profile')
                           and bool(cfg.get('autoloc.locator.profile')))
    if not profile:
        problems.append('locator.profile not set in scautoloc.cfg; '
                        'scautoloc cannot locate origins.  '
                        'Add: locator.profile = iasp91')
    else:
        if deprecated_key_only:
            problems.append(
                f'autoloc.locator.profile = {profile!r} is set but the correct key is '
                f'"locator.profile" – the deprecated key is silently ignored in newer '
                f'SeisComP builds.  Replace with: locator.profile = {profile}'
            )
        # Verify the profile tables actually exist
        tables_dir = os.path.join(SEISCOMP_ROOT, 'share', 'locsat', 'tables')
        p_table = os.path.join(tables_dir, f'{profile}.P')
        if not os.path.exists(p_table):
            problems.append(f'locsat table not found for profile "{profile}": {p_table}')
        else:
            infos.append(f'locator profile: {profile} (tables present)')

    # --- SNR amplitude deadlock ---
    # scautoloc holds every pick in a "waiting for amplitude" queue until it
    # receives an Amplitude(type=amplTypeSNR) on the AMPLITUDE messaging group.
    # scamp only computes amplitudes AFTER an origin exists → deadlock.
    # Fix: sceasyquake uploader.py publishes a companion snr amplitude per pick,
    # and autoloc.minPickSNR must be 0 so those amplitudes are not discarded.
    amp_type_snr = cfg.get('autoloc.amplTypeSNR', 'snr')
    if amp_type_snr:
        infos.append(f'autoloc.amplTypeSNR: {amp_type_snr!r} '
                     f'(sceasyquake uploader.py must publish Amplitude(type={amp_type_snr!r}) '
                     f'per pick or scautoloc will stall)')
    try:
        min_snr = float(cfg.get('autoloc.minPickSNR', '3'))
    except ValueError:
        min_snr = 3.0
    if min_snr > 0:
        problems.append(
            f'autoloc.minPickSNR = {min_snr} – scautoloc will silently discard picks '
            f'whose SNR amplitude is below this threshold.  '
            f'Because sceasyquake publishes synthetic SNR = probability × 10, any pick '
            f'with probability < {min_snr/10:.2f} will be dropped.  '
            f'Set autoloc.minPickSNR = 0 in etc/scautoloc.cfg to use all ML picks.'
        )
    else:
        infos.append(f'autoloc.minPickSNR: {min_snr} (all SNR amplitudes accepted)')

    # --- amplTypeAbs must match amplTypeSNR so hasAmplitude() is satisfied ---
    # scautoloc's Autoloc3::hasAmplitude() only checks pick->amp > 0.
    # pick->amp is set only by amplitudes whose type == amplTypeAbs (default "mb").
    # pick->snr is set by amplitudes whose type == amplTypeSNR (default "snr").
    # sceasyquake publishes Amplitude(type="snr"), so pick->snr is populated but
    # pick->amp stays 0 when amplTypeAbs is the default "mb".  The result:
    # every pick stays "waiting for amplitude" forever → 0 origins ever produced.
    # Fix: autoloc.amplTypeAbs = snr  (in etc/scautoloc.cfg)
    amp_type_abs = cfg.get('autoloc.amplTypeAbs', 'mb')
    if amp_type_abs != amp_type_snr:
        problems.append(
            f'autoloc.amplTypeAbs = {amp_type_abs!r} but autoloc.amplTypeSNR = {amp_type_snr!r}. '
            f"scautoloc's hasAmplitude() checks pick->amp which is only set by amplTypeAbs. "
            f'sceasyquake publishes Amplitude(type={amp_type_snr!r}), so pick->amp stays 0 '
            f'and every pick stalls "waiting for amplitude" → zero origins are ever produced. '
            f'Add to etc/scautoloc.cfg:  autoloc.amplTypeAbs = {amp_type_snr}'
        )
    else:
        infos.append(
            f'autoloc.amplTypeAbs: {amp_type_abs!r} '
            f'(matches amplTypeSNR – hasAmplitude() will fire correctly)'
        )

    # --- network type ---
    net_type = cfg.get('autoloc.networkType', '')
    if net_type not in ('local', 'regional', 'global'):
        problems.append(f'autoloc.networkType="{net_type}" – set to "local" for '
                        'regional seismic networks to disable teleseismic nucleation')
    else:
        infos.append(f'autoloc.networkType: {net_type}')

    # --- minimum phase count ---
    try:
        min_phase = int(cfg.get('autoloc.minPhaseCount', 8))
    except ValueError:
        min_phase = 8
    if min_phase > 6:
        problems.append(f'autoloc.minPhaseCount={min_phase} – for a local/regional '
                        'network consider lowering to 4-6 to detect small events')
    else:
        infos.append(f'autoloc.minPhaseCount: {min_phase}')

    # --- pick buffer age (buffer.pickKeep replaces deprecated autoloc.maxAge) ---
    # scautoloc holds picks in a buffer for association. If buffer.pickKeep is too
    # small (or autoloc.maxAge=0), picks are immediately discarded as "too old" and
    # zero origins are produced. The default is typically 1800s (30 min).
    # For gap recovery replay, a larger value is needed (e.g., 6-24 hours).
    max_age_str = cfg.get('autoloc.maxAge', '')
    pick_keep_str = cfg.get('buffer.pickKeep', '')
    
    if max_age_str:
        try:
            max_age = float(max_age_str)
        except ValueError:
            max_age = None
        
        if max_age is not None and max_age == 0:
            problems.append(
                'autoloc.maxAge = 0 is set – this is DEPRECATED and causes scautoloc to '
                'immediately discard ALL picks as "too old" → zero origins are produced. '
                'Remove autoloc.maxAge and set buffer.pickKeep instead (e.g., 21600 = 6 hours).'
            )
        elif max_age_str and not pick_keep_str:
            problems.append(
                f'autoloc.maxAge = {max_age_str} is DEPRECATED and may not map correctly '
                'to buffer.pickKeep in newer SeisComP versions. '
                'Replace with: buffer.pickKeep = {value in seconds} '
                '(e.g., 3600 for 1 hour, 21600 for 6 hours)'
            )
    
    if pick_keep_str:
        try:
            pick_keep = float(pick_keep_str)
        except ValueError:
            pick_keep = None
        
        if pick_keep is not None:
            if pick_keep < 1800:  # Less than 30 minutes
                problems.append(
                    f'buffer.pickKeep = {pick_keep:.0f}s ({pick_keep/60:.0f} min) is very '
                    'small – picks older than this are immediately discarded. For real-time '
                    'processing use 1800-3600s (30-60 min); for gap recovery replay use '
                    '6-24 hours (21600-86400s).'
                )
            elif pick_keep > 86400:  # More than 24 hours
                infos.append(
                    f'buffer.pickKeep: {pick_keep:.0f}s ({pick_keep/3600:.1f} hours) '
                    '– large value suitable for gap recovery replay'
                )
            else:
                infos.append(f'buffer.pickKeep: {pick_keep:.0f}s ({pick_keep/3600:.1f} hours)')
    elif not max_age_str:
        # Neither is set - will use SeisComP default (typically 1800s)
        infos.append(
            'buffer.pickKeep not set – using SeisComP default (typically 1800s = 30 min). '
            'For gap recovery replay, set buffer.pickKeep = 21600 (6 hours) or larger.'
        )

    return problems, infos


def check_event_pipeline(running_modules):
    """Verify the full pick → origin → event → magnitude pipeline is running.

    Returns a list of problem strings.
    """
    problems = []
    pipeline = [
        ('scdb',        'database writer (scdb not running – picks/origins will NOT '
                        'persist to DB and scolv will show nothing)'),
        ('scamp',       'amplitude calculator (scamp not running – station magnitudes '
                        'will be incomplete; also needs recordstream in global.cfg)'),
        ('sceasyquake', 'ML pick generator (sceasyquake not running – no picks will be generated)'),
        ('scautoloc',   'associator (scautoloc not running – picks will not be associated into origins)'),
        ('scevent',     'event builder (scevent not running – origins will not be grouped into events)'),
        ('scmag',       'magnitude estimator (scmag not running – events will have no magnitude)'),
    ]
    for module, description in pipeline:
        if module not in running_modules:
            problems.append(f'{module} is NOT running – {description}')
    return problems


def check_key_file_validity():
    """Scan station key files for lines that SeisComP rejects as invalid bindings.

    Specifically, `sources.chain.selectors = ...` is NOT a valid station binding
    parameter and causes `seiscomp update-config slarchive` to emit
    ``invalid binding`` errors which can block proper configuration.
    Returns a list of problem descriptions.
    """
    keydir = os.path.join(SEISCOMP_ROOT, 'etc', 'key')
    problems = []
    if not os.path.isdir(keydir):
        return ['key directory missing']
    bad = []
    for fn in os.listdir(keydir):
        if not fn.startswith('station_'):
            continue
        path = os.path.join(keydir, fn)
        try:
            with open(path) as fh:
                for lineno, line in enumerate(fh, 1):
                    stripped = line.strip()
                    # These parameter lines are not valid in a station key file
                    if stripped.startswith('sources.chain.selectors') or stripped.startswith('keep ='):
                        bad.append(f'{fn}:{lineno}: {stripped!r}')
        except OSError:
            pass
    if bad:
        problems.append(f'{len(bad)} invalid parameter line(s) found in key files '
                        '(run bbox apply to rewrite them correctly):')
        for b in bad[:5]:
            problems.append(f'  {b}')
        if len(bad) > 5:
            problems.append(f'  …and {len(bad)-5} more')
    return problems

def check_global_cfg():
    """Verify global.cfg has the minimum required keys for the sceasyquake pipeline.

    Checks:
    - recordstream is set to a SeedLink URI (needed by scamp for waveform access)
    - core.plugins includes dbmysql (needed to access the MySQL database)
    - agencyID / datacenterID are set

    Returns (problems, infos).
    """
    problems = []
    infos = []
    cfg_path = os.path.join(SEISCOMP_ROOT, 'etc', 'global.cfg')
    cfg = _parse_cfg(cfg_path)

    # recordstream – scamp needs this to fetch waveforms
    rs = cfg.get('recordstream', '')
    if not rs:
        problems.append(
            'recordstream not set in global.cfg – scamp cannot fetch waveforms '
            'and will compute no amplitudes, leaving magnitude gaps.  '
            'Add: recordstream = slink://localhost:18000'
        )
    elif 'slink://' not in rs and 'fdsnws://' not in rs:
        problems.append(
            f'recordstream = {rs!r} does not look like a SeedLink or FDSNWS URI.  '
            'For a local SeisComP installation use: recordstream = slink://localhost:18000'
        )
    else:
        infos.append(f'recordstream: {rs}')

    # core.plugins
    plugins = cfg.get('core.plugins', '')
    if 'dbmysql' not in plugins:
        problems.append(
            'core.plugins does not include "dbmysql" – database access will fail.  '
            'Add: core.plugins = dbmysql'
        )
    else:
        infos.append(f'core.plugins: {plugins}')

    # agencyID
    if not cfg.get('agencyID', ''):
        problems.append('agencyID not set in global.cfg')
    else:
        infos.append(f'agencyID: {cfg["agencyID"]}')

    return problems, infos


def check_recent_picks_amplitudes(minutes=10):
    """Check that recent picks and SNR amplitudes are being written to the DB.

    Queries the SeisComP MySQL database for picks and Amplitude(type='snr')
    objects created in the last *minutes* minutes.  A healthy sceasyquake
    installation should have a pick-to-amplitude ratio ≥ 90%.

    Without SNR amplitudes scautoloc holds every pick as "waiting for amplitude"
    and never forms origins, causing a complete deadlock.

    Returns (problems, infos).
    """
    problems = []
    infos = []

    # Count picks in the last N minutes
    pick_sql = (
        f"SELECT COUNT(*) FROM Pick p "
        f"WHERE p._last_modified > NOW() - INTERVAL {minutes} MINUTE;"
    )
    rc, out = _db_query(pick_sql)
    if rc != 0:
        problems.append(f'DB pick query failed (is scdb running?): {out.strip()[:120]}')
        return problems, infos

    pick_count = 0
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pick_count = int(line)
            break
    infos.append(f'Picks in last {minutes} min: {pick_count}')

    if pick_count == 0:
        problems.append(
            f'No picks in the last {minutes} minutes – sceasyquake may not be running '
            f'or streaming waveform data is not arriving.  '
            f'Check: seiscomp status sceasyquake  and  seiscomp status seedlink'
        )
        return problems, infos

    # Count SNR amplitudes in the last N minutes
    amp_sql = (
        f"SELECT COUNT(*) FROM Amplitude a "
        f"WHERE a.type = 'snr' "
        f"AND a._last_modified > NOW() - INTERVAL {minutes} MINUTE;"
    )
    rc, out = _db_query(amp_sql)
    if rc != 0:
        problems.append(f'DB amplitude query failed: {out.strip()[:120]}')
        return problems, infos

    amp_count = 0
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            amp_count = int(line)
            break
    infos.append(f'SNR amplitudes in last {minutes} min: {amp_count}')

    if amp_count == 0:
        problems.append(
            f'No SNR amplitudes in the last {minutes} minutes despite {pick_count} picks.  '
            f'scautoloc will hold every pick as "waiting for amplitude" and never form '
            f'origins – this is the pick-amplitude deadlock.  '
            f'Fix: ensure uploader.py publishes Amplitude(type="snr") alongside each pick '
            f'(see sceasyquake/lib/sceasyquake/uploader.py _send_via_seiscomp).'
        )
    else:
        ratio = amp_count / pick_count
        if ratio < 0.9:
            problems.append(
                f'Pick-amplitude ratio is {ratio:.0%} ({amp_count}/{pick_count}) – '
                f'expected ≥ 90%.  Some picks are missing companion SNR amplitudes; '
                f'scautoloc will stall on those picks indefinitely.'
            )
        else:
            infos.append(f'Pick-amplitude pairing ratio: {ratio:.0%} (healthy ≥ 90%)')

    return problems, infos


def check_picks_and_events(hours=2):
    """Report pick count and event list for the last *hours* hours.

    Picks are counted by their seismic arrival time (Pick.time_value).
    Events are retrieved by joining Event → Origin (preferred) → Magnitude
    (preferred) and filtered by the origin time.

    Returns a dict with keys:
      pick_count   – int or None if query failed
      pick_error   – str error message if query failed
      events       – list of dicts {time, lat, lon, depth_km, mag, mag_type,
                     phases, event_id} sorted newest-first
      event_error  – str error message if query failed
    """
    result = {'pick_count': None, 'pick_error': None,
              'events': [], 'event_error': None}

    # --- picks ---
    pick_sql = (
        f"SELECT COUNT(*) FROM Pick "
        f"WHERE time_value > NOW() - INTERVAL {hours} HOUR;"
    )
    rc, out = _db_query(pick_sql)
    if rc != 0:
        result['pick_error'] = out.strip()[:200]
    else:
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                result['pick_count'] = int(line)
                break

    # --- events via preferred origin + preferred magnitude ---
    event_sql = (
        "SELECT "
        "  po_e.publicID AS event_id, "
        "  o.time_value, "
        "  o.latitude_value, "
        "  o.longitude_value, "
        "  o.depth_value, "
        "  o.quality_usedPhaseCount, "
        "  m.magnitude_value, "
        "  m.type AS mag_type "
        "FROM Event e "
        "JOIN PublicObject po_e ON po_e._oid = e._oid "
        "JOIN PublicObject po_o ON po_o.publicID = e.preferredOriginID "
        "JOIN Origin o ON o._oid = po_o._oid "
        "LEFT JOIN PublicObject po_m ON po_m.publicID = e.preferredMagnitudeID "
        "LEFT JOIN Magnitude m ON m._oid = po_m._oid "
        f"WHERE o.time_value > NOW() - INTERVAL {hours} HOUR "
        "ORDER BY o.time_value DESC;"
    )
    rc, out = _db_query(event_sql)
    if rc != 0:
        result['event_error'] = out.strip()[:200]
    else:
        for line in out.splitlines():
            parts = line.split('\t')
            if len(parts) < 7:
                continue
            event_id, t_val, lat, lon, depth, phases, mag, *rest = parts
            # skip header row
            if event_id == 'event_id':
                continue
            mag_type = rest[0] if rest else ''
            try:
                result['events'].append({
                    'event_id': event_id,
                    'time': t_val,
                    'lat': float(lat),
                    'lon': float(lon),
                    'depth_km': float(depth) if depth not in ('NULL', '') else None,
                    'phases': int(phases) if phases not in ('NULL', '') else None,
                    'mag': float(mag) if mag not in ('NULL', '') else None,
                    'mag_type': mag_type.strip() if mag_type.strip() not in ('NULL', '') else '',
                })
            except (ValueError, IndexError):
                continue

    return result


def check_pick_latency(n=50):
    """Report the latency of the most recent picks.

    Two latency values are computed per pick:

    * **Data age** – ``NOW() - time_value``: how old is the most recent pick
      arrival time?  A healthy real-time system should have picks arriving
      within the last few seconds.  Values > 60 s suggest waveform delivery
      has stopped or sceasyquake is stalled.

    * **Processing latency** – ``_last_modified - time_value``: time between
      the seismic arrival and when sceasyquake wrote the pick to the DB.
      This includes waveform buffering, ML inference, and messaging overhead.
      Typical values are 5-30 s depending on step_seconds and GPU speed.

    Queries the *n* most recent picks (by arrival time) and returns
    min/median/max/mean for each metric, plus the timestamp of the newest pick.

    Returns a dict with keys:
      newest_pick_time  – str UTC timestamp of most recent pick
      age_now_s         – float seconds since newest pick arrival
      latencies_s       – list of processing latency values (float seconds)
      lat_min, lat_med, lat_mean, lat_max  – summary stats (float or None)
      error             – str if query failed
    """
    sql = (
        f"SELECT time_value, _last_modified "
        f"FROM Pick "
        f"ORDER BY time_value DESC "
        f"LIMIT {n};"
    )
    rc, out = _db_query(sql)
    if rc != 0:
        return {'error': out.strip()[:200]}

    rows = []
    for line in out.splitlines():
        parts = line.strip().split('\t')
        if len(parts) < 2 or parts[0] == 'time_value':
            continue
        rows.append(parts)

    if not rows:
        return {'error': 'no picks found in database'}

    from datetime import timezone

    def _parse_dt(s):
        """Parse a MySQL datetime string to a UTC-aware datetime."""
        from datetime import datetime
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
            try:
                return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    newest_dt = _parse_dt(rows[0][0])
    age_now_s = (now - newest_dt).total_seconds() if newest_dt else None

    latencies = []
    for time_val, last_mod in rows:
        t_arr  = _parse_dt(time_val)
        t_mod  = _parse_dt(last_mod)
        if t_arr and t_mod:
            latencies.append((t_mod - t_arr).total_seconds())

    if latencies:
        latencies_sorted = sorted(latencies)
        n_lat = len(latencies_sorted)
        mid = n_lat // 2
        lat_med = (latencies_sorted[mid] if n_lat % 2
                   else (latencies_sorted[mid-1] + latencies_sorted[mid]) / 2)
        return {
            'newest_pick_time': rows[0][0],
            'age_now_s':  round(age_now_s, 1) if age_now_s is not None else None,
            'latencies_s': latencies,
            'lat_min':  round(min(latencies), 1),
            'lat_med':  round(lat_med, 1),
            'lat_mean': round(sum(latencies) / len(latencies), 1),
            'lat_max':  round(max(latencies), 1),
            'count':    n_lat,
        }
    return {
        'newest_pick_time': rows[0][0],
        'age_now_s': round(age_now_s, 1) if age_now_s is not None else None,
        'error': 'could not compute latencies (timestamp parse failed)',
    }


# snapshot / compare --------------------------------------------------------

_DB_URL_RE = re.compile(r'(=\s*)([A-Za-z0-9_.+-]+):([^@\s]+)(@\S+)')


def _redact_secrets(path):
    """Mask DB passwords (e.g. in scmaster.cfg) in a copied config file in place."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return
    redacted = _DB_URL_RE.sub(lambda m: f'{m.group(1)}{m.group(2)}:***REDACTED***{m.group(4)}', text)
    if redacted != text:
        with open(path, 'w') as fh:
            fh.write(redacted)


def _git_info(repo_path):
    """Return {'commit', 'branch', 'dirty'} for a git repo, or None."""
    if not os.path.isdir(os.path.join(repo_path, '.git')):
        return None
    rc, commit = _run(['git', '-C', repo_path, 'rev-parse', 'HEAD'])
    if rc != 0:
        return None
    rc, branch = _run(['git', '-C', repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'])
    rc, dirty_out = _run(['git', '-C', repo_path, 'status', '--porcelain'])
    return {
        'commit': commit.strip(),
        'branch': branch.strip() if rc == 0 else None,
        'dirty': bool(dirty_out.strip()),
    }


def _hash_file(path, chunk_size=1 << 20):
    """Return the sha256 hex digest of a file, or None on error."""
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _seiscomp_version():
    """Parse `seiscomp exec scmaster -V` into a dict of version fields."""
    rc, out = _run(['seiscomp', 'exec', 'scmaster', '-V'])
    info = {}
    for line in out.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            info[k.strip()] = v.strip()
    return info


def _scfakequake_snapshot(seiscomp_root):
    """Capture scfakequake config plus the state of its external FakeQuake repo/model."""
    info = {'cfg': {}}
    cfg_path = os.path.join(seiscomp_root, 'etc', 'scfakequake.cfg')
    info['cfg'] = _parse_cfg(cfg_path)
    info['cfg_exists'] = os.path.exists(cfg_path)

    repo = info['cfg'].get('scfakequake.fakequake_root', '')
    if repo:
        repo = os.path.expanduser(repo)
        info['fakequake_root'] = repo
        info['fakequake_root_exists'] = os.path.isdir(repo)
        info['fakequake_git'] = _git_info(repo)

    model = info['cfg'].get('scfakequake.model_path', '')
    if model:
        model = os.path.expanduser(model)
        info['model_path'] = model
        if os.path.exists(model):
            st = os.stat(model)
            info['model_size_bytes'] = st.st_size
            info['model_mtime'] = datetime.utcfromtimestamp(st.st_mtime).isoformat() + 'Z'
            info['model_sha256'] = _hash_file(model)
        else:
            info['model_exists'] = False

    return info


def _sceasyquake_snapshot():
    """Capture the git state of this easyQuake_seiscomp checkout (sceasyquake/worker code)."""
    return {'git': _git_info(EASYQUAKE_ROOT)}


def snapshot_system(output_dir=None, seiscomp_root=SEISCOMP_ROOT):
    """Archive etc/ configuration, module status, and key module details.

    Copies the whole `etc/` tree (station keys, module .cfg files, global.cfg)
    plus scautoloc's station-locations/grid conf, redacts the DB password in
    the copied scmaster.cfg, and records module status/versions plus
    scfakequake/sceasyquake specific details (git commit, model hash) in a
    manifest.json so two snapshots can be meaningfully diffed later.

    Returns the path to the snapshot directory.
    """
    hostname = socket.gethostname()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = os.path.join('seiscomp_snapshots', f'{hostname}_{ts}')

    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise SystemExit(f'Refusing to overwrite non-empty directory: {output_dir}')

    etc_src = os.path.join(seiscomp_root, 'etc')
    if not os.path.isdir(etc_src):
        raise SystemExit(f'SeisComP etc/ directory not found: {etc_src}')

    shutil.copytree(etc_src, os.path.join(output_dir, 'etc'))
    _redact_secrets(os.path.join(output_dir, 'etc', 'scmaster.cfg'))

    # scautoloc station locations / grid config live under share/, not etc/
    autoloc_share_src = os.path.join(seiscomp_root, 'share', 'scautoloc')
    if os.path.isdir(autoloc_share_src):
        autoloc_dest = os.path.join(output_dir, 'share', 'scautoloc')
        os.makedirs(autoloc_dest, exist_ok=True)
        for fn in glob.glob(os.path.join(autoloc_share_src, '*.conf')):
            shutil.copy2(fn, autoloc_dest)

    running, stopped, raw_status = status_modules()
    with open(os.path.join(output_dir, 'modules.txt'), 'w') as fh:
        fh.write(raw_status)

    version_info = _seiscomp_version()

    db_info = {}
    station_count = _db_station_count()
    if station_count is not None:
        db_info['station_count'] = station_count

    scfakequake_info = _scfakequake_snapshot(seiscomp_root)
    sceasyquake_info = _sceasyquake_snapshot()

    manifest = {
        'hostname': hostname,
        'timestamp': ts,
        'seiscomp_root': seiscomp_root,
        'seiscomp_version': version_info,
        'python_version': sys.version,
        'modules_running': running,
        'modules_stopped': stopped,
        'db_info': db_info,
        'scfakequake': scfakequake_info,
        'sceasyquake': sceasyquake_info,
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    print(f'Snapshot written to: {output_dir}')
    print(f'  host: {hostname}   seiscomp: {version_info.get("Framework", "?")}')
    print(f'  modules running: {len(running)}   stopped: {len(stopped)}')
    if scfakequake_info.get('fakequake_git'):
        g = scfakequake_info['fakequake_git']
        print(f'  scfakequake FakeQuake repo: {g["commit"][:12]} '
              f'({"dirty" if g["dirty"] else "clean"})')
    if sceasyquake_info.get('git'):
        g = sceasyquake_info['git']
        print(f'  sceasyquake (easyQuake_seiscomp) repo: {g["commit"][:12]} '
              f'({"dirty" if g["dirty"] else "clean"})')
    return output_dir


def _iter_relative_files(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            yield os.path.relpath(full, root)


def _read_text_or_none(path):
    try:
        with open(path, encoding='utf-8', errors='strict') as fh:
            return fh.readlines()
    except (OSError, UnicodeDecodeError):
        return None


def compare_snapshots(dir_a, dir_b, output_path=None):
    """Diff two snapshot directories produced by snapshot_system().

    Prints a manifest summary (host/version/module/git differences) followed
    by a unified diff for every config file that differs between the two
    snapshots, plus lists of files only present in one snapshot.
    """
    lines = []

    def emit(s=''):
        lines.append(s)

    for d in (dir_a, dir_b):
        if not os.path.isdir(d):
            raise SystemExit(f'Not a snapshot directory: {d}')

    manifest_a = {}
    manifest_b = {}
    for d, target in ((dir_a, 'a'), (dir_b, 'b')):
        mpath = os.path.join(d, 'manifest.json')
        if os.path.exists(mpath):
            with open(mpath) as fh:
                (manifest_a if target == 'a' else manifest_b).update(json.load(fh))

    emit(f'Comparing snapshots:\n  A: {dir_a}\n  B: {dir_b}')
    emit()
    emit('--- manifest summary ---')
    for key in ('hostname', 'timestamp', 'seiscomp_root'):
        emit(f'  {key:16s} A={manifest_a.get(key, "?")!r}  B={manifest_b.get(key, "?")!r}')
    ver_a = manifest_a.get('seiscomp_version', {}).get('Framework', '?')
    ver_b = manifest_b.get('seiscomp_version', {}).get('Framework', '?')
    emit(f'  {"seiscomp":16s} A={ver_a!r}  B={ver_b!r}')

    for key, label in (('scfakequake', 'scfakequake'), ('sceasyquake', 'sceasyquake')):
        git_a = (manifest_a.get(key) or {}).get('fakequake_git') or (manifest_a.get(key) or {}).get('git')
        git_b = (manifest_b.get(key) or {}).get('fakequake_git') or (manifest_b.get(key) or {}).get('git')
        if git_a or git_b:
            ca = (git_a or {}).get('commit', '?')
            cb = (git_b or {}).get('commit', '?')
            marker = '' if ca == cb else '  <-- DIFFERS'
            emit(f'  {label + " git":16s} A={ca[:12]!r}  B={cb[:12]!r}{marker}')

    running_a, running_b = set(manifest_a.get('modules_running', [])), set(manifest_b.get('modules_running', []))
    only_a = sorted(running_a - running_b)
    only_b = sorted(running_b - running_a)
    if only_a or only_b:
        emit('  modules running only in A: ' + (', '.join(only_a) or '(none)'))
        emit('  modules running only in B: ' + (', '.join(only_b) or '(none)'))
    else:
        emit('  running modules match')
    emit()

    files_a = set(_iter_relative_files(dir_a)) - {'manifest.json'}
    files_b = set(_iter_relative_files(dir_b)) - {'manifest.json'}
    common = sorted(files_a & files_b)
    only_in_a = sorted(files_a - files_b)
    only_in_b = sorted(files_b - files_a)

    n_diff = 0
    n_same = 0
    n_binary_diff = 0
    emit('--- file differences ---')
    for rel in common:
        pa, pb = os.path.join(dir_a, rel), os.path.join(dir_b, rel)
        text_a, text_b = _read_text_or_none(pa), _read_text_or_none(pb)
        if text_a is None or text_b is None:
            if _hash_file(pa) != _hash_file(pb):
                n_binary_diff += 1
                emit(f'Binary files differ: {rel}')
            else:
                n_same += 1
            continue
        if text_a == text_b:
            n_same += 1
            continue
        n_diff += 1
        diff = difflib.unified_diff(text_a, text_b, fromfile=f'A/{rel}', tofile=f'B/{rel}')
        emit(''.join(diff).rstrip('\n'))
        emit()

    if only_in_a:
        emit(f'Only in A ({len(only_in_a)}):')
        for rel in only_in_a:
            emit(f'  {rel}')
    if only_in_b:
        emit(f'Only in B ({len(only_in_b)}):')
        for rel in only_in_b:
            emit(f'  {rel}')

    emit()
    emit(f'Summary: {n_diff} text file(s) differ, {n_binary_diff} binary file(s) differ, '
         f'{n_same} identical, {len(only_in_a)} only in A, {len(only_in_b)} only in B')

    report = '\n'.join(lines)
    if output_path:
        with open(output_path, 'w') as fh:
            fh.write(report)
        print(f'Comparison report written to: {output_path}')
        print(f'Summary: {n_diff} differ, {n_binary_diff} binary differ, {n_same} identical, '
              f'{len(only_in_a)} only in A, {len(only_in_b)} only in B')
    else:
        print(report)
    return report


# apply (snapshot -> new system template) -----------------------------------

# Paths that describe a *specific* set of stations rather than reusable
# module tuning.  bbox_station_service.py is the tool responsible for
# (re)generating these for a new deployment's region, so `apply` always
# leaves them alone rather than overwriting them from another system's
# snapshot.
_APPLY_EXCLUDE_PATTERNS = (
    'etc/key/station_*',
    'etc/key/seedlink/*',
    'etc/inventory/*',
    'share/scautoloc/station-locations.conf',
    'share/scautoloc/grid.conf',
    'share/scautoloc/station.conf',
)


def _matches_exclude(relpath):
    return any(fnmatch.fnmatch(relpath, pat) for pat in _APPLY_EXCLUDE_PATTERNS)


def _needs_review(full_path):
    """Flag files containing a redacted secret or a machine-specific absolute path."""
    try:
        with open(full_path, encoding='utf-8', errors='ignore') as fh:
            text = fh.read()
    except OSError:
        return False
    if 'REDACTED' in text:
        return True
    return bool(re.search(r'=\s*/home/\S+', text))


def classify_snapshot_for_apply(snapshot_dir):
    """Sort a snapshot's etc/ + share/scautoloc files into copy/exclude/review buckets.

    - exclude: station bindings, seedlink source profiles, inventory XML, and
      the autoloc station-locations/grid files.  These describe a specific
      set of stations and should instead be (re)generated by
      bbox_station_service.py for the new deployment's region.
    - review: copied as-is but flagged because the file has a redacted secret
      (e.g. scmaster.cfg's DB password) or a machine-specific absolute path
      (e.g. scfakequake's fakequake_root/model_path) that needs hand editing.
    - copy: everything else - module tuning parameters and binding profile
      templates (etc/key/global, access, scautopick, scwfparam, slarchive,
      etc/defaults, ...).  Safe to reuse verbatim on a new system.
    """
    copy, exclude, review = [], [], []
    for prefix in ('etc', os.path.join('share', 'scautoloc')):
        root = os.path.join(snapshot_dir, prefix)
        if not os.path.isdir(root):
            continue
        for rel in _iter_relative_files(root):
            relpath = os.path.join(prefix, rel).replace(os.sep, '/')
            if _matches_exclude(relpath):
                exclude.append(relpath)
            elif _needs_review(os.path.join(root, rel)):
                review.append(relpath)
            else:
                copy.append(relpath)
    return sorted(copy), sorted(exclude), sorted(review)


def apply_snapshot(snapshot_dir, seiscomp_root=SEISCOMP_ROOT, execute=False):
    """Lay a snapshot's config down as a template on a (typically new) system.

    Defaults to a dry run: reports what would be copied, what is
    intentionally skipped (station/region-specific artifacts that
    bbox_station_service.py should generate for the new deployment), and
    what needs manual editing - without writing anything.  Pass
    execute=True to actually copy; a pre-apply backup snapshot of the
    target is always taken first so the change is reversible.
    """
    if not os.path.isdir(os.path.join(snapshot_dir, 'etc')):
        raise SystemExit(f'Not a snapshot directory (no etc/ found): {snapshot_dir}')

    manifest_path = os.path.join(snapshot_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        src_ver = manifest.get('seiscomp_version', {}).get('Framework', '?')
        dst_ver = _seiscomp_version().get('Framework', '?')
        print(f'Source snapshot: {manifest.get("hostname", "?")} (seiscomp {src_ver})')
        print(f'Target system:   {socket.gethostname()} (seiscomp {dst_ver})')
        if src_ver != '?' and dst_ver != '?' and src_ver != dst_ver:
            print(f'  WARNING: SeisComP version mismatch ({src_ver} vs {dst_ver}) - '
                  f'config keys may differ; review carefully.')
        print()

    copy, exclude, review = classify_snapshot_for_apply(snapshot_dir)

    print(f'Plan for {seiscomp_root}:')
    print(f'  {len(copy)} file(s) to copy as-is (module/tuning templates)')
    print(f'  {len(review)} file(s) to copy but NEED MANUAL EDITING after apply:')
    for rel in review:
        print(f'    - {rel}')
    print(f'  {len(exclude)} file(s) intentionally SKIPPED (station/region-specific - '
          f'run bbox_station_service.py for the new deployment to (re)generate these):')
    for rel in exclude[:10]:
        print(f'    - {rel}')
    if len(exclude) > 10:
        print(f'    ...and {len(exclude) - 10} more')
    print()

    if not execute:
        print('Dry run only - no files were written.  Re-run with --yes to apply.')
        return {'copy': copy, 'exclude': exclude, 'review': review}

    backup_dir = os.path.join(
        'seiscomp_snapshots',
        f'pre-apply_{socket.gethostname()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    print(f'Backing up current target config to {backup_dir} before writing...')
    snapshot_system(backup_dir, seiscomp_root)
    print()

    for rel in copy + review:
        src = os.path.join(snapshot_dir, rel)
        dst = os.path.join(seiscomp_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    print(f'Applied {len(copy) + len(review)} file(s) to {seiscomp_root}.')
    print()
    print('Next steps:')
    print('  1. Fix the flagged files above (real DB password in scmaster.cfg, any')
    print('     machine-specific absolute paths such as scfakequake model/repo paths).')
    print("  2. Run bbox_station_service.py against the new deployment's bounding box to")
    print('     populate etc/key/station_*, etc/key/seedlink/profile_*, etc/inventory/,')
    print('     and share/scautoloc/station-locations.conf for the new stations.')
    print('  3. seiscomp update-config && seiscomp restart')
    print('  4. python examples/status_check.py check   # verify the new system is healthy')
    return {'copy': copy, 'exclude': exclude, 'review': review, 'backup': backup_dir}


# main report -------------------------------------------------------------

def run_check():
    print('SeisComP quick health check')
    print('SEISCOMP_ROOT =', SEISCOMP_ROOT)
    print()

    running, stopped, raw = status_modules()
    print('Running modules:', ', '.join(running))
    print('Stopped/disabled modules (not shown):', ', '.join(stopped))
    print()

    recent, old = check_slarchive_activity()
    if recent is None:
        print('slarchive state directory not found; is slarchive configured?')
    else:
        print(f'slarchive has {len(recent)} stations with activity <10 min old')
        if old:
            print(f'stations with stale state (>10min): {len(old)} sample: {old[:5]}')
    print()

    bindprobs = check_bindings()
    if bindprobs:
        print('Binding problems:')
        for p in bindprobs:
            print('  -', p)
    else:
        print('Key files look sane (contain seedlink lines).')
    print()

    diffs = compare_global()
    if diffs is None:
        print('No backup global.cfg to compare to (~ /proc_backup missing)')
    elif diffs:
        print('global.cfg differs from reference:')
        for k,a,b in diffs:
            print(f'  {k}: current={a!r} reference={b!r}')
        print('Consider merging essential defaults (recordstream, core.plugins, etc).')
    else:
        print('global.cfg matches reference defaults for key keys.')
    print()

    print('--- global.cfg pipeline requirements ---')
    global_issues, global_infos = check_global_cfg()
    for info in global_infos:
        print('  INFO:', info)
    if global_issues:
        for p in global_issues:
            print('  ISSUE:', p)
    else:
        print('  global.cfg has required pipeline keys.')
    print()

    sc_enabled, sc_out = check_scautopick_enabled()
    if sc_enabled:
        print('scautopick module is enabled.')
        # see if debug mode reports missing station configuration
        rc_dbg, dbg_out = run_scautopick_debug()
        if 'No stations added' in dbg_out:
            print('  WARNING: scautopick reports "No stations added (empty module configuration?)"')
            print('  This means the module has no streams configured and will not pick.')
        # also inspect the configuration file for missing stream entries
        cfg_issues = check_scautopick_cfg()
        for issue in cfg_issues:
            print('  CONFIG ISSUE:', issue)
    else:
        print('scautopick NOT enabled.  run: seiscomp enable scautopick && seiscomp start scautopick')
    print()

    print('--- scrttv stream configuration ---')
    scrttv_issues = check_scrttv_cfg()
    for msg in scrttv_issues:
        prefix = '' if msg.startswith('INFO') else 'ISSUE: '
        print(f'  {prefix}{msg}')
    if not any(m for m in scrttv_issues if not m.startswith('INFO')):
        print('  scrttv streams.codes looks configured.')
    print()

    print('--- station key file validity ---')
    key_issues = check_key_file_validity()
    if key_issues:
        for k in key_issues:
            print(' ', k)
        print('  Fix: re-apply bbox bindings so the service rewrites clean key files.')
    else:
        print('  No invalid binding lines found in station key files.')
    print()

    print('--- event processing pipeline ---')
    pipeline_issues = check_event_pipeline(running)
    if pipeline_issues:
        for p in pipeline_issues:
            print('  ISSUE:', p)
        print('  Fix:  seiscomp start scdb scamp scevent scmag sceasyquake')
    else:
        print('  Full pipeline running: scdb → sceasyquake → scautoloc → scevent → scmag')
    print()

    print('--- recent pick and amplitude activity ---')
    pick_issues, pick_infos = check_recent_picks_amplitudes(minutes=10)
    for info in pick_infos:
        print('  INFO:', info)
    if pick_issues:
        for p in pick_issues:
            print('  ISSUE:', p)
    else:
        print('  Pick-amplitude flow healthy.')
    print()

    print('--- picks and events (last 2 hours) ---')
    pa = check_picks_and_events(hours=2)
    if pa['pick_error']:
        print('  ISSUE: pick query failed:', pa['pick_error'])
    else:
        print(f'  Picks (by arrival time): {pa["pick_count"]}')
    if pa['event_error']:
        print('  ISSUE: event query failed:', pa['event_error'])
    elif not pa['events']:
        print('  Events: none in last 2 hours')
    else:
        print(f'  Events: {len(pa["events"])} in last 2 hours')
        for ev in pa['events']:
            mag_str = (f'M{ev["mag"]:.1f} {ev["mag_type"]}'.strip()
                       if ev['mag'] is not None else 'M?')
            depth_str = (f'{ev["depth_km"]:.1f} km'
                         if ev['depth_km'] is not None else 'depth?')
            phases_str = (f'{ev["phases"]} phases'
                          if ev['phases'] is not None else '')
            print(f'    {ev["time"]}  {mag_str}  '
                  f'lat {ev["lat"]:7.3f}  lon {ev["lon"]:8.3f}  '
                  f'{depth_str}  {phases_str}  [{ev["event_id"]}]')
    print()

    print('--- pick latency ---')
    lat = check_pick_latency(n=50)
    if lat.get('error'):
        print('  ISSUE:', lat['error'])
    else:
        age = lat['age_now_s']
        age_str = f'{age:.1f} s' if age is not None else '?'
        age_warn = ' *** STALE' if age is not None and age > 120 else ''
        print(f'  Newest pick arrival:  {lat["newest_pick_time"]} UTC  (age: {age_str}{age_warn})')
        print(f'  Processing latency ({lat["count"]} picks):')
        print(f'    min  {lat["lat_min"]:6.1f} s')
        print(f'    med  {lat["lat_med"]:6.1f} s')
        print(f'    mean {lat["lat_mean"]:6.1f} s')
        print(f'    max  {lat["lat_max"]:6.1f} s')
        if age is not None and age > 120:
            print('  ISSUE: latest pick is >2 min old – waveform delivery or '
                  'sceasyquake may have stalled.  Check seedlink and '
                  'seiscomp status sceasyquake.')
        elif lat['lat_med'] > 60:
            print('  WARN: median processing latency >60 s – consider reducing '
                  'picker.step_seconds in sceasyquake.cfg or checking GPU load.')
    print()

    print('--- scautoloc configuration ---')
    autoloc_issues, autoloc_infos = check_scautoloc_config()
    for info in autoloc_infos:
        print('  INFO:', info)
    if autoloc_issues:
        for p in autoloc_issues:
            print('  ISSUE:', p)
        print()
        print('  Fix: ensure etc/scautoloc.cfg contains:')
        print('    autoloc.stationLocations = @DATADIR@/scautoloc/station-locations.conf')
        print('    autoloc.locator.profile  = iasp91')
        print('    autoloc.networkType      = local')
        print('    autoloc.minPhaseCount    = 4')
        print('  Regenerate station-locations.conf from DB:')
        print('    mysql -u sysop -psysop seiscomp -e \'SELECT n.code, s.code, '
              's.latitude, s.longitude, COALESCE(s.elevation,0) FROM Station s '
              'JOIN Network n ON n._oid=s._parent_oid WHERE s.latitude IS NOT NULL '
              'ORDER BY n.code, s.code;\' | awk \'NR>1{printf '
              '"%-6s %-10s %10.4f %10.4f %7.1f\\n",$1,$2,$3,$4,$5}\' > '
              '$SEISCOMP_ROOT/share/scautoloc/station-locations.conf')
        print('  Then restart: seiscomp restart scautoloc')
    else:
        print('  scautoloc configuration looks correct.')
    print()

    print('Recommendations:')
    print('- Ensure seedlink and slarchive are running (seiscomp status).')
    print('- If slarchive activity is stale, check network and bindings.  It may be')
    print('  useful to run `slarchive -Fi -S "NET_STA" localhost:18000` for a few')
    print('  representative stations to observe errors.  The detailed output is')
    print('  retained in this directory:',
          os.path.join(SEISCOMP_ROOT,'var','lib','slarchive'))
    print('- Make sure `core.plugins` includes dbmysql (or your DB driver) and that')
    print('  the database connection strings in scm and global.cfg are valid.')
    print('- If you want automatic picking/association, enable scautopick and')
    print('  configure its filters in etc/defaults/scautopick.cfg (see proc_backup copy')
    print('  for example).  Also enable an associator module such as scautoloc or a')
    print('  simple external program and add `scautopick` to the list of modules')
    print('  started by the queue.  Here is a minimal command to fix missing modules:')
    print('      seiscomp enable scautopick')
    print('      seiscomp enable scautoloc   # or scautopick2 etc')
    print('      seiscomp update-config')
    print('      seiscomp restart scautopick scautoloc')


def build_argparser():
    p = argparse.ArgumentParser(
        description='SeisComP health check, config snapshot, and snapshot diff tool.')
    sub = p.add_subparsers(dest='command')

    sub.add_parser('check', help='run the live health check (default)')

    snap = sub.add_parser('snapshot', help='archive etc/ config, module status, '
                                            'and scfakequake/sceasyquake details')
    snap.add_argument('-o', '--output', default=None,
                       help='snapshot directory (default: seiscomp_snapshots/<host>_<timestamp>)')

    cmp_p = sub.add_parser('compare', help='diff two snapshot directories')
    cmp_p.add_argument('snapshot_a')
    cmp_p.add_argument('snapshot_b')
    cmp_p.add_argument('-o', '--output', default=None,
                        help='write the diff report to a file instead of stdout')

    app = sub.add_parser('apply', help="lay a snapshot's config down as a template on "
                                        "this (typically new) system; dry-run by default")
    app.add_argument('snapshot_dir')
    app.add_argument('-y', '--yes', action='store_true',
                      help='actually write the files (default only previews the plan)')

    return p


def main():
    args = build_argparser().parse_args()
    command = args.command or 'check'
    if command == 'snapshot':
        snapshot_system(args.output)
    elif command == 'compare':
        compare_snapshots(args.snapshot_a, args.snapshot_b, args.output)
    elif command == 'apply':
        apply_snapshot(args.snapshot_dir, execute=args.yes)
    else:
        run_check()


if __name__ == '__main__':
    main()
