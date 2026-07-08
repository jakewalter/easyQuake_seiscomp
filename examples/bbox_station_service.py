"""Simple Flask service to select a geographic box and query FDSN/Seedlink,
import matching stations into SeisComP, and automatically apply Seedlink chain
bindings so they start streaming immediately.

Usage:
    pip install flask obspy
    python examples/bbox_station_service.py

Open http://localhost:5000 in a browser, draw a rectangle on the basemap.
The server will:
  1. Query IRIS FDSN for stations in the box active in the last 24 h.
  2. Check which are available on the primary/secondary Seedlink servers.
  3. Write StationXML to $SEISCOMP_ROOT/etc/inventory/.
  4. Write etc/key/ binding files and Seedlink chain profiles.
  5. Run scinv sync + seiscomp update-config + seiscomp restart seedlink.
"""

import os

# strip Anaconda/Miniconda library paths before anything else loads
# SeisComP C++ extensions require a newer libstdc++; anaconda's older
# version frequently ends up earlier in LD_LIBRARY_PATH, so we remove it
if 'LD_LIBRARY_PATH' in os.environ:
    parts = os.environ['LD_LIBRARY_PATH'].split(':')
    parts = [p for p in parts if not ('anaconda' in p or 'miniconda' in p)]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(parts)

import subprocess
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory, Response
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.seedlink import Client as SeedlinkClient
from obspy import read_inventory
import logging
import json
import re
# SeisComP/ObsPy SeedLink client is chatty when connections time out;
# silence the low‑level connection logger so our UI isn’t filled with
# ``socket read error`` lines.  also quiet the broader seedlink package so
# ``terminating collect loop`` messages don't spam stdout during checks.
logging.getLogger('obspy.clients.seedlink.client.seedlinkconnection').setLevel(logging.CRITICAL)
logging.getLogger('obspy.clients.seedlink').setLevel(logging.CRITICAL)

app = Flask(__name__)

# directory containing this script (for serving the HTML file)
_HERE = os.path.dirname(os.path.abspath(__file__))

# primary and secondary seedlink hosts (port 18000 default)
PRIMARY_SEEDLINK   = "rtserve.beg.utexas.edu"
SECONDARY_SEEDLINK = "rtserve.iris.washington.edu"
# number of days to retain archived waveform packets; per-station key files
# also contain this value as a ``keep`` setting so individual stations could
# override the global default if needed.
ARCHIVE_KEEP_DAYS   = 7

# Channel band priority order used when selecting a single representative band
# per station for scrttv, scautopick, and SeedLink selectors.
# For scrttv only the Z component is written (e.g. 'HHZ'); scautopick still
# uses all components via '?' so it can detect teleseismic P on horizontals.
CHANNEL_PRIORITY = ('HH', 'EH', 'BH', 'SH', 'CH')

# Networks to exclude from selection and to purge from SeisComP entirely.
# The SY network is a SeisComP synthetic/internal network that should never
# be imported as real streaming stations.
EXCLUDED_NETWORKS = frozenset({'SY'})

# FDSN client (IRIS covers most public networks)
FDSN = FDSNClient("IRIS")

# SeisComP root – auto-detected from environment, falls back to common path
SEISCOMP_ROOT = os.environ.get("SEISCOMP_ROOT", "/home/jwalter/seiscomp")
KEY_DIR        = os.path.join(SEISCOMP_ROOT, "etc", "key")
INV_DIR        = os.path.join(SEISCOMP_ROOT, "etc", "inventory")

def _get_root():
    """Return the current SeisComp root, honouring the environment.

    Many helper functions used to reference the module-level constant which
    was fixed at import time.  This helper will re-read ``SEISCOMP_ROOT`` from
    the environment so tests that modify the variable after import still work.
    """
    return os.environ.get('SEISCOMP_ROOT', SEISCOMP_ROOT)


def _db_credentials():
    """Return (user, password, host, database) parsed from scmaster.cfg, or None."""
    url = _get_db_url()  # e.g. "mysql://sysop:sysop@localhost/seiscomp"
    if not url:
        return None
    m = re.match(r'(?:mysql://)?([^:]+):([^@]+)@([^/]+)/(.+)', url)
    return m.groups() if m else None


def _get_db_url():
    """Return a mysql:// URL from scmaster.cfg for use with ``scinv -d``.

    Reads the ``queues.production.processors.messages.dbstore.write`` entry
    from ``$SEISCOMP_ROOT/etc/scmaster.cfg`` and prepends the ``mysql://``
    driver prefix.  Returns an empty string if the file is not readable.
    """
    cfg_path = os.path.join(_get_root(), 'etc', 'scmaster.cfg')
    try:
        with open(cfg_path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('queues.production.processors.messages.dbstore.write'):
                    val = line.split('=', 1)[1].strip()
                    if not val.startswith('mysql://'):
                        val = 'mysql://' + val
                    return val
    except OSError:
        pass
    return ''


def _sync_inventory(fdsn_xml_path):
    """Convert a FDSN StationXML file to SeisComP native XML and sync to DB.

    SeisComP's ``scinv`` tool only understands its own XML format, not FDSN
    StationXML.  This helper runs ``fdsnxml2inv`` to produce a native inventory
    file (``stations_bbox_sc.xml`` beside the source), then calls
    ``scinv sync -d <db_url>`` to push it into the SeisComP database so that
    scrttv and other modules can find the stations.

    Returns a list of result dicts (one per subprocess call) to be appended to
    the apply results stream.
    """
    results = []
    root = _get_root()
    inv_dir = os.path.join(root, 'etc', 'inventory')
    sc_xml_path = os.path.join(inv_dir, 'stations_bbox_sc.xml')

    # Step A: convert FDSN StationXML → SeisComP native XML
    conv_result = _run(['fdsnxml2inv', fdsn_xml_path, sc_xml_path])
    results.append(dict(step='fdsnxml2inv', **conv_result))
    if conv_result.get('rc', 1) != 0:
        results[-1]['warning'] = 'fdsnxml2inv failed; inventory NOT synced'
        return results

    # Step B: sync the converted SeisComP XML to the database
    db_url = _get_db_url()
    if db_url:
        sync_cmd = ['seiscomp', 'exec', 'scinv', 'sync', '-d', db_url, sc_xml_path]
    else:
        # fall back to messaging-based sync (requires scmaster)
        sync_cmd = ['scinv', 'sync', sc_xml_path]
    sync_result = _run(sync_cmd, timeout=180)
    results.append(dict(step='scinv_sync', **sync_result))
    return results


def fix_existing_key_files(dry_run=False):
    """Strip invalid parameter lines from all station key files under etc/key/.

    SeisComP station key files must contain only module binding names
    (e.g. ``global``, ``seedlink:chain_primary``, ``slarchive``).  Earlier
    versions of this service erroneously wrote ``sources.chain.selectors``
    and ``keep = N`` inline parameter lines which are not valid in a station
    key file and cause ``seiscomp update-config`` to print "invalid binding"
    errors.

    This function scans every ``station_*`` file in etc/key/ and rewrites any
    that contain such lines.  It is safe to call multiple times.  Returns a
    dict with ``fixed`` (list of filenames changed) and ``skipped`` counts.
    """
    # Lines beginning with these prefixes are NOT valid in station key files
    INVALID_PREFIXES = ('sources.chain.selectors', 'keep =', 'keep=')

    root = _get_root()
    keydir = os.path.join(root, 'etc', 'key')
    fixed, total = [], 0
    if not os.path.isdir(keydir):
        return {'fixed': fixed, 'total': total, 'keydir_missing': True}

    for fn in sorted(os.listdir(keydir)):
        if not fn.startswith('station_'):
            continue
        path = os.path.join(keydir, fn)
        total += 1
        try:
            with open(path) as fh:
                lines = fh.readlines()
            clean = [l for l in lines
                     if not any(l.strip().startswith(p) for p in INVALID_PREFIXES)]
            if len(clean) != len(lines):
                if not dry_run:
                    with open(path, 'w') as fh:
                        fh.writelines(clean)
                fixed.append(fn)
        except OSError:
            pass

    return {'fixed': fixed, 'fixed_count': len(fixed), 'total': total,
            'dry_run': dry_run}


def remove_network_from_seiscomp(network_code):
    """Remove all traces of *network_code* from the live SeisComP installation.

    Steps performed:

    1. Delete ``etc/key/station_{NET}_*`` binding files.
    2. Remove ``{NET}.*`` entries from ``streams.codes`` in
       ``etc/scrttv.cfg``, ``etc/sceasyquake.cfg`` and
       ``etc/defaults/scautopick.cfg``.
    3. Filter the network out of both the FDSN StationXML inventory
       (``etc/inventory/stations_bbox.xml``) and the SeisComP native XML
       (``etc/inventory/stations_bbox_sc.xml``).
    4. Delete the network and its stations/channels from the SeisComP
       MySQL database directly so scolv and scautoloc stop seeing them.
    5. Rebuild ``share/scautoloc/station-locations.conf`` from the cleaned DB.
    6. Re-run ``seiscomp update-config`` for seedlink, slarchive, and scrttv.
    7. Restart seedlink, scautoloc, and sceasyquake.

    Returns a dict with ``network`` and ``steps`` (list of per-step results).
    """
    import shutil

    root = _get_root()
    steps = []

    # ------------------------------------------------------------------
    # 1. Remove station key files
    # ------------------------------------------------------------------
    keydir = os.path.join(root, 'etc', 'key')
    prefix = f"station_{network_code}_"
    removed_keys = []
    if os.path.isdir(keydir):
        for fn in os.listdir(keydir):
            if fn.startswith(prefix):
                try:
                    os.remove(os.path.join(keydir, fn))
                    removed_keys.append(fn)
                except OSError as exc:
                    steps.append({'step': 'remove_key', 'file': fn, 'error': str(exc)})
    steps.append({'step': 'remove_key_files',
                  'count': len(removed_keys), 'files': removed_keys})

    # ------------------------------------------------------------------
    # 2. Purge network entries from module config files
    # ------------------------------------------------------------------
    def _filter_cfg(cfg_path, key):
        """Remove all entries starting with `network_code.` from a comma-separated config key."""
        try:
            cfg = {}
            if os.path.exists(cfg_path):
                with open(cfg_path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        k, v = line.split('=', 1)
                        cfg[k.strip()] = v.strip()
            entries = [p.strip() for p in cfg.get(key, '').split(',') if p.strip()]
            before = len(entries)
            entries = [e for e in entries
                       if not e.startswith(f'{network_code}.')]
            cfg[key] = ', '.join(entries)
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, 'w') as fh:
                for k, v in cfg.items():
                    fh.write(f"{k} = {v}\n")
            return {'path': cfg_path, 'removed_entries': before - len(entries)}
        except OSError as exc:
            return {'path': cfg_path, 'error': str(exc)}

    steps.append({'step': 'update_scrttv_cfg',
                  **_filter_cfg(os.path.join(root, 'etc', 'scrttv.cfg'), 'streams.codes')})
    steps.append({'step': 'update_sceasyquake_cfg',
                  **_filter_cfg(os.path.join(root, 'etc', 'sceasyquake.cfg'), 'streams.codes')})
    steps.append({'step': 'update_scautopick_cfg',
                  **_filter_cfg(os.path.join(root, 'etc', 'defaults', 'scautopick.cfg'), 'streams')})

    # ------------------------------------------------------------------
    # 3. Filter network from FDSN and SeisComP inventory XML files
    # ------------------------------------------------------------------
    fdsn_xml = os.path.join(root, 'etc', 'inventory', 'stations_bbox.xml')
    if os.path.exists(fdsn_xml):
        try:
            inv = read_inventory(fdsn_xml)
            before = len(inv.networks)
            inv.networks = [n for n in inv.networks if n.code != network_code]
            removed = before - len(inv.networks)
            buf = BytesIO()
            inv.write(buf, format='STATIONXML')
            buf.seek(0)
            with open(fdsn_xml, 'wb') as fh:
                fh.write(buf.read())
            steps.append({'step': 'filter_fdsn_xml', 'path': fdsn_xml,
                          'removed_networks': removed})
        except Exception as exc:
            # File may be a stub or SeisComP XML rather than FDSN StationXML;
            # log but continue – the DB deletion above already removed the network.
            steps.append({'step': 'filter_fdsn_xml', 'path': fdsn_xml,
                          'skipped': True, 'reason': str(exc)})

    sc_xml = os.path.join(root, 'etc', 'inventory', 'stations_bbox_sc.xml')
    if os.path.exists(sc_xml):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(sc_xml)
            rt = tree.getroot()
            ns_prefix = rt.tag.split('}')[0].lstrip('{') if '}' in rt.tag else ''
            tag = (f'{{{ns_prefix}}}' if ns_prefix else '') + 'network'
            removed_nets = 0
            for parent in list(rt.iter()):
                for child in list(parent):
                    if child.tag == tag and child.get('code') == network_code:
                        parent.remove(child)
                        removed_nets += 1
            shutil.copy2(sc_xml, sc_xml + '.bak')
            ET.indent(tree, space='  ')
            tree.write(sc_xml, xml_declaration=True, encoding='unicode')
            steps.append({'step': 'filter_sc_xml', 'path': sc_xml,
                          'removed_networks': removed_nets})
        except Exception as exc:
            steps.append({'step': 'filter_sc_xml', 'error': str(exc)})

    # ------------------------------------------------------------------
    # 4. Delete from MySQL database
    # ------------------------------------------------------------------
    # SeisComP inventory hierarchy: Network → Station → SensorLocation → Stream
    # We delete bottom-up to respect foreign-key constraints, then clean up
    # orphaned PublicObject registry rows.
    db_sql = f"""
 SET FOREIGN_KEY_CHECKS = 0;
 DELETE s FROM Stream s
   JOIN SensorLocation sl ON sl._oid = s._parent_oid
   JOIN Station st ON st._oid = sl._parent_oid
   JOIN Network nt ON nt._oid = st._parent_oid
   WHERE nt.code = '{network_code}';
 DELETE sl FROM SensorLocation sl
   JOIN Station st ON st._oid = sl._parent_oid
   JOIN Network nt ON nt._oid = st._parent_oid
   WHERE nt.code = '{network_code}';
 DELETE st FROM Station st
   JOIN Network nt ON nt._oid = st._parent_oid
   WHERE nt.code = '{network_code}';
 DELETE FROM Network WHERE code = '{network_code}';
 SET FOREIGN_KEY_CHECKS = 1;
"""
    creds = _db_credentials()
    if creds:
        user, pw, host, db = creds
        db_result = _run(
            ['mysql', f'-u{user}', f'-p{pw}', '-h', host, db,
             '-e', db_sql],
            timeout=60,
        )
        steps.append({'step': 'db_delete', 'network': network_code,
                      'rc': db_result.get('rc'),
                      'output': (db_result.get('stdout', '') + db_result.get('stderr', '')).strip()[-300:]})
    else:
        steps.append({'step': 'db_delete', 'error': 'could not read DB credentials'})

    # ------------------------------------------------------------------
    # 5. Rebuild station-locations.conf now that DB is clean
    # ------------------------------------------------------------------
    try:
        cfg_result = _ensure_scautoloc_cfg()
        steps.append({'step': 'ensure_scautoloc_cfg', **cfg_result})
    except Exception as exc:
        steps.append({'step': 'ensure_scautoloc_cfg', 'error': str(exc)})
    try:
        loc_result = _update_scautoloc_station_locations()
        steps.append({'step': 'update_station_locations', **loc_result})
    except Exception as exc:
        steps.append({'step': 'update_station_locations', 'error': str(exc)})

    # ------------------------------------------------------------------
    # 6/7. Rebuild configs and restart
    # ------------------------------------------------------------------
    steps.append({'step': 'update_config_seedlink',
                  **_run(['seiscomp', 'update-config', 'seedlink'])})
    steps.append({'step': 'update_config_slarchive',
                  **_run(['seiscomp', 'update-config', 'slarchive'])})
    steps.append({'step': 'update_config_scrttv',
                  **_run(['seiscomp', 'update-config', 'scrttv'])})
    steps.append({'step': 'restart_seedlink',
                  **_run(['seiscomp', 'restart', 'seedlink'])})
    steps.append({'step': 'restart_scautoloc',
                  **_run(['seiscomp', 'restart', 'scautoloc'])})
    steps.append({'step': 'restart_sceasyquake',
                  **_run(['seiscomp', 'restart', 'sceasyquake'])})

    return {'network': network_code, 'steps': steps}


@app.route('/admin/remove-network', methods=['POST'])
def admin_remove_network():
    """POST to remove all SeisComP configuration and inventory for a network.

    Body: ``{"network": "SY"}``

    Performs: key-file removal, config purge, XML inventory filtering,
    direct DB deletion, station-locations.conf rebuild, and module restarts.
    """
    payload = request.get_json(force=True, silent=True) or {}
    network = payload.get('network', '').strip().upper()
    if not network:
        return jsonify({'error': '"network" field is required'}), 400
    result = remove_network_from_seiscomp(network)
    return jsonify(result)


@app.route('/admin/fix-keys', methods=['POST'])
def admin_fix_keys():
    """POST to strip invalid binding lines from all existing station key files
    and re-run ``seiscomp update-config`` for affected modules.

    Optional JSON body: ``{"dry_run": true}`` to preview without writing.
    """
    payload = request.get_json(force=True, silent=True) or {}
    dry_run = bool(payload.get('dry_run', False))
    result = fix_existing_key_files(dry_run=dry_run)
    if not dry_run and result.get('fixed_count', 0) > 0:
        # rebuild configs now that key files are clean
        result['update_slarchive'] = _run(['seiscomp', 'update-config', 'slarchive'])
        result['update_seedlink']  = _run(['seiscomp', 'update-config', 'seedlink'])
        result['update_scrttv']    = _run(['seiscomp', 'update-config', 'scrttv'])
    return jsonify(result)


def fix_scrttv_channel_format(dry_run=False):
    """Migrate ``streams.codes`` entries in ``etc/scrttv.cfg`` from 3-part
    ``NET.STA.CHA?`` to the correct 4-part ``NET.STA.*.CHA?`` format that
    SeisComP scrttv requires (Network, Station, Location-wildcard, Channel).

    Without a location component SeisComP cannot match streams that carry a
    location code (e.g. ``4O.AT01.00.HHE``), so scrttv displays nothing even
    though the seedlink server is delivering data.

    Returns a dict with ``migrated`` count and ``total`` entry count.
    """
    cfg_path = os.path.join(_get_root(), 'etc', 'scrttv.cfg')
    if not os.path.exists(cfg_path):
        return {'error': 'scrttv.cfg not found', 'cfg_path': cfg_path}

    cfg = {}
    with open(cfg_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip()

    raw = [p.strip() for p in cfg.get('streams.codes', '').split(',') if p.strip()]
    migrated_count = 0
    new_entries = []
    for entry in raw:
        parts = entry.split('.')
        if len(parts) == 3:
            # 3-part: NET.STA.CHA? → 4-part: NET.STA.*.CHA?
            entry = f"{parts[0]}.{parts[1]}.*.{parts[2]}"
            migrated_count += 1
        new_entries.append(entry)

    # Collapse multiple band patterns per station down to a single best one.
    # Group by NET.STA key; keep the entry whose band appears earliest in
    # CHANNEL_PRIORITY, falling back to alphabetical order.
    seen = {}   # "NET.STA" -> (priority_rank, entry)
    collapsed_count = 0
    final_entries = []
    for entry in new_entries:
        parts = entry.split('.')
        if len(parts) == 4:
            net, sta, _loc, band = parts
            key = f"{net}.{sta}"
            band_prefix = band[:2]
            try:
                rank = CHANNEL_PRIORITY.index(band_prefix)
            except ValueError:
                rank = len(CHANNEL_PRIORITY)
            if key not in seen or rank < seen[key][0]:
                if key in seen:
                    collapsed_count += 1
                    # Replace the previously appended entry with this better one
                    final_entries = [e for e in final_entries
                                     if not e.startswith(f"{key}.")]
                seen[key] = (rank, entry)
                final_entries.append(entry)
            else:
                collapsed_count += 1  # discarded lower-priority duplicate
        else:
            final_entries.append(entry)

    # Ensure only the Z component is kept for display in scrttv
    # (convert trailing '?' wildcard to 'Z', e.g. HH? → HHZ)
    final_entries = [
        e[:-1] + 'Z' if e.endswith('?') else e
        for e in final_entries
    ]

    cfg['streams.codes'] = ', '.join(final_entries)
    if not dry_run:
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        with open(cfg_path, 'w') as fh:
            for k, v in cfg.items():
                fh.write(f"{k} = {v}\n")

    return {
        'total': len(raw),
        'migrated': migrated_count,
        'collapsed': collapsed_count,
        'dry_run': dry_run,
        'cfg_path': cfg_path,
    }


@app.route('/admin/fix-scrttv-cfg', methods=['POST'])
def admin_fix_scrttv_cfg():
    """POST to migrate streams.codes from 3-part to 4-part NSLC format and
    re-run ``seiscomp update-config scrttv``.

    Optional JSON body: ``{"dry_run": true}`` to preview without writing.
    """
    payload = request.get_json(force=True, silent=True) or {}
    dry_run = bool(payload.get('dry_run', False))
    result = fix_scrttv_channel_format(dry_run=dry_run)
    if not dry_run:
        result['update_scrttv'] = _run(['seiscomp', 'update-config', 'scrttv'])
    return jsonify(result)



# ---------------------------------------------------------------------------
# SeisComP integration helpers
# ---------------------------------------------------------------------------

def load_existing_stations():
    """Return set of "NET.STA" codes already present in SeisComP inventory.

    Reads all XML files under $SEISCOMP_ROOT/etc/inventory.  If parsing fails
    for a file it is ignored.
    """
    existing = set()
    inv_dir = os.path.join(_get_root(), 'etc', 'inventory')
    if os.path.isdir(inv_dir):
        for fn in os.listdir(inv_dir):
            if not fn.lower().endswith('.xml'):
                continue
            path = os.path.join(inv_dir, fn)
            try:
                inv = read_inventory(path)
            except Exception:
                continue
            for net in inv:
                for sta in net:
                    existing.add(f"{net.code}.{sta.code}")
    return existing


def _run(cmd, timeout=120):
    """Run a subprocess and return a compact result dict.

    SeisComP binaries link against the system libstdc++.  On machines with
    anaconda in ``LD_LIBRARY_PATH`` the wrong (older) C++ runtime may be
    picked up, leading to ``GLIBCXX_3.4.30 not found`` errors.  Clear the
    conda paths from the environment when executing any SeisComP command so
    that the system-provided libraries are used instead.
    """
    env = os.environ.copy()
    # remove anaconda/lib from LD_LIBRARY_PATH if present
    if 'LD_LIBRARY_PATH' in env:
        parts = env['LD_LIBRARY_PATH'].split(':')
        cleaned = [p for p in parts if 'anaconda' not in p and 'miniconda' not in p]
        env['LD_LIBRARY_PATH'] = ':'.join(cleaned)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return {"cmd": " ".join(cmd), "rc": r.returncode,
                "stdout": r.stdout[-800:], "stderr": r.stderr[-800:]}
    except Exception as exc:
        return {"cmd": " ".join(cmd), "rc": -1, "error": str(exc)}


def _update_scrttv_channels(stations):
    """Ensure ``$SEISCOMP_ROOT/etc/scrttv.cfg`` has exactly one channel
    pattern per station, using the highest-priority available band
    (HH > EH > BH > SH > CH).  Entries for stations in the input list are
    replaced; all other existing entries are preserved (additive for new
    stations, corrective for ones whose band may have changed).
    """
    cfg_path = os.path.join(_get_root(), 'etc', 'scrttv.cfg')
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.strip()
    raw = [p.strip() for p in cfg.get('streams.codes', '').split(',') if p.strip()]
    # Migrate any legacy 3-part entries (NET.STA.CHA?) to 4-part (NET.STA.*.CHA?)
    entries = []
    for entry in raw:
        parts = entry.split('.')
        if len(parts) == 3:
            entry = f"{parts[0]}.{parts[1]}.*.{parts[2]}"
        entries.append(entry)
    for s in stations:
        net = s['network']; sta = s['station']
        # Remove all existing patterns for this station, then add a single
        # Z-component entry so scrttv shows one trace per station.
        prefix = f"{net}.{sta}."
        entries = [e for e in entries if not e.startswith(prefix)]
        chan = _best_z_channel_selector(s.get('channels', []))
        entries.append(f"{net}.{sta}.*.{chan}")
    cfg['streams.codes'] = ', '.join(entries)
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, 'w') as fh:
        for k, v in cfg.items():
            fh.write(f"{k} = {v}\n")


def _update_scautopick_config(stations):
    """Ensure ``etc/defaults/scautopick.cfg`` has a streams entry including
    the given stations.

    SeisComP's scautopick expects full stream selectors (network.station
    plus channel pattern), so we reuse the same channel-prefix logic used for
    the scrttv configuration.  For each station we generate up to three
    wildcarded-channel patterns (e.g. ``XX.YY.HH?``) and append them to the
    comma-separated ``streams`` value.  Existing entries are preserved and the
    function is safe to call repeatedly.
    """
    cfg_path = os.path.join(_get_root(), 'etc', 'defaults', 'scautopick.cfg')
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.strip()
    existing = [p.strip() for p in cfg.get('streams', '').split(',') if p.strip()]

    for s in stations:
        net = s.get('network')
        sta = s.get('station')
        # Remove all existing patterns for this station, then add exactly one
        # pattern for the highest-priority available band.
        prefix = f"{net}.{sta}."
        existing = [e for e in existing if not e.startswith(prefix)]
        pref = _best_channel_prefix(s.get('channels', []))
        existing.append(f"{net}.{sta}.{pref}")
    if existing:
        cfg['streams'] = ', '.join(existing)
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, 'w') as fh:
        for k, v in cfg.items():
            fh.write(f"{k} = {v}\n")


# Keys that must be present in etc/scautoloc.cfg for the pipeline to work.
# amplTypeAbs = snr is critical: scautoloc's hasAmplitude() checks pick->amp
# which is only set by amplTypeAbs-type amplitudes (default "mb").  Since
# sceasyquake publishes type="snr" amplitudes, pick->amp stays 0 unless
# amplTypeAbs is also "snr" – causing every pick to stall "waiting for
# amplitude" and zero origins to be produced.
_SCAUTOLOC_REQUIRED_KEYS = {
    'autoloc.amplTypeAbs':   'snr',
    'autoloc.amplTypeSNR':   'snr',
    'autoloc.minPickSNR':    '0',
    'locator.profile':       'iasp91',
    'autoloc.networkType':   'local',
    'autoloc.minPhaseCount': '4',
}


def _ensure_scautoloc_cfg():
    """Ensure ``$SEISCOMP_ROOT/etc/scautoloc.cfg`` contains all required keys.

    Reads the existing file (if any), adds missing keys, and rewrites it.
    Keys that are already present (any value) are left untouched so user
    customisations are preserved.  Only genuinely missing keys are appended.

    Returns a dict with ``added`` (list of "key = value" strings added),
    ``path`` (cfg file path) and optionally ``error``.
    """
    cfg_path = os.path.join(_get_root(), 'etc', 'scautoloc.cfg')
    existing: dict[str, str] = {}
    lines: list[str] = []

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as fh:
                lines = fh.readlines()
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and '=' in stripped:
                    k, _, v = stripped.partition('=')
                    existing[k.strip()] = v.strip()
        except Exception as exc:
            return {'path': cfg_path, 'error': str(exc), 'added': []}

    added = []
    for key, value in _SCAUTOLOC_REQUIRED_KEYS.items():
        if key not in existing:
            entry = f'{key} = {value}\n'
            lines.append(entry)
            added.append(entry.strip())

    if added:
        try:
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, 'w') as fh:
                fh.writelines(lines)
        except Exception as exc:
            return {'path': cfg_path, 'error': str(exc), 'added': added}

    return {'path': cfg_path, 'added': added}


def _update_scautoloc_station_locations():
    """Regenerate ``$SEISCOMP_ROOT/share/scautoloc/station-locations.conf``
    from the SeisComP inventory database.

    scautoloc looks up station coordinates from this flat file to form
    origins.  After any ``scinv sync`` that imports new stations this file
    must be rebuilt, otherwise scautoloc prints ``MISSING STATION`` and
    ignores picks from those stations.

    The file format is one station per line::

        NET  STA           LAT        LON       ELEV

    Returns a dict with ``written`` (path), ``count`` (stations written),
    and ``error`` (if any).
    """
    import urllib.parse

    out_path = os.path.join(_get_root(), 'share', 'scautoloc',
                            'station-locations.conf')
    bak_path = out_path + '.bak'

    db_url = _get_db_url()   # e.g. "mysql://sysop:sysop@localhost/seiscomp"
    if not db_url:
        return {'error': 'could not determine database URL', 'written': out_path}

    try:
        # Parse mysql://user:pass@host/db URL manually to avoid extra deps
        rest = db_url[len('mysql://'):]
        userinfo, hostdb = rest.split('@', 1)
        user, password = userinfo.split(':', 1)
        host_part, dbname = hostdb.split('/', 1)
        host = host_part

        import subprocess as _sp
        sql = (
            "SELECT n.code, st.code, st.latitude, st.longitude, "
            "COALESCE(st.elevation, 0.0) "
            "FROM Network n "
            "JOIN Station st ON st._parent_oid = n._oid "
            "WHERE st.latitude IS NOT NULL AND st.longitude IS NOT NULL "
            "ORDER BY n.code, st.code;"
        )
        proc = _sp.run(
            ['mysql', f'-u{user}', f'-p{password}', '-h', host,
             '--batch', '--skip-column-names', dbname],
            input=sql, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return {'error': proc.stderr.strip(), 'written': out_path}

        lines = []
        for row in proc.stdout.splitlines():
            parts = row.split('\t')
            if len(parts) < 5:
                continue
            net, sta, lat, lon, elev = parts
            try:
                lines.append(
                    f"{net:<6} {sta:<12}  {float(lat):>9.4f}  "
                    f"{float(lon):>10.4f}  {float(elev):>7.1f}\n"
                )
            except ValueError:
                pass

        if os.path.exists(out_path):
            import shutil
            shutil.copy2(out_path, bak_path)  # keep one backup

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as fh:
            fh.writelines(lines)

        return {'written': out_path, 'count': len(lines), 'backup': bak_path}

    except Exception as exc:
        return {'error': str(exc), 'written': out_path}


def _update_sceasyquake_config(stations):
    """Ensure ``$SEISCOMP_ROOT/etc/sceasyquake.cfg`` has ``streams.codes``
    entries for the given stations, using the same 4-part
    ``NET.STA.*.HH?`` format that sceasyquake expects.

    Existing entries for the given stations are replaced; all other
    existing entries are preserved.  The file is created if absent.
    This ensures sceasyquake will open a SeedLink subscription for the
    newly-imported stations automatically on its next restart.
    """
    cfg_path = os.path.join(_get_root(), 'etc', 'sceasyquake.cfg')
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.strip()
    existing = [p.strip() for p in cfg.get('streams.codes', '').split(',') if p.strip()]
    for s in stations:
        net = s.get('network')
        sta = s.get('station')
        # Remove any existing entry for this station and add a fresh one
        # using the best available channel band (HH > EH > BH > SH > CH).
        prefix = f"{net}.{sta}."
        existing = [e for e in existing if not e.startswith(prefix)]
        pref = _best_channel_prefix(s.get('channels', []))
        existing.append(f"{net}.{sta}.*.{pref}")
    if existing:
        cfg['streams.codes'] = ', '.join(existing)
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, 'w') as fh:
        for k, v in cfg.items():
            fh.write(f"{k} = {v}\n")


def _best_channel_prefix(chan_list):
    """Return the single best channel band prefix for a station.

    Picks the highest-priority band present in ``chan_list`` using the order
    defined by ``CHANNEL_PRIORITY`` (HH > EH > BH > SH > CH).  Returns a
    string like ``'HH?'`` (all 3 components).  Falls back to ``'HH?'`` when
    ``chan_list`` is empty or contains no recognised band.
    """
    present = set()
    for c in chan_list:
        if len(c) >= 2:
            present.add(c[:2])
    for band in CHANNEL_PRIORITY:
        if band in present:
            return band + '?'
    # No known band found – fall back
    if not present:
        return CHANNEL_PRIORITY[0] + '?'   # default HH?
    return sorted(present)[0] + '?'         # first alphabetically


def _best_z_channel_selector(chan_list):
    """Return the Z-component channel code for the best available band.

    Like ``_best_channel_prefix`` but returns only the vertical component
    (e.g. ``'HHZ'``, ``'EHZ'``).  Used for scrttv so the waveform viewer
    displays one trace per station instead of three.
    """
    prefix = _best_channel_prefix(chan_list)  # e.g. 'HH?'
    return prefix[:-1] + 'Z'                  # 'HHZ'


def _channel_selector_prefixes(chan_list):
    """Return unique 2-char band prefixes with '?' for every band in chan_list.

    Kept for backward compatibility.  New code should use
    ``_best_channel_prefix`` which returns the single highest-priority band.
    """
    prefs = []
    for c in chan_list:
        if len(c) >= 2:
            pref = c[:2] + '?'
            if pref not in prefs:
                prefs.append(pref)
    return prefs[:3]


def apply_seiscomp_bindings(stations, stationxml):
    """
    Given a list of station dicts (each having 'network', 'station', 'source')
    and the full StationXML string:

      * Write inventory to $SEISCOMP_ROOT/etc/inventory/stations_bbox.xml
      * Write $SEISCOMP_ROOT/etc/key/station_NET_STA  (global + seedlink chain + slarchive)
      * Write chain profile files for primary/secondary Seedlink servers
      * Run:  scinv sync  →  seiscomp update-config seedlink/slarchive  →  restart seedlink
      * Start scdb (database writer) and scamp (amplitude computation) if not running
      * Restart scautoloc so it reloads station inventory for the new stations
      * Restart sceasyquake so it opens SeedLink subscriptions for new streams
      * verify that a handful of recent samples can be pulled via the local SeedLink
        recordstream (this gives confidence that the binding actually started flow).

    Returns a list of result dicts (one per command).  In addition to the usual
    shell command results, a final dict with ``step"stream"`` may be appended
    giving a boolean "active" flag or a list of stations that successfully
    delivered data.
    """
    # Safety: never process stations from excluded networks regardless of caller
    stations = [s for s in stations if s.get('network') not in EXCLUDED_NETWORKS]

    results = []
    # refresh scrttv configuration so its channel filters include these stations
    try:
        _update_scrttv_channels(stations)
    except Exception:
        pass
    # also ensure scautopick streams list contains these stations so the picker
    # will actually see records for them
    try:
        _update_scautopick_config(stations)
    except Exception:
        pass
    # ensure sceasyquake streams.codes includes the new stations so the ML
    # picker opens SeedLink subscriptions for them on restart
    try:
        _update_sceasyquake_config(stations)
    except Exception:
        pass

    # 1. Persist inventory XML so scinv can pick it up
    root = _get_root()
    inv_dir = os.path.join(root, 'etc', 'inventory')
    os.makedirs(inv_dir, exist_ok=True)
    inv_path = os.path.join(inv_dir, "stations_bbox.xml")
    with open(inv_path, "w") as fh:
        fh.write(stationxml)
    results.append({"step": "write_inventory", "path": inv_path})

    # 2. Create key / seedlink subdirectory layout
    root = _get_root()
    sl_key_dir = os.path.join(root, "etc", "key", "seedlink")
    os.makedirs(sl_key_dir, exist_ok=True)

    # 3. Write chain profiles  (one per Seedlink source)
    #    SeisComP reads etc/key/seedlink/profile_<name>
    # create two station profiles that the seedlink module will read when
    # key files reference ``seedlink:profile_name``.  A profile must set the
    # ``sources`` parameter so that the chain plugin is activated, and it
    # should supply the concrete host/port values under the plugin namespace
    # (``sources.chain.address`` / ``sources.chain.port``).  Without the
    # ``sources`` line update-config will generate a configuration with no
    # plugins, which is what was causing seedlink to die with “no plugins
    # defined”.
    for profile_name, host in [("chain_primary",   PRIMARY_SEEDLINK),
                                ("chain_secondary", SECONDARY_SEEDLINK)]:
        profile_path = os.path.join(sl_key_dir, f"profile_{profile_name}")
        with open(profile_path, "w") as fh:
            fh.write("sources = chain\n")
            fh.write(f"sources.chain.address = {host}\n")
            fh.write("sources.chain.port = 18000\n")
        results.append({"step": "write_sl_profile", "path": profile_path})

    # 4. Write per-station key files
    #    Format: one module binding per line, e.g. "seedlink:chain_primary".  We
    #    also add a ``keep`` parameter so that slarchive will retain data for a
    #    limited number of days (set by ARCHIVE_KEEP_DAYS below).  Without a
    #    ``keep`` entry the default in slarchive is 30 days.
    written, skipped = [], []
    for sta_info in stations:
        net  = sta_info["network"]
        sta  = sta_info["station"]
        src  = sta_info["source"]          # 'primary' or 'secondary'
        sl_profile = f"chain_{src}"

        key_path = os.path.join(root, "etc", "key", f"station_{net}_{sta}")
        try:
            with open(key_path, "w") as fh:
                # Station key files contain only module binding names.
                # sources.chain.selectors does NOT belong here – it is invalid
                # and causes `seiscomp update-config slarchive` to report
                # "invalid binding" errors.  Channel selection for chain
                # happens inside the seedlink profile file and the chain XML
                # that `seiscomp update-config seedlink` generates.
                fh.write("global\n")
                fh.write(f"seedlink:{sl_profile}\n")
                fh.write("slarchive\n")
            written.append(f"{net}.{sta}")
        except Exception as exc:
            skipped.append({"station": f"{net}.{sta}", "error": str(exc)})

    results.append({"step": "write_key_files",
                    "written": len(written), "skipped": skipped})

    # 5. Convert FDSN StationXML → SeisComP native XML and sync to database.
    #    scinv only reads SeisComP XML, not FDSN StationXML; fdsnxml2inv is
    #    required to produce a compatible file before syncing.
    results.extend(_sync_inventory(inv_path))

    # 6. Rebuild module configs from key files
    results.append(_run(["seiscomp", "update-config", "seedlink"]))
    results.append(_run(["seiscomp", "update-config", "slarchive"]))
    # update-config scrttv regenerates the scrttv runtime binding so the GUI
    # picks up the new stations without needing a restart.
    results.append(_run(["seiscomp", "update-config", "scrttv"]))

    # after configuration has been generated, patch the chain XML files to
    # include explicit selectors so the upstream servers will actually send
    # data.  this avoids reliance on the SeisComP configuration parser.
    def _patch_chain():
        import glob, re
        # iterate over every chain file produced by update-config
        for fn in glob.glob(os.path.join(SEISCOMP_ROOT, 'var', 'lib', 'seedlink', 'chain*.xml')):
            try:
                data = open(fn).read().splitlines()
                out = []
                for line in data:
                    for s in stations:
                        code = f"{s['network']}.{s['station']}"
                        # use single best-priority band for the SeedLink selector
                        sel = _best_channel_prefix(s.get('channels', []))
                        if f'id="{code}"' in line and 'selectors="' in line and sel:
                            line = re.sub(r'selectors="[^"]*"', f'selectors="{sel}"', line)
                    out.append(line)
                open(fn,'w').write("\n".join(out))
            except Exception:
                pass
    try:
        _patch_chain()
    except Exception:
        pass

    # ensure the slarchive module is running (start if necessary) so that data
    # written to the SeedLink buffer will actually be archived.
    results.append(_run(["seiscomp", "start", "slarchive"]))

    # ensure scdb (database writer) and scamp (amplitude computation) are
    # running.  scdb must be up for picks/origins to be persisted so that
    # scolv and other modules can see them.
    results.append(_run(["seiscomp", "start", "scdb", "scamp"]))

    # 7. Restart Seedlink so it picks up the new chain entries
    results.append(_run(["seiscomp", "restart", "seedlink"]))

    # Restart scautoloc so it re-reads the station inventory from the database
    # and can form origins using any newly-imported stations.  Restart
    # sceasyquake so it opens SeedLink subscriptions for any new streams added
    # to streams.codes above.
    try:
        cfg_result = _ensure_scautoloc_cfg()
        results.append(dict(step='ensure_scautoloc_cfg', **cfg_result))
    except Exception as exc:
        results.append({'step': 'ensure_scautoloc_cfg', 'error': str(exc)})
    try:
        loc_result = _update_scautoloc_station_locations()
        results.append(dict(step='update_station_locations', **loc_result))
    except Exception as exc:
        results.append({'step': 'update_station_locations', 'error': str(exc)})
    results.append(_run(["seiscomp", "restart", "scautoloc"]))
    results.append(_run(["seiscomp", "restart", "sceasyquake"]))

    # perform a quick streaming test against the local seedlink instance
    # for each station; if data are available in the last few seconds we append
    # a result dictionary so the UI can display the streaming status.
    try:
        streaming = []
        import time
        # wait up to 5 seconds for seedlink to open port and accept connections
        deadline = time.time() + 5
        c = None
        while time.time() < deadline:
            try:
                # short timeout for the local seedlink connection as well
                c = SeedlinkClient("127.0.0.1", timeout=3)
                break
            except Exception:
                time.sleep(0.3)
        if c is None:
            raise RuntimeError("cannot connect to local SeedLink")

        t2 = UTCDateTime()
        t1 = t2 - 60
        for s in stations:
            # attempt to fetch a packet with retries
            got = False
            for _ in range(4):
                try:
                    st = c.get_waveforms(s['network'], s['station'], '', 'HH?', t1, t2)
                    if st and any(tr.stats.npts > 0 for tr in st):
                        streaming.append(f"{s['network']}.{s['station']}")
                        got = True
                        break
                except Exception as exc:
                    last_exc = exc
                time.sleep(0.5)
            if not got and 'last_exc' in locals():
                streaming.append({"station": f"{s['network']}.{s['station']}",
                                  "error": str(last_exc)})
        results.append({"step": "stream", "stations": streaming,
                        "window": f"{t1.isoformat()} - {t2.isoformat()}"})
    except Exception as exc:
        results.append({"step": "stream", "error": f"stream check failed: {exc}"})

    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory(_HERE, 'bbox_station_service.html')


@app.route('/search', methods=['POST'])
def search():
    """Receive a bounding box, query FDSN + Seedlink; return station list + StationXML."""
    payload = request.get_json(force=True)
    bbox = payload.get('bbox')
    if not bbox or len(bbox) != 4:
        return jsonify({'error': 'bad bbox'}), 400
    minlat, minlon, maxlat, maxlon = bbox

    # look back 24 h to focus on recently active stations
    t2 = UTCDateTime()
    t1 = t2 - 24 * 3600
    try:
        # request channels so we know exactly what streams exist for each station
        inventory = FDSN.get_stations(starttime=t1, endtime=t2,
                                       minlatitude=minlat,  maxlatitude=maxlat,
                                       minlongitude=minlon, maxlongitude=maxlon,
                                       level='channel')
    except Exception as exc:
        return jsonify({'error': f'FDSN query failed: {exc}'}), 500

    def seedlink_stations(host):
        # ask a remote seedlink server for the list of available stations.
        # previously we attempted a quick TCP connect to detect unreachable
        # hosts early, but this extra step is unnecessary; the ObsPy client
        # will raise a useful exception on failure and we simply treat that as
        # "no stations available".
        try:
            c = SeedlinkClient(host, timeout=5)
            info = c.get_info(level='station')
            return {f"{n}.{s}" for n, s in info}
        except Exception as exc:
            # log rather than raise; a failure simply means the remote server
            # provides no stations and we filter them out.
            print(f"seedlink {host} error: {exc}")
            return set()

    avail_primary   = seedlink_stations(PRIMARY_SEEDLINK)
    avail_secondary = seedlink_stations(SECONDARY_SEEDLINK)
    print(f"primary: {len(avail_primary)} stations, secondary: {len(avail_secondary)} stations")

    # determine which stations are already in the inventory
    existing = load_existing_stations()

    stations = []
    for net in inventory:
        if net.code in EXCLUDED_NETWORKS:
            continue
        for sta in net:
            code = f"{net.code}.{sta.code}"
            if code in avail_primary:
                src = 'primary'
            elif code in avail_secondary:
                src = 'secondary'
            else:
                continue
            # collect unique channels reported in inventory for this station
            chan_list = []
            try:
                for cha in sta:
                    chan_list.append(cha.code)
            except Exception:
                pass
            stations.append({
                'network':   net.code,
                'station':   sta.code,
                'latitude':  sta.latitude,
                'longitude': sta.longitude,
                'source':    src,
                'present':   code in existing,
                'channels':  chan_list,
            })

    # Serialize inventory to StationXML in-memory
    buf = BytesIO()
    inventory.write(buf, format='STATIONXML')
    buf.seek(0)
    stationxml = buf.read().decode('utf-8')

    return jsonify({'stations': stations, 'stationxml': stationxml})


# original apply_bindings function replaced by streaming version later in file


def _apply_seiscomp_bindings_generator(stations, stationxml, reset=False, skip_stream=False):
    """Yield progress dictionaries while performing the binding steps.

    If ``reset`` is True we remove any previously-written station key files
    and clear the temporary inventory before writing the new list.  This
    ensures a full refresh when the user requests it instead of incrementally
    adding stations over successive selections.
    """
    # Safety: never process stations from excluded networks regardless of caller
    stations = [s for s in stations if s.get('network') not in EXCLUDED_NETWORKS]

    # optionally wipe existing bindings before we start
    if reset:
        root = _get_root()
        keydir = os.path.join(root, 'etc', 'key')
        try:
            for fn in os.listdir(keydir):
                if fn.startswith('station_'):
                    os.remove(os.path.join(keydir, fn))
            yield {"step": "reset_keys", "removed": True}
        except Exception:
            yield {"step": "reset_keys", "removed": False}
    # mirror the synchronous implementation but emit results immediately
    try:
        _update_scrttv_channels(stations)
    except Exception:
        pass

    # inventory
    os.makedirs(INV_DIR, exist_ok=True)
    inv_path = os.path.join(INV_DIR, "stations_bbox.xml")
    with open(inv_path, "w") as fh:
        fh.write(stationxml)
    yield {"step": "write_inventory", "path": inv_path}

    # seedlink profiles
    root = _get_root()
    sl_key_dir = os.path.join(root, "etc", "key", "seedlink")
    os.makedirs(sl_key_dir, exist_ok=True)
    for profile_name, host in [("chain_primary", PRIMARY_SEEDLINK),
                                ("chain_secondary", SECONDARY_SEEDLINK)]:
        profile_path = os.path.join(sl_key_dir, f"profile_{profile_name}")
        with open(profile_path, "w") as fh:
            fh.write("sources = chain\n")
            fh.write(f"sources.chain.address = {host}\n")
            fh.write("sources.chain.port = 18000\n")
        yield {"step": "write_sl_profile", "path": profile_path}

    # key files
    written, skipped = [], []
    for sta_info in stations:
        net, sta = sta_info["network"], sta_info["station"]
        src = sta_info["source"]
        sl_profile = f"chain_{src}"
        key_path = os.path.join(root, "etc", "key", f"station_{net}_{sta}")
        try:
            with open(key_path, "w") as fh:
                # Station key files contain only module binding names; per-station
                # parameter overrides (sources.chain.selectors etc.) are NOT valid
                # here – channel selection happens in the seedlink profile file and
                # the chain XML which `seiscomp update-config seedlink` generates.
                fh.write("global\n")
                fh.write(f"seedlink:{sl_profile}\n")
                fh.write("slarchive\n")
            written.append(f"{net}.{sta}")
        except Exception as exc:
            skipped.append({"station": f"{net}.{sta}", "error": str(exc)})
    yield {"step": "write_key_files", "written": len(written), "skipped": skipped}

    def run_and_yield(cmd):
        res = _run(cmd)
        yield res
        return res

    # update scautopick streams config to include these stations
    try:
        _update_scautopick_config(stations)
    except Exception:
        pass

    # SeisComP commands – order matters:
    # 1. Convert FDSN StationXML → SeisComP XML + scinv sync → populates DB
    # 2. update-config seedlink: rebuilds chain XML with the new stations
    # 3. update-config slarchive: rebuilds slarchive station list
    # 4. update-config scrttv: rebuilds scrttv runtime bindings so the GUI
    #    displays the new stations without needing a manual config edit
    for r in _sync_inventory(inv_path):
        yield r
    yield from run_and_yield(["seiscomp", "update-config", "seedlink"])
    yield from run_and_yield(["seiscomp", "update-config", "slarchive"])
    yield from run_and_yield(["seiscomp", "update-config", "scrttv"])

    # patch chain xml
    try:
        import glob, re
        for fn in glob.glob(os.path.join(SEISCOMP_ROOT, 'var', 'lib', 'seedlink', 'chain*.xml')):
            try:
                data = open(fn).read().splitlines()
                out = []
                for line in data:
                    for s in stations:
                        code = f"{s['network']}.{s['station']}"
                        sel = _best_channel_prefix(s.get('channels', []))
                        if f'id="{code}"' in line and 'selectors="' in line and sel:
                            line = re.sub(r'selectors="[^"]*"', f'selectors="{sel}"', line)
                    out.append(line)
                open(fn, 'w').write("\n".join(out))
            except Exception:
                pass
    except Exception:
        pass

    yield from run_and_yield(["seiscomp", "start", "slarchive"])
    yield from run_and_yield(["seiscomp", "restart", "seedlink"])

    # streaming verification
    if not skip_stream:
        try:
            streaming = []
            import time
            deadline = time.time() + 5
            c = None
            while time.time() < deadline:
                try:
                    c = SeedlinkClient("127.0.0.1", timeout=3)
                    break
                except Exception:
                    time.sleep(0.3)
            if c is None:
                raise RuntimeError("cannot connect to local SeedLink")
            t2 = UTCDateTime(); t1 = t2 - 60
            for s in stations:
                got = False
                for _ in range(4):
                    try:
                        st = c.get_waveforms(s['network'], s['station'], '', 'HH?', t1, t2)
                        if st and any(tr.stats.npts > 0 for tr in st):
                            streaming.append(f"{s['network']}.{s['station']}")
                            got = True
                            break
                    except Exception as exc:
                        last_exc = exc
                    time.sleep(0.5)
                if not got and 'last_exc' in locals():
                    streaming.append({"station": f"{s['network']}.{s['station']}",
                                      "error": str(last_exc)})
            yield {"step": "stream", "stations": streaming,
                   "window": f"{t1.isoformat()} - {t2.isoformat()}"}
        except Exception as exc:
            yield {"step": "stream", "error": f"stream check failed: {exc}"}
    else:
        # skip check entirely
        yield {"step": "stream", "skipped": True}


@app.route('/apply', methods=['POST'])
def apply_bindings():
    """Receive station list + StationXML and stream newline-JSON events.

    The JSON payload may include an optional ``reset`` boolean flag.  When
    true the existing station key files are deleted before writing the new
    bindings, effectively performing a hard refresh.  A ``skip_stream`` flag
    suppresses the post‑binding seedlink check (useful when the server is
    slow or unresponsive).
    """

    payload = request.get_json(force=True)
    stations   = payload.get('stations', [])
    stationxml = payload.get('stationxml', '')
    reset      = bool(payload.get('reset', False))
    skip_stream = bool(payload.get('skip_stream', False))

    # if we're not resetting we skip stations that were already marked
    # "present" to avoid rewriting them unnecessarily
    if not reset:
        stations = [s for s in stations if not s.get('present')]

    if not stationxml:
        return jsonify({'error': 'no stationxml provided'}), 400
    if not stations and not reset:
        return jsonify({'applied': 0, 'results': []})

    def gen():
        try:
            for ev in _apply_seiscomp_bindings_generator(stations, stationxml,
                                                       reset=reset,
                                                       skip_stream=skip_stream):
                yield json.dumps(ev) + "\n"
        except Exception as exc:
            yield json.dumps({'error': str(exc)}) + "\n"
    return Response(gen(), mimetype='application/x-ndjson')


if __name__ == '__main__':
    app.run(debug=True)
