#!/usr/bin/env python3
"""scPhasePaPy – PhasePaPy associator as a SeisComP module."""

import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import seiscomp.client as sc_client
import seiscomp.core as sc_core
import seiscomp.datamodel as dm
import seiscomp.logging as sc_log
import seiscomp.math as sc_math

from sqlalchemy import create_engine, text as sqla_text
from sqlalchemy.orm import sessionmaker

# try to import easyQuake phasepapy from working tree or user home
try:
    from easyQuake.phasepapy.assoc1D import LocalAssociator
    from easyQuake.phasepapy.tables1D import Base, Pick as PhasePick, PickModified, Candidate, Associated
    from easyQuake.phasepapy.scnl import SCNL
except Exception:
    sys.path.insert(0, os.path.expanduser('~/easyQuake'))
    from easyQuake.phasepapy.assoc1D import LocalAssociator
    from easyQuake.phasepapy.tables1D import Base, Pick as PhasePick, PickModified, Candidate, Associated
    from easyQuake.phasepapy.scnl import SCNL


def _dt_from_sc_time(sc_time):
    try:
        return datetime.utcfromtimestamp(sc_time.epoch())
    except Exception:
        return datetime.utcnow()


class SCPhasePaPyApp(sc_client.Application):
    def __init__(self):
        sc_client.Application.__init__(self, len(sys.argv), sys.argv)
        self.setMessagingEnabled(True)
        self.setDatabaseEnabled(True, True)
        self.setPrimaryMessagingGroup('PICK')
        self.addMessagingSubscription('PICK')

        self._pick_id_map = {}  # phasepapy internal pick row id -> SeisComP publicID
        self._pick_queue = []   # queue for buffering picks in memory
        self._station_lat_lon = {}  # 'NET.STA' -> (lat_deg, lon_deg)
        self._cycle_seconds = 30
        self._next_cycle = time.time() + self._cycle_seconds
        self._last_event_id = 0
        self._run_tag = int(time.time())
        self._published_ots = []  # recent published OTs (unix float) for dedup
        self._heartbeat_count = 0

        self._db_dir = '/tmp/scphasepapy'
        self._assoc_db_name = 'assoc.db'
        self._tt_db_name = 'tt.db'

        self._max_km = 350
        self._tt_model = 'iasp91'
        self._tt_source_depth = 5.0
        self._aggregation = 1
        self._aggr_norm = 'L2'
        self._assoc_ot_uncert = 40
        self._nsta_declare = 2
        self._cutoff_outlier = 40
        self._loc_uncert_thresh = 0.5
        self._max_pick_age = 3600

        self._assoc = None

    def createCommandLineDescription(self):
        sc_client.Application.createCommandLineDescription(self)
        self.commandline().addGroup('scPhasePaPy')
        self.commandline().addStringOption('scPhasePaPy', 'db-dir', 'Base directory for PhasePaPy SQLite DBs')
        self.commandline().addStringOption('scPhasePaPy', 'assoc-db', 'PhasePaPy associator DB filename')
        self.commandline().addStringOption('scPhasePaPy', 'tt-db', 'PhasePaPy travel-time DB filename')
        self.commandline().addStringOption('scPhasePaPy', 'tt-model', '1D velocity model name or path (.npz, .nd, .tvel)')
        self.commandline().addDoubleOption('scPhasePaPy', 'tt-source-depth', 'Source depth for 1D travel times')
        self.commandline().addDoubleOption('scPhasePaPy', 'max-km', 'Max distance (km) for candidate pairing')
        self.commandline().addDoubleOption('scPhasePaPy', 'aggregation', 'PhasePaPy aggregation parameter')
        self.commandline().addStringOption('scPhasePaPy', 'aggr-norm', 'PhasePaPy aggr_norm (L1/L2)')
        self.commandline().addDoubleOption('scPhasePaPy', 'assoc-ot-uncert', 'Origin time uncertainty (seconds)')
        self.commandline().addDoubleOption('scPhasePaPy', 'nsta-declare', 'Min station count to declare event')
        self.commandline().addDoubleOption('scPhasePaPy', 'cutoff-outlier', 'Outlier cutoff (km)')
        self.commandline().addDoubleOption('scPhasePaPy', 'loc-uncert-thresh', 'Location uncertainty threshold (deg)')
        self.commandline().addDoubleOption('scPhasePaPy', 'cycle-seconds', 'Run association every N seconds')
        self.commandline().addDoubleOption('scPhasePaPy', 'max-pick-age', 'Max pick age to keep (seconds)')
        return True

    def _cfg(self, key, default=''):
        try:
            items = self.configGetStrings(key)
            if items:
                return ','.join(items)
        except Exception:
            pass
        try:
            return self.configGetString(key)
        except Exception:
            return default

    def init(self):
        if not sc_client.Application.init(self):
            return False

        self._db_dir = self._cfg('phasepapy.db_dir', self._db_dir)
        self._assoc_db_name = self._cfg('phasepapy.db_assoc', self._assoc_db_name)
        self._tt_db_name = self._cfg('phasepapy.db_tt', self._tt_db_name)
        self._tt_model = self._cfg('phasepapy.tt_model', self._tt_model)
        self._tt_source_depth = float(self._cfg('phasepapy.tt_source_depth', str(self._tt_source_depth)))
        self._max_km = float(self._cfg('phasepapy.max_km', str(self._max_km)))
        self._aggregation = float(self._cfg('phasepapy.aggregation', str(self._aggregation)))
        self._aggr_norm = self._cfg('phasepapy.aggr_norm', self._aggr_norm)
        self._assoc_ot_uncert = float(self._cfg('phasepapy.assoc_ot_uncert', str(self._assoc_ot_uncert)))
        self._nsta_declare = int(self._cfg('phasepapy.nsta_declare', str(self._nsta_declare)))
        self._cutoff_outlier = float(self._cfg('phasepapy.cutoff_outlier', str(self._cutoff_outlier)))
        self._loc_uncert_thresh = float(self._cfg('phasepapy.loc_uncert_thresh', str(self._loc_uncert_thresh)))
        self._cycle_seconds = int(self._cfg('phasepapy.cycle_seconds', str(self._cycle_seconds)))
        self._max_pick_age = int(self._cfg('phasepapy.max_pick_age', str(self._max_pick_age)))

        try:
            self._cycle_seconds = int(self.commandline().optionDouble('cycle-seconds'))
        except Exception:
            pass

        try:
            val = self.commandline().optionString('tt-model')
            if val:
                self._tt_model = val
        except Exception:
            pass
            
        try:
            self._tt_source_depth = float(self.commandline().optionDouble('tt-source-depth'))
        except Exception:
            pass

        os.makedirs(self._db_dir, exist_ok=True)

        self._ensure_tt_db()
        self._station_lat_lon = self._load_station_coords()

        assoc_db_url = f"sqlite:///{os.path.join(self._db_dir, self._assoc_db_name)}"
        tt_db_url = f"sqlite:///{os.path.join(self._db_dir, self._tt_db_name)}"

        self._assoc = LocalAssociator(
            db_assoc=assoc_db_url,
            db_tt=tt_db_url,
            max_km=self._max_km,
            aggregation=self._aggregation,
            aggr_norm=self._aggr_norm,
            assoc_ot_uncert=self._assoc_ot_uncert,
            nsta_declare=self._nsta_declare,
            cutoff_outlier=self._cutoff_outlier,
            loc_uncert_thresh=self._loc_uncert_thresh,
        )

        sc_log.info('scPhasePaPy: initialized with assoc_db=%s tt_db=%s' % (assoc_db_url, tt_db_url))

        # Ensure picks table has sc_pick_id column (added by us, not in upstream schema)
        try:
            engine = self._assoc.assoc_db.bind
            with engine.connect() as con:
                con.execute('ALTER TABLE picks ADD COLUMN sc_pick_id VARCHAR(255)')
            sc_log.info('scPhasePaPy: added sc_pick_id column to picks table')
        except Exception:
            pass  # column already exists

        # Initialize _last_event_id to current DB max to avoid republishing old events on restart
        try:
            from sqlalchemy import func
            max_id = self._assoc.assoc_db.query(func.max(Associated.id)).scalar()
            if max_id is not None:
                self._last_event_id = int(max_id)
                sc_log.info('scPhasePaPy: resuming from last Associated.id=%d' % self._last_event_id)
        except Exception as e:
            sc_log.warning('scPhasePaPy: could not initialize last_event_id from DB: %s' % e)

        # Restore _pick_id_map from DB for picks inserted in previous sessions.
        # sc_pick_id is not an ORM column (added via ALTER TABLE), so getattr() on a
        # PhasePick ORM object returns None.  The in-memory _pick_id_map is reset on
        # restart.  Load all persisted sc_pick_id values so _publish_event() can
        # resolve public IDs for picks that predate the current session.
        try:
            engine = self._assoc.assoc_db.bind
            with engine.connect() as _con:
                _rows = list(_con.execute('SELECT id, sc_pick_id FROM picks WHERE sc_pick_id IS NOT NULL'))
            for _r in _rows:
                self._pick_id_map[_r[0]] = _r[1]
            sc_log.info('scPhasePaPy: restored %d pick public IDs from DB into _pick_id_map' % len(_rows))
        except Exception as _e:
            sc_log.warning('scPhasePaPy: could not restore pick IDs from DB: %s' % _e)

        sc_log.info('scPhasePaPy: max_km=%s aggregation=%s aggr_norm=%s assoc_ot_uncert=%s nsta_declare=%s cutoff_outlier=%s loc_uncert_thresh=%s cycle_seconds=%s max_pick_age=%s' %
                    (self._max_km, self._aggregation, self._aggr_norm, self._assoc_ot_uncert,
                    self._nsta_declare, self._cutoff_outlier, self._loc_uncert_thresh,
                    self._cycle_seconds, self._max_pick_age))

        self.enableTimer(self._cycle_seconds)
        return True

    def _load_station_coords(self):
        """Return dict NET.STA -> (lat, lon) from the already-loaded SC inventory."""
        coords = {}
        try:
            inv = sc_client.Inventory.Instance()
            if not inv:
                return coords
            sc_inv = inv.inventory()
            if sc_inv:
                for i in range(sc_inv.networkCount()):
                    net = sc_inv.network(i)
                    for j in range(net.stationCount()):
                        sta = net.station(j)
                        key = '%s.%s' % (net.code(), sta.code())
                        if key not in coords:
                            coords[key] = (sta.latitude(), sta.longitude())
        except Exception as e:
            sc_log.warning('scPhasePaPy: could not load station coords for distance calc: %s' % e)
        sc_log.info('scPhasePaPy: loaded %d station coordinates for arrival distance computation' % len(coords))
        return coords

    def _ensure_tt_db(self):
        """Automatically compute travel times for the given 1D model and stations."""
        tt_db_path = os.path.join(self._db_dir, self._tt_db_name)
        state_file = os.path.join(self._db_dir, "tt_state.txt")
        
        # Load SeisComP stations
        inv = sc_client.Inventory.Instance()
        if not inv:
            sc_log.warning('scPhasePaPy: No SeisComP inventory found! TT DB might be incomplete.')
            return
            
        inv.load(self.query())
        sc_inv = inv.inventory()
        
        stations = []
        if sc_inv:
            for i in range(sc_inv.networkCount()):
                net = sc_inv.network(i)
                for j in range(net.stationCount()):
                    sta = net.station(j)
                    stations.append((net.code(), sta.code(), sta.latitude(), sta.longitude(), sta.elevation()))
        
        if not stations:
            sc_log.warning('scPhasePaPy: No stations found in SeisComP DB.')
            
        # To avoid rebuilding on every restart, hash the configuration
        import hashlib
        state_str = f"{self._tt_model}:{self._tt_source_depth}:{len(stations)}"
        h = hashlib.md5(state_str.encode()).hexdigest()
        
        if os.path.exists(tt_db_path) and os.path.exists(state_file):
            with open(state_file, 'r') as f:
                if f.read().strip() == h:
                    sc_log.info('scPhasePaPy: TT DB is up to date (model %s, %d stations).' % (self._tt_model, len(stations)))
                    return
                    
        sc_log.info('scPhasePaPy: Rebuilding PhasePaPy TT DB %s (model %s, %d stations)...' % (tt_db_path, self._tt_model, len(stations)))
        if os.path.exists(tt_db_path):
            try:
                os.remove(tt_db_path)
            except Exception as e:
                sc_log.error('scPhasePaPy: Could not remove old TT DB: %s' % e)
            
        from easyQuake.phasepapy.tt_stations_1D import BaseTT1D, Station1D, TTtable1D
        tt_engine = create_engine(f"sqlite:///{tt_db_path}", echo=False)
        BaseTT1D.metadata.create_all(tt_engine)
        Session = sessionmaker(bind=tt_engine)
        session = Session()
        
        for net, sta, lat, lon, elev in stations:
            st = Station1D(sta, net, '', lat, lon, elev)
            session.add(st)
        session.commit()
        
        # Build TauPyModel travel times
        import numpy as np
        import obspy.taup
        from obspy.taup.taup_create import build_taup_model
        from obspy.geodetics import kilometer2degrees

        model_name = self._tt_model
        if model_name.endswith('.tvel') or model_name.endswith('.nd'):
            base_name = os.path.splitext(os.path.basename(model_name))[0]
            model_npz = os.path.join(self._db_dir, f"{base_name}.npz")
            if not os.path.exists(model_npz):
                build_taup_model(model_name, output_folder=self._db_dir)
            velmod = obspy.taup.TauPyModel(model=model_npz)
        else:
            velmod = obspy.taup.TauPyModel(model=model_name)
            
        distance_km = np.arange(0, self._max_km + 1.0, 1.0)
        for d_km in distance_km:
            d_deg = kilometer2degrees(d_km)
            p_arrivals = velmod.get_travel_times(source_depth_in_km=self._tt_source_depth, distance_in_degree=d_deg, phase_list=['P','p'])
            s_arrivals = velmod.get_travel_times(source_depth_in_km=self._tt_source_depth, distance_in_degree=d_deg, phase_list=['S','s'])
            
            ptimes = [p.time for p in p_arrivals]
            stimes = [s.time for s in s_arrivals]
            
            if ptimes and stimes:
                p_tt = min(ptimes)
                s_tt = min(stimes)
                tt_entry = TTtable1D(float(d_km), float(d_deg), float(p_tt), float(s_tt), float(s_tt - p_tt))
                session.add(tt_entry)
                
        session.commit()
        session.close()
        
        with open(state_file, 'w') as f:
            f.write(h)
        sc_log.info('scPhasePaPy: TT DB built successfully.')

    def run(self):
        sc_log.info('scPhasePaPy: running – buffering picks and associating every %ds' % self._cycle_seconds)
        return sc_client.Application.run(self)

    def handleTimeout(self):
        try:
            self._heartbeat_count += 1
            if self._heartbeat_count % 10 == 0:
                try:
                    n_picks = self._assoc.assoc_db.query(PhasePick).count()
                    n_cand = self._assoc.assoc_db.query(Candidate).filter(Candidate.assoc_id == None).count()
                    n_assoc = self._assoc.assoc_db.query(Associated).count()
                except Exception:
                    n_picks, n_cand, n_assoc = -1, -1, -1
                sc_log.notice('scPhasePaPy: heartbeat – %d picks, %d unassoc candidates, %d total events' % (n_picks, n_cand, n_assoc))
            self._run_association()
        except Exception as e:
            sc_log.error('scPhasePaPy: unhandled exception in handleTimeout: %s' % e)
            sc_log.error(traceback.format_exc())


    def handleMessage(self, msg):
        nm = dm.NotifierMessage.Cast(msg)
        if not nm:
            return

        for item in nm:
            notifier = dm.Notifier.Cast(item)
            if not notifier:
                continue
            obj = notifier.object()
            pick = dm.Pick.Cast(obj)
            if pick:
                self._buffer_pick(pick)

    def _buffer_pick(self, pick):
        try:
            wfid = pick.waveformID()
            net = wfid.networkCode()
            sta = wfid.stationCode()
            cha = wfid.channelCode()
            loc = wfid.locationCode() or ''
            pick_time = _dt_from_sc_time(pick.time().value())
            phase_hint = ''
            try:
                phase_hint = pick.phaseHint().code()
            except Exception:
                pass

            # keep only P/S picks for associator
            if phase_hint not in ('P', 'S', 'Pg', 'Pn', 'P1', 'Sg', 'Sn', 'Sv', 'S1'):
                return

            self._pick_queue.append({
                'public_id': pick.publicID(),
                'net': net,
                'sta': sta,
                'cha': cha,
                'loc': loc,
                'time': pick_time,
                'phase': 'P' if phase_hint.startswith('P') else 'S',
                'phase_hint': phase_hint,
            })
            sc_log.debug('scPhasePaPy: buffered pick %s to memory queue' % pick.publicID())
        except Exception:
            sc_log.warning('scPhasePaPy: failed to buffer pick in memory')

    def _flush_picks(self):
        if not self._pick_queue:
            return

        try:
            picks_to_add = []
            for p in self._pick_queue:
                scnl = SCNL([p['sta'], p['cha'], p['net'], p['loc']])
                record = PhasePick(scnl, p['time'], '', 1.0, 0.1, datetime.utcnow())
                record.phase = p['phase']
                picks_to_add.append((p, record))

            for _, record in picks_to_add:
                self._assoc.assoc_db.add(record)
            self._assoc.assoc_db.commit()

            engine = self._assoc.assoc_db.bind
            with engine.connect() as con:
                updates = []
                for p, record in picks_to_add:
                    updates.append({'sc_pick_id': p['public_id'], 'id': record.id})
                    self._pick_id_map[record.id] = p['public_id']
                if updates:
                    con.execute(sqla_text('UPDATE picks SET sc_pick_id = :sc_pick_id WHERE id = :id'), updates)
            self._assoc.assoc_db.commit()

            for p, record in picks_to_add:
                sc_log.info('scPhasePaPy: buffered pick %s %s.%s %s %s' % (p['public_id'], p['net'], p['sta'], p['phase_hint'], p['time']))

            sc_log.notice('scPhasePaPy: flushed %d picks to database' % len(self._pick_queue))
            self._pick_queue.clear()
        except Exception as e:
            sc_log.error('scPhasePaPy: failed to flush picks to DB: %s' % e)
            sc_log.error(traceback.format_exc())

    def _prune_picks_incident(self):
        # Mirror scpyocto's sliding-window approach: anchor the cutoff to the
        # latest pick time already in the DB rather than wall-clock utcnow().
        # When sceasyquake is processing a backlog (e.g. ~6 h behind real time),
        # utcnow()-based pruning would delete every pick the instant it is
        # inserted, leaving 0 picks for the associator each cycle.
        try:
            from sqlalchemy import func as sqla_func
            latest = self._assoc.assoc_db.query(sqla_func.max(PhasePick.time)).scalar()
        except Exception:
            latest = None
        # If the DB is empty, there is nothing to prune
        if latest is None:
            return
        now = datetime.utcnow()
        reference = latest
        if reference > now + timedelta(seconds=60):
            reference = now
        cutoff = reference - timedelta(seconds=self._max_pick_age)
        q = self._assoc.assoc_db.query(PhasePick).filter(PhasePick.time < cutoff).all()
        if q:
            sc_log.debug('scPhasePaPy: pruning %d picks older than %s (reference=%s)'
                         % (len(q), cutoff.strftime('%H:%M:%S'), reference.strftime('%H:%M:%S')))
        for old in q:
            try:
                del self._pick_id_map[old.id]
            except KeyError:
                pass
            self._assoc.assoc_db.delete(old)
        self._assoc.assoc_db.commit()

    def _run_association(self):
        self._prune_picks_incident()
        self._flush_picks()
        try:
            # id_candidate_events() appends new PickModified/Candidate rows on every
            # call without deduplication, so clear all unassociated ones first – they
            # are fully regenerated from the picks table each cycle.
            self._assoc.assoc_db.query(Candidate).filter(Candidate.assoc_id == None).delete()
            self._assoc.assoc_db.query(PickModified).filter(PickModified.assoc_id == None).delete()
            self._assoc.assoc_db.commit()

            self._assoc.id_candidate_events()

            # Resync picks.modified_id to the actual auto-increment PickModified IDs.
            # pick_cluster() assigns picks.modified_id from a per-call local counter
            # (resets to 0 each invocation) while PickModified.id is a SQLite
            # auto-increment that never resets after row deletion.  After the first
            # cycle the two diverge, so _publish_event's join
            #   PhasePick.filter(modified_id == pm.id)
            # returns nothing and events are published with 0 arrivals.
            # Fix: raw SQL correlated update matching by (sta, net, time) –
            # pick_cluster stores exactly pick.time on the PickModified row.
            try:
                result = self._assoc.assoc_db.execute(sqla_text(
                    'UPDATE picks'
                    ' SET modified_id = ('
                    '  SELECT pm.id FROM picks_modified pm'
                    '  WHERE pm.sta = picks.sta'
                    '    AND pm.net = picks.net'
                    '    AND pm.time = picks.time'
                    '    AND pm.assoc_id IS NULL'
                    '  LIMIT 1'
                    ' )'
                    ' WHERE picks.assoc_id IS NULL'
                ))
                self._assoc.assoc_db.commit()
                self._assoc.assoc_db.expire_all()
                sc_log.debug('scPhasePaPy: modified_id resync updated %d picks' % result.rowcount)
            except Exception as e:
                sc_log.warning('scPhasePaPy: modified_id resync failed: %s' % e)

            # Remove spurious same-phase candidates (P+P or S+S).
            # PhasePaPy's id_candidate_events() pairs picks by arrival-time order
            # and ignores phase labels, so a DL picker that fires twice on P at the
            # same station creates false P+P candidates.  Delete them here.
            try:
                removed = self._assoc.assoc_db.execute(sqla_text(
                    'DELETE FROM candidate'
                    ' WHERE assoc_id IS NULL'
                    ' AND EXISTS ('
                    '  SELECT 1 FROM picks_modified pm_p, picks_modified pm_s'
                    '  WHERE pm_p.id = candidate.p_modified_id'
                    '    AND pm_s.id = candidate.s_modified_id'
                    '    AND pm_p.phase IS NOT NULL'
                    '    AND pm_s.phase IS NOT NULL'
                    '    AND pm_p.phase = pm_s.phase'
                    ' )'
                ))
                self._assoc.assoc_db.commit()
                if removed.rowcount:
                    sc_log.debug('scPhasePaPy: removed %d same-phase (P+P/S+S) spurious candidates'
                                 % removed.rowcount)
            except Exception as e:
                sc_log.warning('scPhasePaPy: same-phase candidate filter failed: %s' % e)

            n_cand = self._assoc.assoc_db.query(Candidate).filter(Candidate.assoc_id == None).count()
            sc_log.notice('scPhasePaPy: %d candidate events from %d picks' % (n_cand, self._assoc.assoc_db.query(PhasePick).filter(PhasePick.assoc_id == None).count()))
            self._assoc.associate_candidates()
            self._assoc.single_phase()

            # Belt-and-suspenders: directly sync picks.assoc_id from picks_modified.
            # set_assoc_id() sets picks.assoc_id via ORM picks.modified_id == pm.id.
            # If that chain breaks (e.g. SQLAlchemy session caching), picks.assoc_id
            # stays NULL and the same picks are re-processed every cycle, publishing
            # the same event repeatedly.  This SQL bypasses the ORM entirely.
            try:
                self._assoc.assoc_db.execute(sqla_text(
                    'UPDATE picks SET assoc_id = ('
                    '  SELECT pm.assoc_id FROM picks_modified pm'
                    '  WHERE pm.id = picks.modified_id AND pm.assoc_id IS NOT NULL'
                    '  LIMIT 1'
                    ') WHERE picks.assoc_id IS NULL'
                    '  AND EXISTS ('
                    '    SELECT 1 FROM picks_modified pm2'
                    '    WHERE pm2.id = picks.modified_id AND pm2.assoc_id IS NOT NULL'
                    '  )'
                ))
                self._assoc.assoc_db.commit()
                self._assoc.assoc_db.expire_all()
            except Exception as e:
                sc_log.warning('scPhasePaPy: assoc_id sync failed: %s' % e)

            self._publish_new_events()
        except Exception as e:
            sc_log.error('scPhasePaPy: association step failed: %s' % str(e))
            sc_log.error(traceback.format_exc())

    def _publish_new_events(self):
        session = self._assoc.assoc_db
        events = session.query(Associated).filter(Associated.id > self._last_event_id).order_by(Associated.id).all()
        for event in events:
            # Compute OT as unix float for dedup check
            ot_unix = None
            try:
                import calendar
                if isinstance(event.ot, datetime):
                    ot_unix = float(calendar.timegm(event.ot.timetuple())) + event.ot.microsecond / 1e6
                elif isinstance(event.ot, str):
                    for _fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                        try:
                            _parsed = datetime.strptime(event.ot, _fmt)
                            ot_unix = float(calendar.timegm(_parsed.timetuple())) + _parsed.microsecond / 1e6
                            break
                        except ValueError:
                            pass
                else:
                    ot_unix = float(event.ot)
            except Exception:
                pass

            # Skip if we recently published an event with the same OT (within 3 s)
            if ot_unix is not None and any(abs(ot_unix - prev) < 3.0 for prev in self._published_ots):
                sc_log.debug('scPhasePaPy: skipping duplicate OT %s (Associated.id=%d)' % (event.ot, event.id))
                self._last_event_id = event.id
                continue

            if self._publish_event(event):
                self._last_event_id = event.id
                if ot_unix is not None:
                    self._published_ots.append(ot_unix)
                    self._published_ots = self._published_ots[-200:]

    def _publish_event(self, event):
        try:
            origin = dm.Origin.Create(f'scphasepapy-{event.id}')
            origin.setLatitude(dm.RealQuantity(float(event.latitude) if event.latitude is not None else 0.0))
            origin.setLongitude(dm.RealQuantity(float(event.longitude) if event.longitude is not None else 0.0))
            origin.setDepth(dm.RealQuantity(0.0))

            # Avoid SWIG Time allocation leaking at shutdown by explicitly managing the object
            if isinstance(event.ot, datetime):
                import calendar
                t_sc = sc_core.Time(float(calendar.timegm(event.ot.timetuple())) + event.ot.microsecond / 1e6)
            elif isinstance(event.ot, str):
                # SQLite may return TIMESTAMP columns as strings
                _parsed = None
                for _fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
                    try:
                        _parsed = datetime.strptime(event.ot, _fmt)
                        break
                    except ValueError:
                        pass
                if _parsed is not None:
                    import calendar
                    t_sc = sc_core.Time(float(calendar.timegm(_parsed.timetuple())) + _parsed.microsecond / 1e6)
                else:
                    t_sc = sc_core.Time()
            else:
                try:
                    t_sc = sc_core.Time(float(event.ot))
                except Exception:
                    t_sc = sc_core.Time()
            origin.setTime(dm.TimeQuantity(t_sc))
            del t_sc

            origin.setEvaluationMode(dm.AUTOMATIC)
            origin.setMethodID('phasepapy')

            try:
                ci = dm.CreationInfo()
                ci.setAuthor('scphasepapy')
                try:
                    ci.setAgencyID(self.agencyID())
                except Exception:
                    pass
                ci.setCreationTime(sc_core.Time.GMT())
                origin.setCreationInfo(ci)
            except Exception:
                pass

            # Arrivals must be added inside Notifier.Enable/Disable so the ADD
            # notifications for each arrival are captured in the notifier message.
            dm.Notifier.Enable()
            ep = dm.EventParameters()
            ep.add(origin)
            stations_used = set()
            # Use raw SQL to retrieve picks and their sc_pick_id.
            # sc_pick_id is not declared in the PhasePick ORM model (added via ALTER TABLE),
            # so ORM attribute access always returns None.  The in-memory _pick_id_map is
            # restored from DB at startup but may lag for picks added in the same cycle.
            # Raw SQL is the only reliable path.
            try:
                _pm_rows = self._assoc.assoc_db.execute(sqla_text(
                    'SELECT pm.id, pm.phase'
                    ' FROM picks_modified pm'
                    ' WHERE pm.assoc_id = :eid'
                ), {'eid': event.id}).fetchall()
            except Exception as _e:
                sc_log.warning('scPhasePaPy: could not query PickModified for event %d: %s' % (event.id, _e))
                _pm_rows = []
            sc_log.debug('scPhasePaPy: event %d: found %d PickModified rows' % (event.id, len(_pm_rows)))
            for _pm_id, _pm_phase in _pm_rows:
                phase_code = (_pm_phase or '').upper()
                try:
                    _pick_rows = self._assoc.assoc_db.execute(sqla_text(
                        'SELECT p.id, p.sc_pick_id, p.net, p.sta'
                        ' FROM picks p'
                        ' WHERE p.modified_id = :mid'
                    ), {'mid': _pm_id}).fetchall()
                except Exception as _e:
                    sc_log.warning('scPhasePaPy: could not query picks for pm.id=%d: %s' % (_pm_id, _e))
                    _pick_rows = []
                for _pick_id, _sc_pick_id, _net, _sta in _pick_rows:
                    public_id = _sc_pick_id or self._pick_id_map.get(_pick_id)
                    if not public_id:
                        sc_log.debug('scPhasePaPy: no public_id for pick id=%d pm.id=%d' % (_pick_id, _pm_id))
                        continue
                    arrival = dm.Arrival()
                    arrival.setPickID(public_id)
                    arrival.setPhase(dm.Phase(phase_code or 'P'))
                    arrival.setWeight(1.0)
                    try:
                        arrival.setTimeResidual(0.0)
                    except Exception:
                        pass
                    try:
                        sta_key = '%s.%s' % (_net or '', _sta or '')
                        sta_coords = self._station_lat_lon.get(sta_key)
                        if sta_coords:
                            dist_deg, az, _ = sc_math.delazi_wgs84(
                                float(event.latitude), float(event.longitude),
                                sta_coords[0], sta_coords[1],
                            )
                            arrival.setDistance(dist_deg)
                            arrival.setAzimuth(az)
                    except Exception:
                        pass
                    origin.add(arrival)
                    stations_used.add(sta_key)
            n_used = origin.arrivalCount()
            try:
                oq = dm.OriginQuality()
                oq.setUsedPhaseCount(n_used)
                oq.setAssociatedPhaseCount(n_used)
                oq.setUsedStationCount(len(stations_used) if stations_used else n_used)
                origin.setQuality(oq)
            except Exception:
                pass
            dm.Notifier.Disable()
            notifier_msg = dm.Notifier.GetMessage(True)

            if notifier_msg and notifier_msg.size() > 0:
                self.connection().send('LOCATION', notifier_msg)
                sc_log.info('scPhasePaPy: published event %d (origin %s) with %d arrivals' % (event.id, origin.publicID(), origin.arrivalCount()))
                return True

            sc_log.warning('scPhasePaPy: no notifier message generated for event %d' % event.id)
            return False
        except Exception as e:
            sc_log.error('scPhasePaPy: failed to publish event %d: %s' % (event.id, str(e)))
            sc_log.error(traceback.format_exc())
            return False


if __name__ == '__main__':
    app = SCPhasePaPyApp()
    sys.exit(app())
