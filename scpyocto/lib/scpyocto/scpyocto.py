#!/usr/bin/env python3
"""scPyOcto – PyOcto phase associator as a SeisComP module."""

import os
import sys
import time
import traceback
import threading
from datetime import datetime

import seiscomp.client as sc_client
import seiscomp.core as sc_core
import seiscomp.datamodel as dm
import seiscomp.logging as sc_log
import seiscomp.math as sc_math

import pandas as pd
import pyocto


class SCPyOctoApp(sc_client.Application):
    def __init__(self):
        sc_client.Application.__init__(self, len(sys.argv), sys.argv)
        self.setMessagingEnabled(True)
        self.setDatabaseEnabled(True, True)
        self.setPrimaryMessagingGroup('PICK')
        self.addMessagingSubscription('PICK')

        self._assoc_thread = None
        self._assoc_results = None

        # {public_id: {time (unix float), station (NET.STA), phase (P/S)}}
        self._pick_buffer = {}
        # List of published origin times (unix float) for deduplication
        self._published_ots = []
        # Monotonically increasing counter for publicID generation.
        # Combined with _run_tag (seconds-since-epoch at startup) to form
        # publicIDs that are unique across restarts and avoid the silent
        # OP_ADD-duplicate-key drop in scdb that leaves stale old origins.
        self._event_counter = 0
        self._run_tag = int(time.time())
        self._heartbeat_count = 0

        # Defaults
        self._cycle_seconds = 30
        self._max_pick_age = 1800

        # Bounding box
        self._lat_min = 28.0
        self._lat_max = 38.0
        self._lon_min = -107.0
        self._lon_max = -93.0
        self._depth_min = 0.0
        self._depth_max = 50.0

        # 0D velocity model
        self._vp = 6.0
        self._vs = 3.46
        self._tolerance = 2.0

        # Association parameters
        self._time_before = 300.0
        self._n_picks = 6
        self._n_p_picks = 3
        self._n_s_picks = 0
        self._n_p_and_s_picks = 0
        self._pick_match_tolerance = 2.0
        self._min_interevent_time = 3.0
        self._time_slicing = 1200.0

        self._assoc = None
        self._stations_df = None
        self._station_lat_lon = {}  # NET.STA -> (lat_deg, lon_deg)

    # ------------------------------------------------------------------
    # Command-line and config
    # ------------------------------------------------------------------

    def createCommandLineDescription(self):
        sc_client.Application.createCommandLineDescription(self)
        self.commandline().addGroup('scPyOcto')
        self.commandline().addDoubleOption('scPyOcto', 'lat-min', 'Minimum latitude of search area (deg)')
        self.commandline().addDoubleOption('scPyOcto', 'lat-max', 'Maximum latitude of search area (deg)')
        self.commandline().addDoubleOption('scPyOcto', 'lon-min', 'Minimum longitude of search area (deg)')
        self.commandline().addDoubleOption('scPyOcto', 'lon-max', 'Maximum longitude of search area (deg)')
        self.commandline().addDoubleOption('scPyOcto', 'depth-min', 'Minimum depth (km)')
        self.commandline().addDoubleOption('scPyOcto', 'depth-max', 'Maximum depth (km)')
        self.commandline().addDoubleOption('scPyOcto', 'vp', 'P wave velocity for 0D model (km/s)')
        self.commandline().addDoubleOption('scPyOcto', 'vs', 'S wave velocity for 0D model (km/s)')
        self.commandline().addDoubleOption('scPyOcto', 'tolerance', 'Velocity model tolerance (s)')
        self.commandline().addDoubleOption('scPyOcto', 'time-before', 'PyOcto time_before overlap (s)')
        self.commandline().addDoubleOption('scPyOcto', 'n-picks', 'Minimum picks to declare an event')
        self.commandline().addDoubleOption('scPyOcto', 'n-p-picks', 'Minimum P picks per event')
        self.commandline().addDoubleOption('scPyOcto', 'n-s-picks', 'Minimum S picks per event')
        self.commandline().addDoubleOption('scPyOcto', 'n-p-and-s-picks', 'Min stations with both P and S picks')
        self.commandline().addDoubleOption('scPyOcto', 'pick-match-tolerance', 'Pick match tolerance (s)')
        self.commandline().addDoubleOption('scPyOcto', 'min-interevent-time', 'Minimum inter-event time (s)')
        self.commandline().addDoubleOption('scPyOcto', 'time-slicing', 'PyOcto time slicing parameter (s)')
        self.commandline().addDoubleOption('scPyOcto', 'cycle-seconds', 'Run association every N seconds')
        self.commandline().addDoubleOption('scPyOcto', 'max-pick-age', 'Maximum pick age to retain (s)')
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

    def _cfg_float(self, cfg_key, cli_key, default):
        try:
            return float(self.commandline().optionDouble(cli_key))
        except Exception:
            pass
        return float(self._cfg(cfg_key, str(default)))

    def _cfg_int(self, cfg_key, cli_key, default):
        try:
            return int(self.commandline().optionDouble(cli_key))
        except Exception:
            pass
        return int(self._cfg(cfg_key, str(default)))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self):
        if not sc_client.Application.init(self):
            return False

        self._lat_min = self._cfg_float('pyocto.lat_min', 'lat-min', self._lat_min)
        self._lat_max = self._cfg_float('pyocto.lat_max', 'lat-max', self._lat_max)
        self._lon_min = self._cfg_float('pyocto.lon_min', 'lon-min', self._lon_min)
        self._lon_max = self._cfg_float('pyocto.lon_max', 'lon-max', self._lon_max)
        self._depth_min = self._cfg_float('pyocto.depth_min', 'depth-min', self._depth_min)
        self._depth_max = self._cfg_float('pyocto.depth_max', 'depth-max', self._depth_max)
        self._vp = self._cfg_float('pyocto.vp', 'vp', self._vp)
        self._vs = self._cfg_float('pyocto.vs', 'vs', self._vs)
        self._tolerance = self._cfg_float('pyocto.tolerance', 'tolerance', self._tolerance)
        self._time_before = self._cfg_float('pyocto.time_before', 'time-before', self._time_before)
        self._n_picks = self._cfg_int('pyocto.n_picks', 'n-picks', self._n_picks)
        self._n_p_picks = self._cfg_int('pyocto.n_p_picks', 'n-p-picks', self._n_p_picks)
        self._n_s_picks = self._cfg_int('pyocto.n_s_picks', 'n-s-picks', self._n_s_picks)
        self._n_p_and_s_picks = self._cfg_int('pyocto.n_p_and_s_picks', 'n-p-and-s-picks', self._n_p_and_s_picks)
        self._pick_match_tolerance = self._cfg_float('pyocto.pick_match_tolerance', 'pick-match-tolerance', self._pick_match_tolerance)
        self._min_interevent_time = self._cfg_float('pyocto.min_interevent_time', 'min-interevent-time', self._min_interevent_time)
        self._time_slicing = self._cfg_float('pyocto.time_slicing', 'time-slicing', self._time_slicing)
        self._cycle_seconds = self._cfg_int('pyocto.cycle_seconds', 'cycle-seconds', self._cycle_seconds)
        self._max_pick_age = self._cfg_int('pyocto.max_pick_age', 'max-pick-age', self._max_pick_age)

        self._stations_df = self._load_stations()
        if self._stations_df is None or len(self._stations_df) == 0:
            sc_log.error('scPyOcto: no stations loaded from inventory – cannot start')
            return False

        # Build fast lookup for distance computation in _publish_event
        self._station_lat_lon = {
            row['id']: (row['latitude'], row['longitude'])
            for _, row in self._stations_df.iterrows()
        }

        velocity_model = pyocto.VelocityModel0D(self._vp, self._vs, self._tolerance)

        self._assoc = pyocto.OctoAssociator.from_area(
            lat=(self._lat_min, self._lat_max),
            lon=(self._lon_min, self._lon_max),
            zlim=(self._depth_min, self._depth_max),
            velocity_model=velocity_model,
            time_before=self._time_before,
            n_picks=self._n_picks,
            n_p_picks=self._n_p_picks,
            n_s_picks=self._n_s_picks,
            n_p_and_s_picks=self._n_p_and_s_picks,
            pick_match_tolerance=self._pick_match_tolerance,
            min_interevent_time=self._min_interevent_time,
            time_slicing=self._time_slicing,
        )

        # Project station coordinates into the associator's local CRS
        self._assoc.transform_stations(self._stations_df)

        sc_log.info(
            'scPyOcto: initialized %d stations area lat=[%.2f,%.2f] lon=[%.2f,%.2f] depth=[%.1f,%.1f]km' %
            (len(self._stations_df), self._lat_min, self._lat_max,
             self._lon_min, self._lon_max, self._depth_min, self._depth_max)
        )
        sc_log.info(
            'scPyOcto: vp=%.2f vs=%.2f tol=%.2f n_picks=%d n_p=%d n_s=%d n_ps=%d match_tol=%.2f cycle=%ds' %
            (self._vp, self._vs, self._tolerance, self._n_picks,
             self._n_p_picks, self._n_s_picks, self._n_p_and_s_picks,
             self._pick_match_tolerance, self._cycle_seconds)
        )

        self.enableTimer(self._cycle_seconds)
        return True

    def _load_stations(self):
        """Load SeisComP inventory and return a stations DataFrame for PyOcto."""
        try:
            inv = sc_client.Inventory.Instance()
            if not inv:
                sc_log.warning('scPyOcto: no SeisComP inventory instance')
                return None
            inv.load(self.query())
            sc_inv = inv.inventory()
            rows = []
            if sc_inv:
                seen = set()
                for i in range(sc_inv.networkCount()):
                    net = sc_inv.network(i)
                    for j in range(net.stationCount()):
                        sta = net.station(j)
                        sid = f'{net.code()}.{sta.code()}'
                        if sid in seen:
                            continue
                        seen.add(sid)
                        rows.append({
                            'id': sid,
                            'latitude': sta.latitude(),
                            'longitude': sta.longitude(),
                            'elevation': sta.elevation(),  # meters
                        })
            df = pd.DataFrame(rows).reset_index(drop=True)
            sc_log.info('scPyOcto: loaded %d stations from inventory' % len(df))
            return df
        except Exception as e:
            sc_log.error('scPyOcto: failed to load stations: %s' % e)
            sc_log.error(traceback.format_exc())
            return None

    def run(self):
        sc_log.info('scPyOcto: running – buffering picks and associating every %ds' % self._cycle_seconds)
        return sc_client.Application.run(self)

    def handleTimeout(self):
        try:
            self._heartbeat_count += 1
            if self._heartbeat_count % 10 == 0:
                sc_log.notice('scPyOcto: heartbeat – %d picks in buffer' % len(self._pick_buffer))
            self._run_association()
        except Exception as e:
            sc_log.error('scPyOcto: unhandled exception in handleTimeout: %s' % e)
            sc_log.error(traceback.format_exc())

    def handleClose(self):
        pass

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

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
            pick_time_unix = pick.time().value().epoch()

            phase_hint = ''
            try:
                phase_hint = pick.phaseHint().code()
            except Exception:
                pass

            if phase_hint in ('P', 'Pg', 'Pn', 'P1', 'pb', 'pP', 'Pb'):
                phase = 'P'
            elif phase_hint in ('S', 'Sg', 'Sn', 'S1', 'Sv', 'Sb', 'sS'):
                phase = 'S'
            else:
                return  # only accept phase-labelled picks

            station_id = f'{net}.{sta}'
            self._pick_buffer[pick.publicID()] = {
                'time': pick_time_unix,
                'station': station_id,
                'phase': phase,
            }
            sc_log.debug('scPyOcto: buffered %s %s %s' % (pick.publicID(), station_id, phase))
        except Exception as e:
            sc_log.warning('scPyOcto: failed to buffer pick: %s' % e)

    # ------------------------------------------------------------------
    # Association cycle
    # ------------------------------------------------------------------

    def _prune_old_picks(self):
        if not self._pick_buffer:
            return
        # Find the latest pick time in the buffer, but cap it at current system time + 60s
        now = time.time()
        max_time = max(p['time'] for p in self._pick_buffer.values())
        if max_time > now + 60.0:
            max_time = now
        cutoff = max_time - self._max_pick_age
        expired = [pid for pid, p in self._pick_buffer.items() if p['time'] < cutoff]
        for pid in expired:
            del self._pick_buffer[pid]
        if expired:
            sc_log.debug('scPyOcto: pruned %d expired picks' % len(expired))
        # Drop published OT records that are now beyond the sliding window
        self._published_ots = [ot for ot in self._published_ots if ot > cutoff]

    def _run_association(self):
        # 1. Process previous background thread results if finished
        if self._assoc_thread is not None:
            if self._assoc_thread.is_alive():
                sc_log.debug('scPyOcto: association background thread is still active')
                return
            else:
                # Thread finished! Process the results.
                results = self._assoc_results
                self._assoc_thread = None
                self._assoc_results = None

                err = results.get('error')
                if err:
                    sc_log.error('scPyOcto: association background step failed: %s' % err[0])
                    sc_log.error(err[1])
                else:
                    events = results.get('events')
                    assignments = results.get('assignments')
                    pub_ids = results.get('pub_ids')

                    if events is not None and len(events) > 0:
                        try:
                            self._assoc.transform_events(events)
                            sc_log.notice('scPyOcto: %d candidate event(s) from %d picks' % (len(events), len(pub_ids)))

                            for _, evt in events.iterrows():
                                ot_unix = float(evt['time'])

                                evt_assignments = assignments[assignments['event_idx'] == evt['idx']]
                                assigned_pub_ids = [pub_ids[int(r['pick_idx'])]
                                                    for _, r in evt_assignments.iterrows()
                                                    if int(r['pick_idx']) < len(pub_ids)]

                                # Skip already-published origin times
                                if any(abs(ot_unix - prev) < self._min_interevent_time
                                       for prev in self._published_ots):
                                    # Consume the associated picks so they are not repeatedly associated in future runs
                                    for pid in assigned_pub_ids:
                                        self._pick_buffer.pop(pid, None)
                                    continue

                                if self._publish_event(evt, assigned_pub_ids):
                                    self._published_ots.append(ot_unix)
                                    # Consume the associated picks so they are not reused in future runs
                                    for pid in assigned_pub_ids:
                                        self._pick_buffer.pop(pid, None)
                        except Exception as e:
                            sc_log.error('scPyOcto: failed to process background results: %s' % e)
                            sc_log.error(traceback.format_exc())
                    else:
                        sc_log.notice('scPyOcto: no events this cycle (%d picks in buffer)' % len(pub_ids))

        # 2. Prune old picks from the buffer (in main thread)
        self._prune_old_picks()

        if len(self._pick_buffer) < self._n_picks:
            sc_log.debug('scPyOcto: %d picks – below n_picks threshold (%d)' %
                         (len(self._pick_buffer), self._n_picks))
            return

        # 3. Spawn a new background thread
        pub_ids = list(self._pick_buffer.keys())
        picks_data = [self._pick_buffer[pid].copy() for pid in pub_ids]
        picks_df = pd.DataFrame(picks_data)

        self._assoc_results = {
            'pub_ids': pub_ids,
            'events': None,
            'assignments': None,
            'error': None
        }

        def worker(assoc, p_df, s_df, res_dict):
            try:
                evs, ass = assoc.associate(p_df, s_df)
                res_dict['events'] = evs
                res_dict['assignments'] = ass
            except Exception as e:
                res_dict['error'] = (e, traceback.format_exc())

        self._assoc_thread = threading.Thread(
            target=worker,
            args=(self._assoc, picks_df, self._stations_df, self._assoc_results)
        )
        sc_log.debug('scPyOcto: starting background association thread with %d picks' % len(pub_ids))
        self._assoc_thread.start()

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_event(self, evt, assigned_pub_ids):
        try:
            self._event_counter += 1
            pub_id = f'scpyocto.{self._run_tag}.{self._event_counter:04d}'

            origin = dm.Origin.Create(pub_id)
            origin.setLatitude(dm.RealQuantity(float(evt['latitude'])))
            origin.setLongitude(dm.RealQuantity(float(evt['longitude'])))
            origin.setDepth(dm.RealQuantity(float(evt.get('depth', 0.0))))

            t_sc = sc_core.Time(float(evt['time']))
            origin.setTime(dm.TimeQuantity(t_sc))
            del t_sc

            origin.setEvaluationMode(dm.AUTOMATIC)
            origin.setMethodID('pyocto')
            origin.setEarthModelID('0D_vp%.2f_vs%.2f' % (self._vp, self._vs))

            try:
                ci = dm.CreationInfo()
                ci.setAuthor('scpyocto')
                try:
                    ci.setAgencyID(self.agencyID())
                except Exception:
                    pass
                ci.setCreationTime(sc_core.Time.GMT())
                origin.setCreationInfo(ci)
            except Exception:
                pass

            dm.Notifier.Enable()
            ep = dm.EventParameters()
            ep.add(origin)
            for pid in assigned_pub_ids:
                phase_code = self._pick_buffer.get(pid, {}).get('phase', 'P')
                station_id = self._pick_buffer.get(pid, {}).get('station', '')
                arrival = dm.Arrival()
                arrival.setPickID(pid)
                arrival.setPhase(dm.Phase(phase_code))
                arrival.setWeight(1.0)
                try:
                    sta_coords = self._station_lat_lon.get(station_id)
                    if sta_coords:
                        dist_deg, az, _ = sc_math.delazi_wgs84(
                            float(evt['latitude']), float(evt['longitude']),
                            sta_coords[0], sta_coords[1],
                        )
                        arrival.setDistance(dist_deg)
                        arrival.setAzimuth(az)
                except Exception:
                    pass
                origin.add(arrival)
            n_used = origin.arrivalCount()
            try:
                oq = dm.OriginQuality()
                oq.setUsedPhaseCount(n_used)
                oq.setAssociatedPhaseCount(n_used)
                oq.setUsedStationCount(len({self._pick_buffer.get(p, {}).get('station', '') for p in assigned_pub_ids}))
                origin.setQuality(oq)
            except Exception:
                pass
            dm.Notifier.Disable()
            notifier_msg = dm.Notifier.GetMessage(True)

            if notifier_msg and notifier_msg.size() > 0:
                self.connection().send('LOCATION', notifier_msg)
                sc_log.info(
                    'scPyOcto: published %s OT=%s lat=%.3f lon=%.3f depth=%.1fkm nobs=%d' % (
                        pub_id,
                        datetime.utcfromtimestamp(evt['time']).strftime('%Y-%m-%dT%H:%M:%S'),
                        evt['latitude'], evt['longitude'],
                        evt.get('depth', 0.0), len(assigned_pub_ids)
                    )
                )
                return True

            sc_log.warning('scPyOcto: no notifier message generated for %s' % pub_id)
            return False

        except Exception as e:
            sc_log.error('scPyOcto: failed to publish event: %s' % e)
            sc_log.error(traceback.format_exc())
            return False


if __name__ == '__main__':
    app = SCPyOctoApp()
    sys.exit(app())
