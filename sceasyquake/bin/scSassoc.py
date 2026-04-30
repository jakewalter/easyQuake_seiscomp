#!/usr/bin/env python3
"""scSassoc – automatic S-phase associator for SeisComP.

scautoloc only accepts picks with phaseHint='P' (hard-coded in C++).
This daemon bridges the gap: it subscribes to the PICK and LOCATION
messaging groups, buffers incoming S picks from sceasyquake, and for
every new or updated origin it finds matching S picks at each
P-associated station using the iasp91 travel-time table.  Matched S
picks are added as non-defining arrivals (weight=0) and sent back to
the LOCATION group so that scdb, scevent, scamp, and scmag all see them.

Workflow
--------
  PICK  ──► (buffer S picks in memory)
  LOCATION ──► (receive P-only origin from scautoloc)
                 │
                 ▼
           for each P arrival:
             compute expected S time via TTT
             find best S pick from same net.sta within ±window
           add matched S picks as non-defining arrivals
                 │
                 ▼
  LOCATION ──► (send updated origin notifiers; scdb writes them)

Usage
-----
    # Standalone (test / dev):
    python3 scSassoc.py -H localhost --console=1 -vvv

    # Managed by SeisComP:
    cp scSassoc.py $(seiscomp exec --path)/scSassoc
    seiscomp enable scSassoc
    seiscomp start scSassoc

Configuration (add to /home/jwalter/seiscomp/etc/scSassoc.cfg)
--------------------------------------------------------------
    # Agency of S picks to accept (default: any agency)
    # sassoc.sAgency = se

    # Travel-time table model
    sassoc.tttModel = iasp91

    # S-P association window in seconds (default 3.0)
    sassoc.window = 3.0

    # Max epicentral distance in degrees to attempt S association (default 20)
    sassoc.maxDist = 20.0
"""
import sys
import time
from collections import defaultdict

import seiscomp.client as sc_client
import seiscomp.core as sc_core
import seiscomp.datamodel as dm
import seiscomp.math as sc_math
import seiscomp.seismology as sm
import seiscomp.logging as sc_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _epoch(sc_time):
    """Convert a seiscomp.core.Time to float Unix epoch seconds."""
    return float(sc_time)


def _station_coords_from_dist_az(olat, olon, dist_deg, az_deg):
    """
    Compute approximate station lat/lon given origin coordinates,
    epicentral distance (degrees) and azimuth (degrees).
    Uses seiscomp.math.delandaz2coord which returns [lat, lon].
    """
    try:
        result = sc_math.delandaz2coord(dist_deg, az_deg, olat, olon)
        return result[0], result[1]
    except Exception:
        # Fallback: along north meridian (error < 1 s for S at < 20 deg)
        return olat + dist_deg, olon


def _get_s_ttime(ttt, olat, olon, depth_km, sta_lat, sta_lon):
    """
    Return the first S/Sg/Sn/Sv travel time (in seconds) from ttt.compute(),
    or None if no S phase is found.
    """
    try:
        ttlist = list(ttt.compute(
            olat, olon, max(float(depth_km), 0.01),
            float(sta_lat), float(sta_lon), 0.0
        ))
    except Exception:
        return None
    for r in ttlist:
        if r.phase in ('S', 'Sg', 'Sn', 'Sv', 'S1'):
            return r.time
    return None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class SSAssocApp(sc_client.Application):
    """S-phase associator – adds S arrivals to scautoloc P-only origins."""

    def __init__(self):
        sc_client.Application.__init__(self, len(sys.argv), sys.argv)
        self.setMessagingEnabled(True)
        self.setDatabaseEnabled(True, False)
        # Send results to LOCATION; subscribe to both PICK and LOCATION
        self.setPrimaryMessagingGroup('LOCATION')
        self.addMessagingSubscription('PICK')
        self.addMessagingSubscription('LOCATION')

        # In-memory S pick buffer: net.sta -> list of pick dicts
        self._s_picks = defaultdict(list)
        # Configurable parameters
        self._ttt_model  = 'iasp91'
        self._window     = 3.0    # ± seconds for S-P matching
        self._max_dist   = 20.0   # degrees; no S match beyond this
        self._pick_keep  = 3600   # seconds to keep buffered picks
        self._s_agency   = ''     # empty = accept S picks from any agency

        self._ttt = sm.TravelTimeTable()

    # ------------------------------------------------------------------
    # SeisComP Application lifecycle
    # ------------------------------------------------------------------

    def createCommandLineDescription(self):
        sc_client.Application.createCommandLineDescription(self)
        self.commandline().addGroup('scSassoc')
        self.commandline().addStringOption(
            'scSassoc', 'ttt-model',
            'Travel time table model (default: iasp91)'
        )
        self.commandline().addDoubleOption(
            'scSassoc', 'window',
            'S-P association time window in seconds (default: 3.0)'
        )
        self.commandline().addDoubleOption(
            'scSassoc', 'max-dist',
            'Max epicentral distance in degrees to attempt S association (default: 20)'
        )
        self.commandline().addStringOption(
            'scSassoc', 's-agency',
            'Only buffer S picks from this agency ID (default: any)'
        )
        return True

    def init(self):
        if not sc_client.Application.init(self):
            return False

        # Read config values; fall back to defaults on missing keys
        try:
            self._ttt_model = self.configGetString('sassoc.tttModel')
        except Exception:
            pass
        try:
            self._window = self.configGetDouble('sassoc.window')
        except Exception:
            pass
        try:
            self._max_dist = self.configGetDouble('sassoc.maxDist')
        except Exception:
            pass
        try:
            self._s_agency = self.configGetString('sassoc.sAgency')
        except Exception:
            pass

        # CLI overrides
        try:
            v = self.commandline().optionString('ttt-model')
            if v:
                self._ttt_model = v
        except Exception:
            pass
        try:
            v = self.commandline().optionDouble('window')
            self._window = v
        except Exception:
            pass
        try:
            v = self.commandline().optionDouble('max-dist')
            self._max_dist = v
        except Exception:
            pass
        try:
            v = self.commandline().optionString('s-agency')
            if v:
                self._s_agency = v
        except Exception:
            pass

        try:
            self._ttt.setModel(self._ttt_model)
        except Exception as exc:
            sc_log.warning(
                'scSassoc: could not set TTT model "%s": %s',
                self._ttt_model, str(exc)
            )

        sc_log.info(
            'scSassoc: tttModel=%s window=%.1fs maxDist=%.1fdeg'
            ' sAgency="%s"',
            self._ttt_model, self._window, self._max_dist, self._s_agency
        )
        return True

    def run(self):
        sc_log.info('scSassoc: running – buffering S picks and watching for new origins')
        return sc_client.Application.run(self)

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    def handleMessage(self, msg):
        nm = dm.NotifierMessage.Cast(msg)
        if not nm:
            return
        # Process each notifier in the batch
        for item in nm:
            notifier = dm.Notifier.Cast(item)
            if not notifier:
                continue
            obj = notifier.object()
            # Buffer incoming S picks
            pick = dm.Pick.Cast(obj)
            if pick:
                self._bufferSPick(pick)
                continue
            # Try to add S arrivals to new/updated origins
            origin = dm.Origin.Cast(obj)
            if origin and notifier.operation() in (dm.OP_ADD, dm.OP_UPDATE):
                self._associateS(origin)

    # ------------------------------------------------------------------
    # S pick buffering
    # ------------------------------------------------------------------

    def _bufferSPick(self, pick):
        """Store an S pick in the in-memory buffer."""
        try:
            hint = pick.phaseHint().code()
        except Exception:
            hint = ''
        if hint not in ('S', 'Sg', 'Sn'):
            return

        if self._s_agency:
            try:
                if pick.creationInfo().agencyID() != self._s_agency:
                    return
            except Exception:
                return

        wfid = pick.waveformID()
        net  = wfid.networkCode()
        sta  = wfid.stationCode()
        t    = _epoch(pick.time().value())

        self._s_picks[f'{net}.{sta}'].append({
            'id':   pick.publicID(),
            't':    t,
            'cha':  wfid.channelCode(),
            'loc':  wfid.locationCode(),
            'hint': hint,
        })
        self._prunePicks()

    def _prunePicks(self):
        """Remove S picks older than _pick_keep seconds."""
        cutoff = time.time() - self._pick_keep
        empty = []
        for key, picks in self._s_picks.items():
            pruned = [p for p in picks if p['t'] > cutoff]
            if pruned:
                self._s_picks[key] = pruned
            else:
                empty.append(key)
        for key in empty:
            del self._s_picks[key]

    # ------------------------------------------------------------------
    # S association
    # ------------------------------------------------------------------

    def _associateS(self, origin):
        """
        For each P arrival in *origin*, try to find a matching S pick and
        add it as a non-defining (weight=0) arrival.  Send the updated
        origin back to the LOCATION group via a NotifierMessage.
        """
        # Extract origin hypocenter
        try:
            olat  = origin.latitude().value()
            olon  = origin.longitude().value()
            odep  = origin.depth().value()
            otime = _epoch(origin.time().value())
        except Exception:
            return

        # Collect pick IDs already in the origin (to avoid duplicates)
        used_pick_ids = set()
        # Track stations that already have an S arrival
        s_assoc_sta = set()
        for i in range(origin.arrivalCount()):
            arr = origin.arrival(i)
            used_pick_ids.add(arr.pickID())
            try:
                if arr.phase().code() in ('S', 'Sg', 'Sn'):
                    # Get station from pick
                    pk = dm.Pick.Find(arr.pickID())
                    if pk:
                        wfid = pk.waveformID()
                        s_assoc_sta.add(
                            f'{wfid.networkCode()}.{wfid.stationCode()}'
                        )
            except Exception:
                pass

        new_arrivals = []

        for i in range(origin.arrivalCount()):
            arr = origin.arrival(i)
            try:
                phase_code = arr.phase().code()
            except Exception:
                continue
            if phase_code not in ('P', 'Pg', 'Pn', 'P1'):
                continue

            # Get epicentral distance and azimuth from the arrival
            try:
                dist = arr.distance()
            except Exception:
                continue
            if dist > self._max_dist:
                continue
            try:
                az = arr.azimuth()
            except Exception:
                az = 0.0

            # Get station identity from the P pick
            pk = dm.Pick.Find(arr.pickID())
            if not pk:
                # Pick not in memory (e.g. module started after scautoloc)
                # Skip — we can only act on picks received during our lifetime
                continue
            wfid = pk.waveformID()
            net  = wfid.networkCode()
            sta  = wfid.stationCode()
            key  = f'{net}.{sta}'

            if key in s_assoc_sta:
                continue  # Already have S for this station

            # Compute approximate station lat/lon and expected S time
            sta_lat, sta_lon = _station_coords_from_dist_az(olat, olon, dist, az)
            s_ttime = _get_s_ttime(self._ttt, olat, olon, odep, sta_lat, sta_lon)
            if s_ttime is None:
                continue

            expected_s = otime + s_ttime

            # Find best matching S pick for this station within ±window
            best     = None
            best_dt  = self._window
            for c in self._s_picks.get(key, []):
                if c['id'] in used_pick_ids:
                    continue
                dt = abs(c['t'] - expected_s)
                if dt < best_dt:
                    best_dt = dt
                    best = c

            if not best:
                continue

            residual = best['t'] - expected_s
            sc_log.info(
                'scSassoc: %s pick=%s dist=%.2f res=%.2f s',
                key, best['id'], dist, residual
            )
            new_arrivals.append((best, residual))
            used_pick_ids.add(best['id'])
            s_assoc_sta.add(key)

        if not new_arrivals:
            return

        # Add the new arrivals to the origin and send notifiers
        dm.Notifier.Enable()
        for best, residual in new_arrivals:
            sarr = dm.Arrival()
            sarr.setPickID(best['id'])
            sarr.setPhase(dm.Phase('S'))
            sarr.setTimeResidual(residual)
            sarr.setWeight(0.0)
            try:
                sarr.setTimeUsed(False)
            except Exception:
                pass
            origin.add(sarr)  # automatically registers Notifier(OP_ADD, sarr)

        notifier_msg = dm.Notifier.GetMessage(True)
        dm.Notifier.Disable()

        if notifier_msg:
            self.connection().send('LOCATION', notifier_msg)
            sc_log.info(
                'scSassoc: added %d S arrivals to origin %s',
                len(new_arrivals), origin.publicID()
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = SSAssocApp()
    sys.exit(app())
