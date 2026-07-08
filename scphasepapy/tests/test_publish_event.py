"""
Tests for scphasepapy._publish_event

Run:
    cd scphasepapy && seiscomp-python -m pytest tests/test_publish_event.py -v
"""

import sys
import os
import calendar
import random
import unittest
from datetime import datetime
from unittest.mock import MagicMock

import seiscomp.core as sc_core
import seiscomp.datamodel as dm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from scphasepapy.scphasepapy import SCPhasePaPyApp


# ---------------------------------------------------------------------------
# Compatibility helper
# ---------------------------------------------------------------------------

def _w(arr):
    """arr.weight() returns float in SC3, FloatWithUncertainty in SC4."""
    w = arr.weight()
    return w.value() if hasattr(w, 'value') else float(w)


# ---------------------------------------------------------------------------
# Stub ORM rows
# ---------------------------------------------------------------------------

class _FakePick:
    def __init__(self, pk_id, sc_id, net='NM', sta='PBMO', modified_id=None):
        self.id = pk_id
        self.sc_pick_id = sc_id
        self.network = net
        self.station = sta
        self.modified_id = modified_id or pk_id


class _FakePickModified:
    def __init__(self, pm_id, assoc_id, phase='P'):
        self.id = pm_id
        self.assoc_id = assoc_id
        self.phase = phase


class _FakeAssociated:
    def __init__(self, eid, lat, lon, ot):
        self.id = eid
        self.latitude = lat
        self.longitude = lon
        self.ot = ot


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _make_app(pms, picks_by_pm_id, pick_id_map=None):
    """
    pms            – list of _FakePickModified for the event
    picks_by_pm_id – {pm.id: [_FakePick, ...]}
    pick_id_map    – {pick.id: sc_pick_id}  (overrides pick.sc_pick_id)
    """
    app = object.__new__(SCPhasePaPyApp)
    app._pick_id_map = pick_id_map or {}
    app._cycle_seconds = 30
    app._last_event_id = 0
    app._heartbeat_count = 0
    app.agencyID = MagicMock(return_value='TEST')

    # Station coordinates for distance computation
    app._station_lat_lon = {
        'NM.PBMO': (34.5, -106.4),
        'NM.MIAR': (34.0, -93.6),
    }

    captured = []
    app.connection = MagicMock()
    app.connection.return_value = app.connection
    app.connection.send = lambda g, m: captured.append((g, m))
    app._captured = captured

    # Build mock assoc_db
    # Use a shared counter so successive PhasePick queries return the right rows
    _pick_call = [0]
    _pm_list = list(pms)
    db = MagicMock()

    def _query(cls):
        q = MagicMock()
        def _filter(*a):
            f = MagicMock()
            name = cls.__name__ if hasattr(cls, '__name__') else str(cls)
            if 'PickModified' in name or cls is _FakePickModified:
                f.all.return_value = list(pms)
            else:  # PhasePick – consecutive calls match pm order
                idx = _pick_call[0]
                _pick_call[0] += 1
                pm_id = _pm_list[idx].id if idx < len(_pm_list) else -1
                f.all.return_value = picks_by_pm_id.get(pm_id, [])
            return f
        q.filter = _filter
        return q

    db.query = _query

    assoc = MagicMock()
    assoc.assoc_db = db
    app._assoc = assoc

    # Drain notifier
    dm.Notifier.SetEnabled(False)
    dm.Notifier.GetMessage(True)
    return app


def _do_publish(pms, picks_by_pm_id, lat=35.0, lon=-106.0, ot=None):
    """Publish one event; return (result, notifier_msg_or_None)."""
    if ot is None:
        ot = datetime(2025, 3, 15, 12, 0, 0)
    # Use large random event ID to avoid SC publicID pool collisions across tests
    eid = random.randint(300000, 999999)
    pick_id_map = {}
    for pl in picks_by_pm_id.values():
        for p in pl:
            pick_id_map[p.id] = p.sc_pick_id
    app = _make_app(pms, picks_by_pm_id, pick_id_map)
    event = _FakeAssociated(eid, lat, lon, ot)
    result = app._publish_event(event)
    msg = app._captured[0][1] if app._captured else None
    return result, msg


def _arrivals_in(msg):
    if msg is None:
        return []
    arrs = []
    for item in msg:
        n = dm.Notifier.Cast(item)
        if n:
            arr = dm.Arrival.Cast(n.object())
            if arr:
                arrs.append(arr)
    return arrs


def _origin_in(msg):
    if msg is None:
        return None
    for item in msg:
        n = dm.Notifier.Cast(item)
        if n:
            orig = dm.Origin.Cast(n.object())
            if orig:
                return orig
    return None


# Shared fixture data
_PM1 = _FakePickModified(1, None, 'P')
_PM2 = _FakePickModified(2, None, 'S')
_PK1 = _FakePick(10, 'NM.PBMO..HHZ.P.2025', 'NM', 'PBMO', modified_id=1)
_PK2 = _FakePick(11, 'NM.MIAR..HHZ.S.2025', 'NM', 'MIAR', modified_id=2)
_PMS  = [_PM1, _PM2]
_PKMAP = {1: [_PK1], 2: [_PK2]}


# ---------------------------------------------------------------------------
# Tests – origin time parsing
# ---------------------------------------------------------------------------

class TestOriginTimeParsing(unittest.TestCase):

    def _pub(self, ot):
        result, msg = _do_publish(_PMS, _PKMAP, ot=ot)
        return result, _origin_in(msg)

    def test_datetime_utc_roundtrip(self):
        dt = datetime(2025, 3, 15, 12, 0, 0)
        expected = float(calendar.timegm(dt.timetuple()))
        result, orig = self._pub(dt)
        self.assertTrue(result)
        self.assertIsNotNone(orig)
        self.assertAlmostEqual(orig.time().value().epoch(), expected, places=1)

    def test_datetime_preserves_microseconds(self):
        dt = datetime(2025, 3, 15, 12, 0, 0, 500000)
        expected = float(calendar.timegm(dt.timetuple())) + 0.5
        result, orig = self._pub(dt)
        self.assertIsNotNone(orig)
        self.assertAlmostEqual(orig.time().value().epoch(), expected, delta=0.01)

    def test_string_with_fractional_seconds(self):
        dt = datetime(2025, 3, 15, 12, 0, 0, 750000)
        expected = float(calendar.timegm(datetime(2025, 3, 15, 12, 0, 0).timetuple())) + 0.75
        result, orig = self._pub('2025-03-15 12:00:00.750000')
        self.assertIsNotNone(orig)
        self.assertAlmostEqual(orig.time().value().epoch(), expected, delta=0.01)

    def test_string_without_fractional_seconds(self):
        expected = float(calendar.timegm(datetime(2025, 3, 15, 12, 0, 0).timetuple()))
        result, orig = self._pub('2025-03-15 12:00:00')
        self.assertIsNotNone(orig)
        self.assertAlmostEqual(orig.time().value().epoch(), expected, places=1)

    def test_invalid_string_does_not_crash(self):
        """Unparseable OT should not crash; origin still published with fallback time."""
        result, orig = self._pub('not-a-date')
        self.assertIsNotNone(orig)


# ---------------------------------------------------------------------------
# Tests – notifier content
# ---------------------------------------------------------------------------

class TestPublishEventNotifierContent(unittest.TestCase):

    def setUp(self):
        dm.Notifier.SetEnabled(False)
        dm.Notifier.GetMessage(True)

    def _pub(self, ot=None):
        return _do_publish(_PMS, _PKMAP, ot=ot or datetime(2025, 3, 15, 12, 0, 0))

    def test_returns_true(self):
        result, _ = self._pub()
        self.assertTrue(result)

    def test_send_called_on_location_group(self):
        eid = random.randint(300000, 999999)
        pick_id_map = {_PK1.id: _PK1.sc_pick_id, _PK2.id: _PK2.sc_pick_id}
        app = _make_app(_PMS, _PKMAP, pick_id_map)
        event = _FakeAssociated(eid, 35.0, -106.0, datetime(2025, 3, 15, 12, 0, 0))
        app._publish_event(event)
        self.assertEqual(len(app._captured), 1)
        self.assertEqual(app._captured[0][0], 'LOCATION')

    def test_arrivals_in_notifier_message(self):
        result, msg = self._pub()
        arrivals = _arrivals_in(msg)
        self.assertEqual(len(arrivals), 2, 'Expected 2 Arrivals, got %d' % len(arrivals))

    def test_arrival_weights_are_defining(self):
        result, msg = self._pub()
        for arr in _arrivals_in(msg):
            self.assertEqual(_w(arr), 1.0, 'weight must be 1.0 so scevent accepts origin')

    def test_arrival_phases_are_correct(self):
        result, msg = self._pub()
        phases = {a.phase().code() for a in _arrivals_in(msg)}
        self.assertIn('P', phases)
        self.assertIn('S', phases)

    def test_arrival_pick_ids_reference_sc_picks(self):
        result, msg = self._pub()
        pick_ids = {a.pickID() for a in _arrivals_in(msg)}
        self.assertIn(_PK1.sc_pick_id, pick_ids)
        self.assertIn(_PK2.sc_pick_id, pick_ids)

    def test_arrival_distance_set(self):
        """scmag requires arrival.distance; verify it is > 0 for known stations."""
        result, msg = self._pub()
        for arr in _arrivals_in(msg):
            d = arr.distance()
            dist_val = d.value() if hasattr(d, 'value') else float(d)
            self.assertGreater(dist_val, 0.0,
                'arrival.distance must be > 0 so scmag can compute magnitudes')

    def test_arrival_azimuth_set(self):
        result, msg = self._pub()
        for arr in _arrivals_in(msg):
            az = arr.azimuth()
            az_val = az.value() if hasattr(az, 'value') else float(az)
            self.assertGreaterEqual(az_val, 0.0)
            self.assertLess(az_val, 360.0)

    def test_unknown_station_does_not_crash(self):
        """An arrival whose station is not in _station_lat_lon should not fail publishing."""
        eid = random.randint(300000, 999999)
        pm_unk = _FakePickModified(99, None, 'P')
        pk_unk = _FakePick(99, 'XX.UNKN..HHZ.P', 'XX', 'UNKN', modified_id=99)
        app = _make_app([pm_unk], {99: [pk_unk]}, {99: pk_unk.sc_pick_id})
        event = _FakeAssociated(eid, 35.0, -106.0, datetime(2025, 3, 15, 12, 0, 0))
        result = app._publish_event(event)
        self.assertTrue(result)

    def test_origin_has_required_metadata(self):
        result, msg = self._pub()
        orig = _origin_in(msg)
        self.assertIsNotNone(orig)
        self.assertEqual(orig.evaluationMode(), dm.AUTOMATIC)
        self.assertEqual(orig.methodID(), 'phasepapy')
        self.assertEqual(orig.quality().usedPhaseCount(), 2)

    def test_origin_public_id_format(self):
        result, msg = self._pub()
        orig = _origin_in(msg)
        self.assertIn('scphasepapy-', orig.publicID())

    def test_no_picks_still_publishes(self):
        """An event with no picks_modified rows still sends (empty-arrival origin)."""
        result, msg = _do_publish([], {}, ot=datetime(2025, 3, 15, 12, 0, 0))
        self.assertTrue(result)
        self.assertIsNotNone(msg)


if __name__ == '__main__':
    unittest.main()
