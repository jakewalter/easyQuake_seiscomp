"""
Tests for scpyocto._publish_event

Run:
    cd scpyocto && seiscomp-python -m pytest tests/test_publish_event.py -v
"""

import sys
import os
import time
import random
import unittest
from unittest.mock import MagicMock

import seiscomp.core as sc_core
import seiscomp.datamodel as dm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from scpyocto.scpyocto import SCPyOctoApp


def _w(arr):
    """Compatibility helper: arr.weight() returns float in SC3, object in SC4."""
    w = arr.weight()
    return w.value() if hasattr(w, 'value') else float(w)


def _make_app():
    """Return an SCPyOctoApp with SC infrastructure stubbed out."""
    app = object.__new__(SCPyOctoApp)
    app._pick_buffer = {}
    app._published_ots = []
    # Large random start prevents SC publicID pool collisions across tests
    app._event_counter = random.randint(300000, 999999)
    app._run_tag = int(time.time())
    app._heartbeat_count = 0
    app._cycle_seconds = 30
    app._max_pick_age = 1800
    app._lat_min = 28.0;  app._lat_max = 38.0
    app._lon_min = -107.0; app._lon_max = -93.0
    app._depth_min = 0.0; app._depth_max = 50.0
    app._vp = 6.0; app._vs = 3.46; app._tolerance = 2.0
    app._time_before = 300.0; app._n_picks = 6; app._n_p_picks = 3
    app._n_s_picks = 0; app._n_p_and_s_picks = 0
    app._pick_match_tolerance = 2.0; app._min_interevent_time = 3.0
    app._time_slicing = 1200.0
    app._assoc = None; app._stations_df = None
    app._station_lat_lon = {
        'NM.PBMO': (34.5, -106.4),
        'NM.MIAR': (34.0, -93.6),
    }
    app.agencyID = MagicMock(return_value='TEST')
    app.connection = MagicMock()
    app.connection.return_value = app.connection
    app.connection.send = MagicMock(return_value=True)
    return app


def _make_evt(lat=35.5, lon=-106.3, depth=10.0, t=None):
    import pandas as pd
    if t is None:
        t = time.time() - 30.0
    return pd.Series({'latitude': lat, 'longitude': lon, 'depth': depth, 'time': t})


def _publish_and_capture(app, assigned_pub_ids, evt=None):
    """Call _publish_event and return (result, notifier_msg)."""
    captured = []
    app.connection.send = lambda g, m: captured.append((g, m))
    if evt is None:
        evt = _make_evt()
    result = app._publish_event(evt, assigned_pub_ids)
    if captured:
        return result, captured[0][1]
    return result, None


def _arrivals_in(msg):
    arrs = []
    for item in msg:
        n = dm.Notifier.Cast(item)
        if not n:
            continue
        arr = dm.Arrival.Cast(n.object())
        if arr:
            arrs.append(arr)
    return arrs


def _origin_in(msg):
    for item in msg:
        n = dm.Notifier.Cast(item)
        if not n:
            continue
        orig = dm.Origin.Cast(n.object())
        if orig:
            return orig
    return None


class TestPublishEvent(unittest.TestCase):

    def setUp(self):
        self.app = _make_app()
        now = time.time() - 30.0
        self.app._pick_buffer['PICK.NM.PBMO.P'] = {'time': now,       'station': 'NM.PBMO', 'phase': 'P'}
        self.app._pick_buffer['PICK.NM.MIAR.S'] = {'time': now + 3.2, 'station': 'NM.MIAR', 'phase': 'S'}
        # Drain any leftover notifier state
        dm.Notifier.SetEnabled(False)
        dm.Notifier.GetMessage(True)

    # ------------------------------------------------------------------
    # Basic success / send path
    # ------------------------------------------------------------------

    def test_returns_true_on_success(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        self.assertTrue(result)

    def test_send_called_on_location_group(self):
        captured = []
        self.app.connection.send = lambda g, m: captured.append((g, m))
        self.app._publish_event(_make_evt(), list(self.app._pick_buffer.keys()))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], 'LOCATION')

    def test_event_counter_increments(self):
        before = self.app._event_counter
        self.app._publish_event(_make_evt(), [])
        self.assertEqual(self.app._event_counter, before + 1)

    # ------------------------------------------------------------------
    # Notifier message content – arrivals
    # ------------------------------------------------------------------

    def test_notifier_message_contains_arrivals(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        self.assertIsNotNone(msg, 'No notifier message captured')
        arrivals = _arrivals_in(msg)
        self.assertEqual(len(arrivals), 2, 'Expected 2 Arrivals in notifier message, got %d' % len(arrivals))

    def test_notifier_message_contains_origin(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        self.assertIsNotNone(_origin_in(msg))

    def test_arrival_pick_ids_correct(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        pick_ids = {a.pickID() for a in _arrivals_in(msg)}
        self.assertIn('PICK.NM.PBMO.P', pick_ids)
        self.assertIn('PICK.NM.MIAR.S', pick_ids)

    def test_arrival_phases_correct(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        phases = {a.phase().code() for a in _arrivals_in(msg)}
        self.assertIn('P', phases)
        self.assertIn('S', phases)

    def test_arrival_weights_are_defining(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        for arr in _arrivals_in(msg):
            self.assertEqual(_w(arr), 1.0, 'Arrival weight must be 1.0 (defining) so scevent accepts it')

    def test_zero_picks_still_publishes(self):
        """An origin with no arrivals should still send (origin itself is valid)."""
        result, msg = _publish_and_capture(self.app, [])
        self.assertTrue(result)
        self.assertIsNotNone(msg)

    def test_arrival_distance_set(self):
        """scmag requires arrival.distance; verify it is set for known stations."""
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        for arr in _arrivals_in(msg):
            dist = arr.distance()
            # distance() may return FloatWithUncertainty or float
            dist_val = dist.value() if hasattr(dist, 'value') else float(dist)
            self.assertGreater(dist_val, 0.0, 'arrival.distance must be > 0 so scmag can compute magnitudes')

    def test_arrival_azimuth_set(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        for arr in _arrivals_in(msg):
            az = arr.azimuth()
            az_val = az.value() if hasattr(az, 'value') else float(az)
            self.assertGreaterEqual(az_val, 0.0)
            self.assertLess(az_val, 360.0)

    def test_arrival_distance_zero_picks_still_publishes(self):
        """Unknown station should not crash publishing."""
        self.app._pick_buffer['PICK.XX.UNKN.P'] = {
            'time': time.time() - 30.0, 'station': 'XX.UNKN', 'phase': 'P'
        }
        result, msg = _publish_and_capture(self.app, ['PICK.XX.UNKN.P'])
        self.assertTrue(result)

    # ------------------------------------------------------------------
    # Origin metadata required by scevent / scmag
    # ------------------------------------------------------------------

    def test_origin_evaluation_mode_is_automatic(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        orig = _origin_in(msg)
        self.assertIsNotNone(orig)
        self.assertEqual(orig.evaluationMode(), dm.AUTOMATIC)

    def test_origin_method_id(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        orig = _origin_in(msg)
        self.assertEqual(orig.methodID(), 'pyocto')

    def test_origin_earth_model_id(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        orig = _origin_in(msg)
        self.assertIn('vp', orig.earthModelID().lower())

    def test_origin_quality_used_phase_count(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        orig = _origin_in(msg)
        self.assertEqual(orig.quality().usedPhaseCount(), 2)

    def test_origin_quality_used_station_count(self):
        result, msg = _publish_and_capture(self.app, list(self.app._pick_buffer.keys()))
        orig = _origin_in(msg)
        self.assertGreaterEqual(orig.quality().usedStationCount(), 1)

    # ------------------------------------------------------------------
    # Origin time
    # ------------------------------------------------------------------

    def test_origin_time_matches_input(self):
        t_unix = 1743000000.0
        result, msg = _publish_and_capture(self.app, [], evt=_make_evt(t=t_unix))
        orig = _origin_in(msg)
        self.assertIsNotNone(orig)
        self.assertAlmostEqual(orig.time().value().epoch(), t_unix, places=1)


if __name__ == '__main__':
    unittest.main()
