"""Regressions from the 2026-10-28 30-day backfill.

1. IRIS answered HTTP 429 after six batches and every remaining batch was
   dropped within seconds (5458 of 5808 gaps unfilled).  Bulk requests must
   back off and retry on 429.
2. Overlapping gaps chained into one 30-day window and every involved channel
   was re-picked over the whole month, duplicating live picks.  Only channels
   with an active gap may be read, and only picks inside that station's gap
   may be published.
"""
from types import SimpleNamespace

import numpy as np
from obspy import Stream, Trace, UTCDateTime

import gap_recovery as gr

T0 = UTCDateTime("2026-10-18T00:00:00")


def test_bulk_fetch_retries_after_429(monkeypatch):
    monkeypatch.setattr(gr.time, "sleep", lambda s: None)

    class Client:
        calls = 0

        def get_waveforms_bulk(self, bulk, quality=None):
            Client.calls += 1
            if Client.calls < 3:
                raise Exception("Sent too many requests in a given amount of time "
                                "('rate limiting').\nHTTP Status code: 429")
            return Stream([Trace(np.zeros(10))])

    st = gr._get_waveforms_bulk_with_backoff(Client(), [], 1)
    assert len(st) == 1 and Client.calls == 3


def test_429_wait_escalates_through_quota_windows():
    from datetime import datetime, timezone
    now = datetime(2026, 10, 28, 7, 27, 20, tzinfo=timezone.utc)
    assert gr._rate_limit_wait_s(0, now) == 60
    # hourly quota resets at 08:00 UTC
    assert gr._rate_limit_wait_s(1, now) == (32 * 60 + 40) + 30
    # daily quota resets at 00:00 UTC
    assert gr._rate_limit_wait_s(2, now) == (16 * 3600 + 32 * 60 + 40) + 30


def test_obspy_429_exception_is_recognised():
    from obspy.clients.fdsn.header import FDSNTooManyRequestsException
    assert gr._is_rate_limited(FDSNTooManyRequestsException("x", None))


def test_non_429_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(gr.time, "sleep", lambda s: None)

    class Client:
        calls = 0

        def get_waveforms_bulk(self, bulk, quality=None):
            Client.calls += 1
            raise ValueError("boom")

    try:
        gr._get_waveforms_bulk_with_backoff(Client(), [], 1)
    except ValueError:
        pass
    assert Client.calls == 1


class FakeSDS:
    reads = []

    def __init__(self, path):
        pass

    def get_waveforms(self, net, sta, loc, cha, t1, t2):
        FakeSDS.reads.append((sta, t1, t2))
        return Stream([Trace(np.zeros(int(t2 - t1)), header=dict(
            network=net, station=sta, location=loc, channel=cha,
            starttime=t1, sampling_rate=1.0))])


class EveryMinutePredictor:
    """Emits one pick per station at every whole minute of the chunk."""

    def predict(self, st):
        picks = []
        for tr in st:
            t = UTCDateTime(int(tr.stats.starttime.timestamp // 60 + 1) * 60)
            while t < tr.stats.endtime:
                picks.append(SimpleNamespace(
                    network=tr.stats.network, station=tr.stats.station,
                    location=tr.stats.location, channel=tr.stats.channel,
                    time=t, phase="P", prob=0.9))
                t += 60
        return picks


class Uploader:
    def __init__(self):
        self.picks = []

    def send_pick(self, **kw):
        self.picks.append(kw)


def test_only_gap_spans_are_picked(monkeypatch):
    import obspy.clients.filesystem.sds as sds_mod
    monkeypatch.setattr(sds_mod, "Client", FakeSDS)
    FakeSDS.reads = []
    day = 86400
    # A long gap on A spans 10 days, chaining B's short gaps at both ends
    # into one merged window; B has live data (and live picks) in between.
    gaps = [
        gr.Gap("TX", "A", "00", "HHZ", T0, T0 + 10 * day, 10 * day, "t"),
        gr.Gap("TX", "B", "00", "HHZ", T0, T0 + 600, 600, "t"),
        gr.Gap("TX", "B", "00", "HHZ", T0 + 9 * day, T0 + 9 * day + 600, 600, "t"),
    ]
    up = Uploader()
    gr._run_picking(gaps, "/nonexistent", EveryMinutePredictor(), up,
                    300.0, 30.0, gr.RecoveryStats())

    b_picks = [p["time"] for p in up.picks if p["station"] == "B"]
    assert b_picks, "B's gap windows must still be picked"
    for pt in b_picks:
        assert (T0 <= pt <= T0 + 600) or (T0 + 9 * day <= pt <= T0 + 9 * day + 600)
    # B must not be read during the 9 days between its gaps.
    b_reads = [t1 for sta, t1, _ in FakeSDS.reads if sta == "B"]
    assert all(t1 < T0 + 1200 or t1 > T0 + 9 * day - 600 for t1 in b_reads)
    assert any(p["station"] == "A" for p in up.picks)


# --- EarthScope account token ------------------------------------------------

import base64
import json
import time as _time


def _jwt(exp):
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{b64({'alg': 'none'})}.{b64({'exp': exp})}.sig"


def _reset_tokens(monkeypatch):
    monkeypatch.setattr(gr, "_es_tokens", {})
    monkeypatch.delenv("ES_OAUTH2__REFRESH_TOKEN", raising=False)


def test_no_credentials_means_anonymous(tmp_path, monkeypatch):
    _reset_tokens(monkeypatch)
    assert gr._earthscope_access_token(tmp_path / "missing.json") is None


def test_valid_access_token_is_used_without_refresh(tmp_path, monkeypatch):
    _reset_tokens(monkeypatch)
    at = _jwt(_time.time() + 3600)
    f = tmp_path / "tokens.json"
    f.write_text(json.dumps({"access_token": at, "refresh_token": "rt"}))

    def boom(*a, **k):
        raise AssertionError("should not refresh")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert gr._earthscope_access_token(f) == at


def test_expiring_token_is_refreshed_and_saved(tmp_path, monkeypatch):
    _reset_tokens(monkeypatch)
    f = tmp_path / "tokens.json"
    f.write_text(json.dumps({"access_token": _jwt(_time.time() + 60),
                             "refresh_token": "rt-old"}))
    fresh = _jwt(_time.time() + 86400)
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"access_token": fresh, "refresh_token": "rt-new"}).encode()

    def fake_urlopen(req, timeout=None):
        sent["body"] = req.data.decode()
        return Resp()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert gr._earthscope_access_token(f) == fresh
    assert "grant_type=refresh_token" in sent["body"] and "rt-old" in sent["body"]
    saved = json.loads(f.read_text())
    assert saved == {"access_token": fresh, "refresh_token": "rt-new"}


def test_auth_header_is_set_on_obspy_client(monkeypatch):
    client = SimpleNamespace(request_headers={"User-Agent": "x"})
    monkeypatch.setattr(gr, "_earthscope_access_token", lambda: "tok")
    assert gr._apply_earthscope_auth(client)
    assert client.request_headers["Authorization"] == "Bearer tok"
    monkeypatch.setattr(gr, "_earthscope_access_token", lambda: None)
    assert not gr._apply_earthscope_auth(client)
    assert "Authorization" not in client.request_headers
