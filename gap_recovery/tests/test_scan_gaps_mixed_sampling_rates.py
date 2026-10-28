"""scan_gaps() correctness across both scan modes.

Two regressions are locked down here:

1. A channel whose sampling rate changes mid-window used to abort the entire
   scan.  obspy refuses to merge a Stream when two traces share an id but
   disagree on sampling rate; scan_gaps() merges inside its per-channel loop,
   so one reconfigured datalogger killed the other ~1100 channels and the run
   filled zero gaps (seen 2026-06-19 and 2026-09-17, the latter caused by
   TX.PB13 switching 200 Hz -> 100 Hz).

2. The header-based scan must agree with the full-read scan.  Naively passing
   ``headonly=True`` to obspy's SDSClient does not: it discards traces that
   need trimming to the window, fabricating multi-hour phantom gaps.
"""
import io
import logging

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

import gap_recovery as gr

BOTH_MODES = pytest.mark.parametrize(
    "use_headers", [True, False], ids=["headers", "full_read"])


def _write_sds(root, net, sta, loc, cha, segments):
    """Write segments into an SDS day-file, appending records like slarchive."""
    for rate, start, dur in segments:
        tr = Trace(np.zeros(int(rate * dur), dtype=np.int32))
        tr.stats.network, tr.stats.station = net, sta
        tr.stats.location, tr.stats.channel = loc, cha
        tr.stats.sampling_rate, tr.stats.starttime = rate, start

        d = root / str(start.year) / net / sta / f"{cha}.D"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{net}.{sta}.{loc}.{cha}.D.{start.year}.{start.julday:03d}"

        buf = io.BytesIO()
        Stream([tr]).write(buf, format="MSEED", reclen=512)
        with open(f, "ab") as fh:
            fh.write(buf.getvalue())


@pytest.fixture
def archive(tmp_path):
    """Three channels, each with a real 600 s interior gap.

    MMM01 additionally switches 100 Hz -> 200 Hz across that gap.  ZZZ01 sorts
    after it, so its presence in the results proves the scan did not abort.
    """
    now = UTCDateTime()
    root = tmp_path / "archive"
    for sta, second_rate in (("AAA01", 100.0), ("MMM01", 200.0), ("ZZZ01", 100.0)):
        _write_sds(root, "TX", sta, "00", "HHZ",
                   [(100.0, now - 3600, 600), (second_rate, now - 2400, 2400)])
    return root


# --------------------------------------------------------------------------
# 1. mixed sampling rates
# --------------------------------------------------------------------------

@BOTH_MODES
def test_mixed_sampling_rates_do_not_abort_the_scan(archive, use_headers):
    gaps = gr.scan_gaps(archive, 3600, min_gap_s=300, use_headers=use_headers)
    scanned = {g.station for g in gaps}
    assert scanned == {"AAA01", "MMM01", "ZZZ01"}, (
        "scan stopped early at the mixed-rate channel: %s" % sorted(scanned))


@BOTH_MODES
def test_gap_is_still_detected_across_a_rate_change(archive, use_headers):
    gaps = [g for g in gr.scan_gaps(archive, 3600, min_gap_s=300,
                                    use_headers=use_headers)
            if g.station == "MMM01"]
    assert len(gaps) == 1
    assert gaps[0].source == "intra_file"
    assert gaps[0].duration_s == pytest.approx(600, abs=2)


def test_helper_preserves_every_trace():
    """_merge_per_sampling_rate is only reachable on the full-read path."""
    now = UTCDateTime()
    st = Stream()
    for rate, offset in ((100.0, 0), (200.0, 1200)):
        tr = Trace(np.zeros(int(rate * 600), dtype=np.int32))
        tr.stats.network, tr.stats.station = "TX", "PB01"
        tr.stats.location, tr.stats.channel = "00", "HHZ"
        tr.stats.sampling_rate = rate
        tr.stats.starttime = now + offset
        st.append(tr)

    with pytest.raises(Exception):
        st.copy().merge(method=1, fill_value=None)

    merged = gr._merge_per_sampling_rate(st, "TX.PB01.00.HHZ", Exception("x"))
    assert {tr.stats.sampling_rate for tr in merged} == {100.0, 200.0}
    assert merged.count() == 2


# --------------------------------------------------------------------------
# 2. the two scan modes must agree
# --------------------------------------------------------------------------

def _key(gaps):
    """Comparable, tolerant-to-sub-second summary of a gap set."""
    return sorted((g.station, g.channel, g.source, round(g.duration_s))
                  for g in gaps)


def test_header_scan_matches_full_read(archive):
    assert _key(gr.scan_gaps(archive, 3600, min_gap_s=300, use_headers=True)) == \
           _key(gr.scan_gaps(archive, 3600, min_gap_s=300, use_headers=False))


def test_header_scan_matches_full_read_with_ragged_coverage(tmp_path):
    """Leading gap, trailing gap, several interior gaps, and a wholly absent channel."""
    now = UTCDateTime()
    root = tmp_path / "archive"
    # starts late (lead gap) and stops early (tail gap), with two interior gaps
    _write_sds(root, "TX", "RAG01", "00", "HHZ",
               [(100.0, now - 3000, 400), (100.0, now - 2000, 400),
                (100.0, now - 1000, 300)])
    # full coverage, no gaps at all
    _write_sds(root, "TX", "FUL01", "00", "HHZ", [(100.0, now - 3600, 3600)])
    # present on disk but entirely outside the window
    _write_sds(root, "TX", "OLD01", "00", "HHZ", [(100.0, now - 30 * 86400, 600)])

    a = gr.scan_gaps(root, 3600, min_gap_s=300, use_headers=True)
    b = gr.scan_gaps(root, 3600, min_gap_s=300, use_headers=False)
    assert _key(a) == _key(b)
    assert "FUL01" not in {g.station for g in a}
    assert "OLD01" in {g.station for g in a}


def test_header_scan_reads_data_spilled_past_its_day_file(tmp_path):
    """Day-files are not strictly bounded -- DOY 256 held samples through Sep 14.

    The header scan must widen the day range it opens, or such data looks
    missing and the scan invents a gap that the full read does not see.
    """
    now = UTCDateTime()
    root = tmp_path / "archive"
    # one continuous hour, but filed under *yesterday's* day-file
    yesterday = now - 86400
    seg_start = now - 3600
    tr = Trace(np.zeros(int(100.0 * 3600), dtype=np.int32))
    tr.stats.network, tr.stats.station = "TX", "SPL01"
    tr.stats.location, tr.stats.channel = "00", "HHZ"
    tr.stats.sampling_rate, tr.stats.starttime = 100.0, seg_start
    d = root / str(yesterday.year) / "TX" / "SPL01" / "HHZ.D"
    d.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Stream([tr]).write(buf, format="MSEED", reclen=512)
    (d / f"TX.SPL01.00.HHZ.D.{yesterday.year}.{yesterday.julday:03d}").write_bytes(
        buf.getvalue())

    gaps = gr.scan_gaps(root, 3600, min_gap_s=300, use_headers=True)
    assert not [g for g in gaps if g.station == "SPL01"], (
        "spilled data was missed: %s" % [(g.source, g.duration_s) for g in gaps])


# --------------------------------------------------------------------------
# 3. progress logging
# --------------------------------------------------------------------------

def test_progress_is_logged_even_when_every_channel_is_missing(tmp_path, caplog):
    """The 'entire window missing' path ends the iteration early.

    On a degraded archive that is the common case, so a progress update written
    at the bottom of the loop would almost never fire -- which is the situation
    this logging exists for.
    """
    now = UTCDateTime()
    root = tmp_path / "archive"
    stale = now - 30 * 86400          # present on disk, outside every window we scan
    for i in range(60):
        _write_sds(root, "TX", "S%02d" % i, "00", "HHZ", [(100.0, stale, 60)])

    with caplog.at_level(logging.INFO, logger="gap_recovery"):
        gaps = gr.scan_gaps(root, 3600, min_gap_s=300)

    assert len(gaps) == 60, "expected every channel to be missing in-window"
    assert all(g.source == "missing_file" for g in gaps)

    progress = [r.getMessage() for r in caplog.records if "scan progress" in r.getMessage()]
    assert progress, "no progress lines emitted on the all-missing path"


def test_progress_reports_running_totals(tmp_path, caplog):
    now = UTCDateTime()
    root = tmp_path / "archive"
    stale = now - 30 * 86400
    for i in range(80):
        _write_sds(root, "TX", "T%02d" % i, "00", "HHZ", [(100.0, stale, 60)])

    with caplog.at_level(logging.INFO, logger="gap_recovery"):
        gr.scan_gaps(root, 3600, min_gap_s=300)

    progress = [r.getMessage() for r in caplog.records if "scan progress" in r.getMessage()]
    assert len(progress) >= 2, "expected several updates across 80 channels"
    counts = [int(m.split("scan progress: ")[1].split("/")[0]) for m in progress]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)

    done = [r.getMessage() for r in caplog.records if "Gap scan complete" in r.getMessage()]
    assert done and "min" in done[-1], "completion line should report elapsed time"


# --------------------------------------------------------------------------
# 4. predictor loading must fail loudly, not silently
# --------------------------------------------------------------------------

class _Predictor:
    """Stand-in for a SeisBench predictor.

    Mirrors the real failure mode: load_model() raises (or quietly leaves
    _loaded_model as None) and the object stays usable, warning once per
    picking window instead of raising.
    """

    def __init__(self, fail_times=0, never_loads=False):
        self.fail_times = fail_times
        self.never_loads = never_loads
        self.calls = 0
        self._loaded_model = None

    def load_model(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("CUDA out of memory")
        if not self.never_loads:
            self._loaded_model = object()
        return True


def test_load_retries_and_succeeds(monkeypatch):
    monkeypatch.setattr(gr.time, "sleep", lambda *_: None)
    p = _Predictor(fail_times=2)
    assert gr._load_model_with_retry(p, "test", attempts=3, base_delay=0) is True
    assert p.calls == 3


def test_load_gives_up_and_reports_failure(monkeypatch):
    monkeypatch.setattr(gr.time, "sleep", lambda *_: None)
    p = _Predictor(fail_times=99)
    assert gr._load_model_with_retry(p, "test", attempts=3, base_delay=0) is False
    assert p.calls == 3


def test_silent_no_model_counts_as_failure(monkeypatch):
    """The 2026-09-17 case: load_model() returns truthy but loads nothing.

    Before the fix this was indistinguishable from success and produced 1,600
    empty picking windows over 1h45m.
    """
    monkeypatch.setattr(gr.time, "sleep", lambda *_: None)
    p = _Predictor(never_loads=True)
    assert gr._load_model_with_retry(p, "test", attempts=2, base_delay=0) is False
