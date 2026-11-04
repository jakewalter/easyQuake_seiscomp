"""Pick uploader: publish Pick objects to the SeisComP messaging bus.

Two delivery modes, tried in order:
1. **SeisComP messaging** – uses the ``seiscomp.DataModel.Notifier`` pattern and
   an active ``seiscomp.Communication.Connection`` (passed in by the calling
   Application, or opened automatically from ``seiscomp.cfg``).
2. **YAML fallback** – writes a pick YAML file under ``out_dir`` so that a
   separate ingestion process can import it later (useful when SeisComP
   bindings are absent or the messaging server is unreachable).

Usage (inside a seiscomp.Client.Application subclass)::

    uploader = PickUploader(connection=self.connection())
    uploader.send_pick(network='CI', station='ABC', location='', channel='HHZ',
                       time=UTCDateTime(), phase='P', probability=0.87)
"""

import os
import time as _time
import logging
from typing import Optional

import yaml

log = logging.getLogger(__name__)

# Default spool directory – use $HOME so it's always writable
_DEFAULT_OUT_DIR = os.path.join(os.path.expanduser('~'), 'sceasyquake', 'picks')

# ---------------------------------------------------------------------------
# Optional SeisComP bindings (SC5 uses lowercase module names)
# ---------------------------------------------------------------------------
try:
    import seiscomp.datamodel as sc_dm      # type: ignore
    import seiscomp.core as sc_core         # type: ignore
    HAS_SEISCOMP = True
except Exception:
    HAS_SEISCOMP = False


def _sc_time(utc_time) -> 'sc_core.Time':
    """Convert an ObsPy UTCDateTime (or anything with .isoformat()) to sc_core.Time."""
    try:
        from obspy import UTCDateTime
        if isinstance(utc_time, UTCDateTime):
            ts = str(utc_time)  # '2024-01-15T12:34:56.789000Z'
            ts = ts.rstrip('Z')
            return sc_core.Time.FromString(ts, '%Y-%m-%dT%H:%M:%S.%f')
    except Exception:
        pass
    # Generic fallback: use epoch seconds if available
    try:
        return sc_core.Time(float(utc_time))
    except Exception:
        pass
    return sc_core.Time.GMT()


# ---------------------------------------------------------------------------
# PickUploader
# ---------------------------------------------------------------------------

class PickUploader:
    """Send SeisComP Pick messages (or YAML files as a fallback).

    Parameters
    ----------
    connection:
        An active ``seiscomp.Communication.Connection`` (obtained from
        ``seiscomp.Client.Application.connection()``).  If *None* the uploader
        tries to fall back to YAML files.
    out_dir:
        Directory for YAML fallback files.
    source:
        String identifying this module (used as publicID prefix and author).
    """

    def __init__(
        self,
        connection=None,
        out_dir: str = None,
        source: str = 'sceasyquake',
        agency_id: str = None,
        pick_rate: float = 0.0,
        reconnect: bool = False,
        reconnect_interval_s: float = 10.0,
    ):
        """
        Parameters
        ----------
        pick_rate:
            Maximum picks per second to publish to the messaging bus.
            Set to 0 (default) for no throttling — appropriate for the
            live sceasyquake process.  Gap-recovery runs should use a
            value such as 20.0 to avoid flooding scmaster and destabilising
            other connected clients (e.g. the live sceasyquake process).
        reconnect:
            On a failed send, call ``connection.reconnect()`` (at most once
            per *reconnect_interval_s*) and retry.  Off by default: a regular
            Application's own message loop handles reconnects, but
            gap_recovery overrides ``run()`` so nothing else ever reconnects
            (2026-10-29: scmaster segfaulted and was restarted by the watchdog
            at 05:43; the backfill kept failing every send for 8 h after).
        """
        self._connection = connection
        self.out_dir = out_dir or _DEFAULT_OUT_DIR
        self.source = source
        self.agency_id = agency_id or source  # use system agencyID if provided
        self._pick_rate = float(pick_rate)          # picks/s; 0 = unlimited
        self._min_interval = (1.0 / pick_rate) if pick_rate > 0 else 0.0
        self._last_send_ts: float = 0.0
        self._reconnect = reconnect
        self._reconnect_interval = float(reconnect_interval_s)
        self._last_reconnect_ts: float = float('-inf')
        os.makedirs(self.out_dir, exist_ok=True)
        self._pick_counter = 0
        self._yaml_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_connection(self, connection):
        """Attach or replace the messaging connection (e.g. after app init)."""
        self._connection = connection

    def send_pick(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        time,
        phase: str = 'P',
        probability: Optional[float] = None,
        method: str = 'PhaseNet',
        author: Optional[str] = None,
    ):
        """Send a pick.

        Returns ``True`` on successful messaging delivery, or the path to the
        YAML fallback file on failure / unavailability.
        """
        pick_data = {
            'network': network,
            'station': station,
            'location': location,
            'channel': channel,
            'time': str(time),
            'phase': phase,
            'probability': float(probability) if probability is not None else None,
            'method': method,
            'author': author or self.source,
        }

        if HAS_SEISCOMP and self._connection is not None:
            try:
                return self._send_via_seiscomp(pick_data, time)
            except Exception as exc:
                log.warning('SeisComP pick send failed (%s); falling back to YAML', exc)

        return self._write_yaml(pick_data)

    # ------------------------------------------------------------------
    # SeisComP messaging path
    # ------------------------------------------------------------------

    def _send_via_seiscomp(self, pick_data: dict, time_obj) -> bool:
        """Construct a DataModel.Pick and send it as a Notifier to the PICK group."""
        self._pick_counter += 1
        public_id = f'Pick/{self.source}/{int(_time.time() * 1e6)}/{self._pick_counter}'

        # Build the Pick object (outside Notifier scope so it can be fully populated)
        pick = sc_dm.Pick.Create(public_id)
        if pick is None:
            pick = sc_dm.Pick.Create()

        # Time
        sc_time = _sc_time(time_obj)
        tq = sc_dm.TimeQuantity(sc_time)
        pick.setTime(tq)

        # WaveformStreamID
        wfid = sc_dm.WaveformStreamID(
            pick_data['network'],
            pick_data['station'],
            pick_data['location'] or '',
            pick_data['channel'],
            '',
        )
        pick.setWaveformID(wfid)

        # Phase hint
        ph = sc_dm.Phase()
        ph.setCode(pick_data['phase'])
        pick.setPhaseHint(ph)

        # Method
        pick.setMethodID(pick_data['method'])

        # Evaluation mode / status – SC5 uses constants directly
        try:
            pick.setEvaluationMode(sc_dm.AUTOMATIC)
            pick.setEvaluationStatus(sc_dm.PRELIMINARY)
        except Exception:
            pass

        # Author info via creation info
        try:
            ci = sc_dm.CreationInfo()
            ci.setAuthor(pick_data['author'])
            ci.setAgencyID(self.agency_id)
            ci.setCreationTime(sc_core.Time.GMT())
            pick.setCreationInfo(ci)
        except Exception:
            pass

        # --- Generate the ADD notifier by attaching pick to EventParameters ---
        # sc_dm.Notifier only captures parent.add(child) calls; Pick.Create()
        # alone does not register a notifier even inside Enable/Disable scope.
        sc_dm.Notifier.Enable()
        try:
            ep = sc_dm.EventParameters()
            ep.add(pick)   # ← this is what triggers the OP_ADD notifier
        finally:
            sc_dm.Notifier.Disable()

        msg = sc_dm.Notifier.GetMessage(True)
        if msg is None or not msg.size():
            log.warning('Notifier produced empty message for pick %s', public_id)
            return False

        # Connection.send() returns bool -- it does NOT raise on a dead/broken
        # connection, it just returns False. The caller here used to ignore
        # that return value entirely and log "Published" / return True
        # regardless, so once scmaster died mid-run every subsequent pick was
        # silently dropped for the rest of the process's life while this code
        # kept reporting success (confirmed 2026-08-09: a gap_recovery run
        # lost 95% of its picks this way after a single mid-run scmaster
        # crash, discovered only by cross-checking "picks sent" in the log
        # against actual rows in the DB). Raise so send_pick()'s existing
        # try/except falls back to the YAML spool instead of lying about
        # success.
        if not self._send_msg('PICK', msg):
            raise RuntimeError(f'connection.send() returned False for pick {public_id} '
                                '(messaging connection likely down)')
        # Throttle to avoid flooding scmaster when publishing historical picks
        # from gap_recovery (which can emit thousands of picks in rapid succession).
        # The live sceasyquake process uses pick_rate=0 (no throttle) so real-time
        # latency is unaffected.
        if self._min_interval > 0:
            elapsed = _time.monotonic() - self._last_send_ts
            wait = self._min_interval - elapsed
            if wait > 0:
                _time.sleep(wait)
        self._last_send_ts = _time.monotonic()
        log.info(
            'Published Pick %s → %s.%s.%s.%s @ %s (p=%.2f)',
            public_id,
            pick_data['network'], pick_data['station'],
            pick_data['location'], pick_data['channel'],
            pick_data['time'],
            pick_data['probability'] or 0.0,
        )

        # Also publish a companion SNR amplitude so scautoloc does not stall
        # in "waiting for amplitude" state.  scautoloc subscribes to the
        # AMPLITUDE group and requires an amplitude of type "snr" (controlled
        # by autoloc.amplTypeSNR) before it will process each pick.  Without
        # this, picks sit queued indefinitely because scamp only computes
        # amplitudes *after* scautoloc has already formed an origin (deadlock).
        # We use probability * 10 as a proxy for SNR (range ~0–10), which
        # comfortably exceeds the default minPickSNR threshold of 3.0.
        try:
            prob = float(pick_data.get('probability') or 0.5)
            snr_val = max(0.01, prob * 10.0)

            amp_public_id = f'Amplitude/{self.source}/{int(_time.time() * 1e6)}/{self._pick_counter}'
            amp = sc_dm.Amplitude.Create(amp_public_id)
            if amp is None:
                amp = sc_dm.Amplitude.Create()

            amp.setType('snr')
            amp.setPickID(public_id)
            amp.setWaveformID(wfid)

            amp_val = sc_dm.RealQuantity(snr_val)
            amp.setAmplitude(amp_val)

            try:
                amp_ci = sc_dm.CreationInfo()
                amp_ci.setAuthor(pick_data['author'])
                amp_ci.setAgencyID(self.agency_id)
                amp_ci.setCreationTime(sc_core.Time.GMT())
                amp.setCreationInfo(amp_ci)
            except Exception:
                pass

            sc_dm.Notifier.Enable()
            try:
                ep2 = sc_dm.EventParameters()
                ep2.add(amp)
            finally:
                sc_dm.Notifier.Disable()

            msg2 = sc_dm.Notifier.GetMessage(True)
            if msg2 is not None and msg2.size():
                # Best-effort by design (a missing amplitude just leaves that
                # one pick stalled in scautoloc rather than failing the whole
                # pick), but still worth a visible warning rather than
                # silently swallowing it at debug level -- a dead connection
                # here fails the same way as the PICK send above and used to
                # go completely unnoticed.
                if self._send_msg('AMPLITUDE', msg2):
                    log.debug('Published SNR amplitude %.2f for pick %s', snr_val, public_id)
                else:
                    log.warning('SNR amplitude send returned False for pick %s '
                                '(messaging connection likely down)', public_id)
        except Exception as exc:
            log.debug('SNR amplitude publish failed (non-fatal): %s', exc)

        return True

    def _send_msg(self, group: str, msg) -> bool:
        """``connection.send()``, plus one throttled reconnect-and-retry if enabled."""
        if self._connection.send(group, msg):
            return True
        if not self._reconnect:
            return False
        now = _time.monotonic()
        if now - self._last_reconnect_ts < self._reconnect_interval:
            return False
        self._last_reconnect_ts = now
        log.warning('Send to %s failed; reconnecting to messaging', group)
        try:
            self._connection.reconnect()
        except Exception as exc:
            log.warning('Messaging reconnect failed: %s', exc)
            return False
        if not self._connection.isConnected():
            log.warning('Messaging reconnect did not succeed; will retry in %.0f s',
                        self._reconnect_interval)
            return False
        log.info('Reconnected to messaging')
        return bool(self._connection.send(group, msg))

    # ------------------------------------------------------------------
    # YAML fallback path
    # ------------------------------------------------------------------

    def _write_yaml(self, data: dict) -> str:
        # ms timestamp alone collides when picks fail faster than 1/ms (the
        # throttle only runs after a successful send): 2026-10-29 a backfill
        # wrote 84,830 fallbacks into 66,584 files, silently overwriting ~18k.
        self._yaml_counter += 1
        fname = os.path.join(
            self.out_dir,
            f'pick_{int(_time.time() * 1000)}_{os.getpid()}_{self._yaml_counter}.yml')
        with open(fname, 'w') as fh:
            yaml.safe_dump(data, fh)
        log.info('Wrote pick YAML fallback → %s', fname)
        return fname
