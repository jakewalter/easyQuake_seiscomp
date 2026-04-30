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
        """
        self._connection = connection
        self.out_dir = out_dir or _DEFAULT_OUT_DIR
        self.source = source
        self.agency_id = agency_id or source  # use system agencyID if provided
        self._pick_rate = float(pick_rate)          # picks/s; 0 = unlimited
        self._min_interval = (1.0 / pick_rate) if pick_rate > 0 else 0.0
        self._last_send_ts: float = 0.0
        os.makedirs(self.out_dir, exist_ok=True)
        self._pick_counter = 0

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

        self._connection.send('PICK', msg)
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
                self._connection.send('AMPLITUDE', msg2)
                log.debug('Published SNR amplitude %.2f for pick %s', snr_val, public_id)
        except Exception as exc:
            log.debug('SNR amplitude publish failed (non-fatal): %s', exc)

        return True

    # ------------------------------------------------------------------
    # YAML fallback path
    # ------------------------------------------------------------------

    def _write_yaml(self, data: dict) -> str:
        fname = os.path.join(self.out_dir, f'pick_{int(_time.time() * 1000)}.yml')
        with open(fname, 'w') as fh:
            yaml.safe_dump(data, fh)
        log.info('Wrote pick YAML fallback → %s', fname)
        return fname
