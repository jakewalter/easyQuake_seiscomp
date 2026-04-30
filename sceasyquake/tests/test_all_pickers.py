"""Unit tests for all sceasyquake predictor modules.

Tests use monkeypatching / light fakes so that a real GPU or downloaded model
weights are never required.  The full seisbench annotate() path is exercised
with a fake model that returns synthetic probability traces.

Run with::

    cd sceasyquake
    pytest tests/test_all_pickers.py -v
"""

import sys
import types
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from obspy import Trace, Stream, UTCDateTime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 100.0
NPTS = 4000  # 40 s at 100 Hz – enough for PhaseNet / EQT


def _make_trace(station='TST', channel='HHZ', sr=SR, npts=NPTS, net='TX', loc='00'):
    tr = Trace(data=np.random.randn(npts).astype(np.float32))
    tr.stats.network = net
    tr.stats.station = station
    tr.stats.location = loc
    tr.stats.channel = channel
    tr.stats.sampling_rate = sr
    tr.stats.starttime = UTCDateTime(2026, 1, 1)
    return tr


def _make_stream(station='TST', sr=SR, npts=NPTS):
    st = Stream()
    for cha in ('HHZ', 'HHN', 'HHE'):
        st += _make_trace(station=station, channel=cha, sr=sr, npts=npts)
    return st


def _prob_trace(ann_channel, station='TST', net='TX', loc='00',
                sr=SR, npts=NPTS, peak_at=2000, peak_val=0.95):
    """Create a synthetic annotation probability trace with one clear peak."""
    data = np.zeros(npts, dtype=np.float32)
    data[peak_at] = peak_val
    data[peak_at - 1] = 0.3
    data[peak_at + 1] = 0.3
    tr = Trace(data=data)
    tr.stats.network = net
    tr.stats.station = station
    tr.stats.location = loc
    tr.stats.channel = ann_channel
    tr.stats.sampling_rate = sr
    tr.stats.starttime = UTCDateTime(2026, 1, 1)
    return tr


def _fake_annotate(st, p_suffix='_P', s_suffix='_S'):
    """Return synthetic annotation stream for any input stream."""
    out = Stream()
    seen = set()
    for tr in st:
        key = (tr.stats.network, tr.stats.station, tr.stats.location)
        if key in seen:
            continue
        seen.add(key)
        # P annotation at sample 2000, S annotation at sample 2800
        out += _prob_trace(f'Model{p_suffix}', station=tr.stats.station,
                           net=tr.stats.network, loc=tr.stats.location,
                           sr=tr.stats.sampling_rate, npts=tr.stats.npts,
                           peak_at=2000, peak_val=0.95)
        out += _prob_trace(f'Model{s_suffix}', station=tr.stats.station,
                           net=tr.stats.network, loc=tr.stats.location,
                           sr=tr.stats.sampling_rate, npts=tr.stats.npts,
                           peak_at=2800, peak_val=0.85)
    return out


def _build_fake_sbm(model_class_name='PhaseNet'):
    """Build a fake seisbench module with a mock model class."""
    fake_model = MagicMock()
    fake_model.annotate.side_effect = lambda st, **kw: _fake_annotate(st)
    fake_model.eval.return_value = fake_model
    fake_model.cuda.return_value = fake_model

    FakeClass = MagicMock()
    FakeClass.from_pretrained.return_value = fake_model
    FakeClass.load.return_value = fake_model
    FakeClass.return_value = fake_model  # FakeClass() → fake_model

    fake_sbm = types.SimpleNamespace()
    setattr(fake_sbm, model_class_name, FakeClass)
    # Add all three for the generic seisbench predictor
    for name in ('PhaseNet', 'GPD', 'EQTransformer'):
        if not hasattr(fake_sbm, name):
            setattr(fake_sbm, name, FakeClass)
    return fake_sbm, fake_model, FakeClass


# ---------------------------------------------------------------------------
# PhaseNetPredictor
# ---------------------------------------------------------------------------

class TestPhaseNetPredictor:
    def test_stub_single_trace(self):
        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        p = PhaseNetPredictor(backend='stub')
        tr = _make_trace()
        picks = p.predict(tr)
        assert len(picks) == 1
        assert picks[0]['phase'] == 'P'
        assert picks[0]['station'] == 'TST'
        assert 0.0 < picks[0]['probability'] <= 1.0

    def test_stub_stream(self):
        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        p = PhaseNetPredictor(backend='stub')
        st = _make_stream()
        picks = p.predict(st)
        # One simulated pick per trace
        assert len(picks) == 3

    def test_stub_predict_multi(self):
        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        p = PhaseNetPredictor(backend='stub')
        pairs = [('TX.A..HHZ', _make_stream('A')), ('TX.B..HHZ', _make_stream('B'))]
        picks = p.predict_multi(pairs)
        assert len(picks) >= 2

    def test_seisbench_backend(self, monkeypatch):
        fake_sbm, fake_model, FakeClass = _build_fake_sbm('PhaseNet')
        monkeypatch.setitem(sys.modules, 'seisbench', types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'seisbench.models', fake_sbm)

        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        p = PhaseNetPredictor(backend='seisbench', threshold=0.5)
        p._model = ('seisbench', fake_sbm)
        p._loaded_model = fake_model
        st = _make_stream()
        picks = p.predict(st)
        assert any(pk['phase'] == 'P' for pk in picks)
        assert any(pk['phase'] == 'S' for pk in picks)

    def test_seisbench_predict_multi_single_annotate_call(self, monkeypatch):
        """predict_multi() must call annotate() exactly once for seisbench."""
        fake_sbm, fake_model, _ = _build_fake_sbm('PhaseNet')
        monkeypatch.setitem(sys.modules, 'seisbench', types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'seisbench.models', fake_sbm)

        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        import torch
        p = PhaseNetPredictor(backend='seisbench', threshold=0.5)
        p._model = ('seisbench', fake_sbm)
        p._loaded_model = fake_model

        pairs = [
            ('TX.A..HHZ', _make_stream('A')),
            ('TX.B..HHZ', _make_stream('B')),
            ('TX.C..HHZ', _make_stream('C')),
        ]
        p.predict_multi(pairs)
        assert fake_model.annotate.call_count == 1, (
            "predict_multi should call annotate() once for seisbench, "
            f"but called {fake_model.annotate.call_count} times"
        )

    def test_pick_dict_fields(self):
        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        p = PhaseNetPredictor(backend='stub')
        picks = p.predict(_make_trace())
        required = {'network', 'station', 'location', 'channel', 'time', 'phase',
                    'probability', 'method'}
        assert required.issubset(picks[0].keys())


# ---------------------------------------------------------------------------
# GPDPredictor
# ---------------------------------------------------------------------------

class TestGPDPredictor:
    def _setup_fake_gpd(self, monkeypatch):
        fake_sbm, fake_model, FakeClass = _build_fake_sbm('GPD')
        # GPD annotation channels use  GPD_P / GPD_S / GPD_N naming
        fake_model.annotate.side_effect = lambda st, **kw: _fake_annotate(
            st, p_suffix='_P', s_suffix='_S'
        )
        monkeypatch.setitem(sys.modules, 'easyQuake', types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'easyQuake.seisbench', fake_sbm)
        return fake_sbm, fake_model, FakeClass

    def test_default_threshold_is_0994(self):
        from sceasyquake.predictors.gpd import GPDPredictor
        p = GPDPredictor()
        assert p.threshold == pytest.approx(0.994)

    def test_custom_threshold(self):
        from sceasyquake.predictors.gpd import GPDPredictor
        p = GPDPredictor(threshold=0.7)
        assert p.threshold == pytest.approx(0.7)

    def test_predict_returns_picks(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup_fake_gpd(monkeypatch)
        from sceasyquake.predictors.gpd import GPDPredictor
        p = GPDPredictor(threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        st = _make_stream()
        picks = p.predict(st)
        assert any(pk['phase'] == 'P' for pk in picks)
        assert all(pk['method'] == 'GPD' for pk in picks)

    def test_predict_multi_single_annotate_call(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup_fake_gpd(monkeypatch)
        from sceasyquake.predictors.gpd import GPDPredictor
        p = GPDPredictor(threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        pairs = [('TX.A..HHZ', _make_stream('A')), ('TX.B..HHZ', _make_stream('B'))]
        p.predict_multi(pairs)
        assert fake_model.annotate.call_count == 1

    def test_custom_weights_statedict(self, monkeypatch, tmp_path):
        """Resolution: .pt file → load_state_dict path."""
        import torch
        fake_sbm, fake_model, FakeClass = _build_fake_sbm('GPD')
        monkeypatch.setitem(sys.modules, 'easyQuake', types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'easyQuake.seisbench', fake_sbm)

        pt_file = tmp_path / 'custom.pt'
        # Write a minimal state dict
        torch.save({}, str(pt_file))

        with patch('torch.load', return_value={}):
            from sceasyquake.predictors.gpd import GPDPredictor
            p = GPDPredictor(model_path=str(pt_file), threshold=0.5)
            p._sbm_module = fake_sbm
            # _load_weights should call GPD() then load_state_dict
            model = p._load_weights(fake_sbm)
            assert model is not None  # returned the fake


# ---------------------------------------------------------------------------
# EQTransformerPredictor
# ---------------------------------------------------------------------------

class TestEQTransformerPredictor:
    def _setup_fake_eqt(self, monkeypatch):
        fake_sbm, fake_model, FakeClass = _build_fake_sbm('EQTransformer')
        monkeypatch.setitem(sys.modules, 'easyQuake', types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'easyQuake.seisbench', fake_sbm)
        return fake_sbm, fake_model, FakeClass

    def test_predict_p_and_s(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup_fake_eqt(monkeypatch)
        from sceasyquake.predictors.eqtransformer import EQTransformerPredictor
        p = EQTransformerPredictor(threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        st = _make_stream()
        picks = p.predict(st)
        phases = {pk['phase'] for pk in picks}
        assert 'P' in phases
        assert 'S' in phases

    def test_predict_multi_single_annotate_call(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup_fake_eqt(monkeypatch)
        from sceasyquake.predictors.eqtransformer import EQTransformerPredictor
        p = EQTransformerPredictor(threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        pairs = [('TX.A..HHZ', _make_stream('A')),
                 ('TX.B..HHZ', _make_stream('B')),
                 ('TX.C..HHZ', _make_stream('C'))]
        p.predict_multi(pairs)
        assert fake_model.annotate.call_count == 1

    def test_method_label(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup_fake_eqt(monkeypatch)
        from sceasyquake.predictors.eqtransformer import EQTransformerPredictor
        p = EQTransformerPredictor(threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        picks = p.predict(_make_stream())
        assert all(pk['method'] == 'EQTransformer' for pk in picks)

    def test_custom_pretrained(self, monkeypatch):
        fake_sbm, fake_model, FakeClass = self._setup_fake_eqt(monkeypatch)
        from sceasyquake.predictors.eqtransformer import EQTransformerPredictor
        p = EQTransformerPredictor(pretrained='ethz', threshold=0.3)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        assert p.pretrained == 'ethz'

    def test_no_model_returns_empty(self):
        from sceasyquake.predictors.eqtransformer import EQTransformerPredictor
        p = EQTransformerPredictor(threshold=0.5)
        picks = p.predict(_make_stream())
        assert picks == []


# ---------------------------------------------------------------------------
# SeisBenchPredictor (generic)
# ---------------------------------------------------------------------------

class TestSeisBenchPredictor:
    def _setup(self, monkeypatch, model_class='PhaseNet'):
        fake_sbm, fake_model, FakeClass = _build_fake_sbm(model_class)
        monkeypatch.setitem(sys.modules, 'easyQuake', types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'easyQuake.seisbench', fake_sbm)
        return fake_sbm, fake_model, FakeClass

    def test_phasenet_architecture(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup(monkeypatch, 'PhaseNet')
        from sceasyquake.predictors.seisbench import SeisBenchPredictor
        p = SeisBenchPredictor(model='PhaseNet', pretrained='stead', threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        picks = p.predict(_make_stream())
        assert any(pk['phase'] == 'P' for pk in picks)
        assert all('SeisBench-PhaseNet' in pk['method'] for pk in picks)

    def test_gpd_architecture(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup(monkeypatch, 'GPD')
        from sceasyquake.predictors.seisbench import SeisBenchPredictor
        p = SeisBenchPredictor(model='GPD', pretrained='original', threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        picks = p.predict(_make_stream())
        assert all('SeisBench-GPD' in pk['method'] for pk in picks)

    def test_eqtransformer_architecture(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup(monkeypatch, 'EQTransformer')
        from sceasyquake.predictors.seisbench import SeisBenchPredictor
        p = SeisBenchPredictor(model='EQTransformer', pretrained='original', threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        picks = p.predict(_make_stream())
        assert all('SeisBench-EQTransformer' in pk['method'] for pk in picks)

    def test_predict_multi_single_annotate_call(self, monkeypatch):
        fake_sbm, fake_model, _ = self._setup(monkeypatch, 'PhaseNet')
        from sceasyquake.predictors.seisbench import SeisBenchPredictor
        p = SeisBenchPredictor(model='PhaseNet', threshold=0.5)
        p._sbm_module = fake_sbm
        p._loaded_model = fake_model
        pairs = [(f'TX.S{i}..HHZ', _make_stream(f'S{i}')) for i in range(5)]
        p.predict_multi(pairs)
        assert fake_model.annotate.call_count == 1, (
            "predict_multi must issue exactly one annotate() call "
            f"(got {fake_model.annotate.call_count})"
        )

    def test_unsupported_model_raises(self, monkeypatch):
        fake_sbm, _, _ = self._setup(monkeypatch, 'PhaseNet')
        from sceasyquake.predictors.seisbench import SeisBenchPredictor
        p = SeisBenchPredictor(model='NonExistentModel', threshold=0.5)
        p._sbm_module = fake_sbm
        with pytest.raises(RuntimeError, match='not found in seisbench backend'):
            p.load_model()

    def test_custom_statedict_path(self, monkeypatch, tmp_path):
        """A .pt file is loaded via load_state_dict, not from_pretrained."""
        import torch
        fake_sbm, fake_model, FakeClass = self._setup(monkeypatch, 'PhaseNet')
        pt_file = tmp_path / 'weights.pt'
        torch.save({}, str(pt_file))

        with patch('torch.load', return_value={}):
            from sceasyquake.predictors.seisbench import SeisBenchPredictor
            p = SeisBenchPredictor(model='PhaseNet', model_path=str(pt_file))
            p._sbm_module = fake_sbm
            model = p._load_weights(fake_sbm)
        # Should have called FakeClass() to get an instance, then load_state_dict
        assert model is not None
        FakeClass.from_pretrained.assert_not_called()

    def test_custom_seisbench_dir(self, monkeypatch, tmp_path):
        """A directory path is loaded via ModelClass.load()."""
        fake_sbm, fake_model, FakeClass = self._setup(monkeypatch, 'PhaseNet')
        model_dir = tmp_path / 'my_model'
        model_dir.mkdir()
        # Create the .json that seisbench expects
        (model_dir / 'my_model.json').write_text('{}')

        from sceasyquake.predictors.seisbench import SeisBenchPredictor
        p = SeisBenchPredictor(model='PhaseNet', model_path=str(model_dir))
        p._sbm_module = fake_sbm
        model = p._load_weights(fake_sbm)
        FakeClass.load.assert_called_once()
        FakeClass.from_pretrained.assert_not_called()


# ---------------------------------------------------------------------------
# Stream factory (_make_predictor)
# ---------------------------------------------------------------------------

class TestMakePredictor:
    """Tests the _make_predictor factory in sceasyquake-stream.py."""

    def _import_factory(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            'sceasyquake_stream',
            os.path.join(os.path.dirname(__file__),
                         '..', 'bin', 'sceasyquake-stream.py')
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_backend_phasenet(self):
        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        mod = self._import_factory()
        p = mod._make_predictor(backend='phasenet', threshold=0.5, device='cpu')
        assert isinstance(p, PhaseNetPredictor)

    def test_backend_stub(self):
        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        mod = self._import_factory()
        p = mod._make_predictor(backend='stub', threshold=0.5, device='cpu')
        assert isinstance(p, PhaseNetPredictor)
        assert p.backend == 'stub'

    def test_backend_gpd(self):
        from sceasyquake.predictors.gpd import GPDPredictor
        mod = self._import_factory()
        p = mod._make_predictor(backend='gpd', gpd_threshold=0.994, threshold=0.5, device='cpu')
        assert isinstance(p, GPDPredictor)
        assert p.threshold == pytest.approx(0.994)

    def test_backend_eqtransformer(self):
        from sceasyquake.predictors.eqtransformer import EQTransformerPredictor
        mod = self._import_factory()
        p = mod._make_predictor(backend='eqtransformer', threshold=0.3, device='cpu')
        assert isinstance(p, EQTransformerPredictor)

    def test_backend_seisbench(self):
        from sceasyquake.predictors.seisbench import SeisBenchPredictor
        mod = self._import_factory()
        p = mod._make_predictor(backend='seisbench', model='GPD',
                                threshold=0.5, device='cpu')
        assert isinstance(p, SeisBenchPredictor)
        assert p.model_name == 'GPD'

    def test_unknown_backend_falls_back(self):
        from sceasyquake.predictors.phasenet import PhaseNetPredictor
        mod = self._import_factory()
        p = mod._make_predictor(backend='bogus', threshold=0.5, device='cpu')
        assert isinstance(p, PhaseNetPredictor)


# ---------------------------------------------------------------------------
# Pick dict schema
# ---------------------------------------------------------------------------

class TestPickSchema:
    REQUIRED = {'network', 'station', 'location', 'channel',
                'time', 'phase', 'probability', 'method'}

    @pytest.mark.parametrize('backend', ['phasenet', 'gpd', 'eqtransformer', 'seisbench'])
    def test_pick_schema_common_fields(self, backend, monkeypatch):
        """All predictor backends must return picks with the standard field set."""
        fake_sbm, fake_model, FakeClass = _build_fake_sbm('PhaseNet')
        monkeypatch.setitem(sys.modules, 'easyQuake', types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'easyQuake.seisbench', fake_sbm)
        monkeypatch.setitem(sys.modules, 'seisbench', types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'seisbench.models', fake_sbm)

        if backend == 'phasenet':
            # Seisbench path
            from sceasyquake.predictors.phasenet import PhaseNetPredictor
            p = PhaseNetPredictor(backend='seisbench', threshold=0.5)
            p._model = ('seisbench', fake_sbm)
            p._loaded_model = fake_model
        elif backend == 'gpd':
            from sceasyquake.predictors.gpd import GPDPredictor
            p = GPDPredictor(threshold=0.5)
            p._sbm_module = fake_sbm
            p._loaded_model = fake_model
        elif backend == 'eqtransformer':
            from sceasyquake.predictors.eqtransformer import EQTransformerPredictor
            p = EQTransformerPredictor(threshold=0.5)
            p._sbm_module = fake_sbm
            p._loaded_model = fake_model
        else:
            from sceasyquake.predictors.seisbench import SeisBenchPredictor
            p = SeisBenchPredictor(model='PhaseNet', threshold=0.5)
            p._sbm_module = fake_sbm
            p._loaded_model = fake_model

        picks = p.predict(_make_stream())
        assert len(picks) > 0, f'{backend} returned no picks on synthetic stream'
        for pk in picks:
            missing = self.REQUIRED - pk.keys()
            assert not missing, f'{backend} pick missing fields: {missing}'
            assert pk['phase'] in ('P', 'S'), f"Invalid phase: {pk['phase']}"
            assert 0.0 <= pk['probability'] <= 1.0
