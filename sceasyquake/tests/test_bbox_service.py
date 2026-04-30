from examples import bbox_station_service
import json


def test_channel_selector_prefixes():
    # various channel lists should yield first-two-character patterns with '?'
    assert bbox_station_service._channel_selector_prefixes(['HHE', 'HHN', 'HHZ']) == ['HH?']
    assert bbox_station_service._channel_selector_prefixes(['BHZ', 'LHZ', 'FHZ']) == ['BH?', 'LH?', 'FH?']
    # duplicates should be removed and order preserved
    assert bbox_station_service._channel_selector_prefixes(['SHZ', 'SHN', 'EHZ']) == ['SH?', 'EH?']
    # more than three unique prefixes is truncated
    assert bbox_station_service._channel_selector_prefixes(['AA1','BB2','CC3','DD4']) == ['AA?','BB?','CC?']
    # short codes ignored gracefully
    assert bbox_station_service._channel_selector_prefixes(['X','YZ']) == ['YZ?']


def test_best_channel_prefix():
    # HH takes priority over everything
    assert bbox_station_service._best_channel_prefix(['BHZ','HHZ','EHZ']) == 'HH?'
    # EH beats BH
    assert bbox_station_service._best_channel_prefix(['BHZ','EHZ']) == 'EH?'
    # only BH available
    assert bbox_station_service._best_channel_prefix(['BHZ','BHN','BHE']) == 'BH?'
    # SH and CH: SH wins
    assert bbox_station_service._best_channel_prefix(['SHZ','CHZ']) == 'SH?'
    # CH only
    assert bbox_station_service._best_channel_prefix(['CHZ','CHN','CHE']) == 'CH?'
    # empty list falls back to HH?
    assert bbox_station_service._best_channel_prefix([]) == 'HH?'
    # unknown bands: returns alphabetically first
    assert bbox_station_service._best_channel_prefix(['ZZ1','XY2']) == 'XY?'


def test_apply_bindings_stream(monkeypatch, tmp_path):
    # prepare minimal environment and dummy stations
    os = __import__('os')
    import tempfile
    root = tmp_path / "scroot"
    root.mkdir()
    os.environ['SEISCOMP_ROOT'] = str(root)
    # ensure key dirs exist
    (root / 'etc' / 'key' / 'seedlink').mkdir(parents=True)
    # stub _run to avoid calling real binaries
    monkeypatch.setattr(bbox_station_service, '_run', lambda cmd, **kw: {'cmd':' '.join(cmd),'rc':0,'stdout':'','stderr':''})
    client = bbox_station_service.app.test_client()
    stations=[{'network':'XX','station':'YY','source':'primary','present':False,'channels':['HHZ']}]
    resp = client.post('/apply', json={'stations':stations,'stationxml':'<xml/>'})
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    lines = [l for l in text.splitlines() if l.strip()]
    # first nonempty line should be inventory event
    ev0 = json.loads(lines[0])
    assert ev0.get('step') == 'write_inventory'


def test_apply_bindings_reset(tmp_path, monkeypatch):
    # if reset flag is supplied existing station files should be removed
    import os
    root = tmp_path / "scroot"
    root.mkdir()
    os.environ['SEISCOMP_ROOT'] = str(root)
    keydir = root / 'etc' / 'key'
    (keydir).mkdir(parents=True)
    # create a fake station key that should be deleted
    fake = keydir / 'station_OLD_ZZ'
    fake.write_text('dummy')
    monkeypatch.setattr(bbox_station_service, '_run', lambda cmd, **kw: {'cmd':' '.join(cmd),'rc':0,'stdout':'','stderr':''})
    client = bbox_station_service.app.test_client()
    stations=[{'network':'AA','station':'BB','source':'primary','present':False,'channels':['HHZ']}]
    resp = client.post('/apply', json={'stations':stations,'stationxml':'<xml/>','reset':True})
    assert resp.status_code == 200
    # confirm the fake file has been removed
    assert not fake.exists()


def test_apply_bindings_skip_stream(tmp_path, monkeypatch):
    # ensure the generator emits a skipped step when skip_stream is True
    import os
    root = tmp_path / "scroot"
    root.mkdir()
    os.environ['SEISCOMP_ROOT'] = str(root)
    monkeypatch.setattr(bbox_station_service, '_run', lambda cmd, **kw: {'cmd':' '.join(cmd),'rc':0,'stdout':'','stderr':''})
    client = bbox_station_service.app.test_client()
    stations=[{'network':'FF','station':'GG','source':'primary','present':False,'channels':['BHZ']}]
    resp = client.post('/apply', json={'stations':stations,'stationxml':'<xml/>','skip_stream':True})
    assert resp.status_code == 200
    out = resp.get_data(as_text=True)
    assert '"skipped": true' in out
    # ensure no attempt to contact Seedlink by monkeypatching the client
    # we can monkeypatch the SeedlinkClient constructor to raise if called
    def bad_constructor(*args, **kw):
        raise AssertionError("should not be invoked")
    monkeypatch.setattr(bbox_station_service, 'SeedlinkClient', bad_constructor)
    # call handler directly to exercise path, not HTTP
    gen = bbox_station_service._apply_seiscomp_bindings_generator(stations, '<xml/>', reset=False)
    # manually simulate skip_stream by rewriting call
    # easier: call generator through apply_bindings to use skip_stream
    resp2 = client.post('/apply', json={'stations':stations,'stationxml':'<xml/>','skip_stream':True})
    assert resp2.status_code == 200


def test_update_scautopick_config(tmp_path):
    import os
    from examples import bbox_station_service
    root = tmp_path / "scroot"
    root.mkdir()
    os.environ['SEISCOMP_ROOT'] = str(root)
    cfg_dir = root / 'etc' / 'defaults'
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / 'scautopick.cfg'
    cfg_file.write_text('# nothing\n')
    stations=[
        {'network':'AA','station':'BBB','channels':['BHZ','HHZ']},
        {'network':'CC','station':'DDD','channels':['EHZ']}
    ]
    bbox_station_service._update_scautopick_config(stations)
    content = cfg_file.read_text()
    assert 'streams' in content
    # HH takes priority over BH
    assert 'AA.BBB.HH?' in content
    assert 'AA.BBB.BH?' not in content
    assert 'CC.DDD.EH?' in content
    # bare station code should not be added when channels exist
    assert 'AA.BBB,' not in content
    # running again shouldn't duplicate the patterns
    bbox_station_service._update_scautopick_config(stations)
    assert content == cfg_file.read_text()


def test_update_scrttv_channels_nslc_format(tmp_path):
    """_update_scrttv_channels must write exactly one 4-part NET.STA.*.CHA? entry
    per station using the highest-priority available band."""
    import os
    from examples import bbox_station_service
    root = tmp_path / "scroot"
    root.mkdir()
    os.environ['SEISCOMP_ROOT'] = str(root)
    (root / 'etc').mkdir(parents=True)
    stations = [
        {'network': 'TX', 'station': 'PB01', 'channels': ['HHZ', 'HHN', 'HHE', 'BHZ']},
        {'network': '4O', 'station': 'AT01', 'channels': ['EHZ', 'BHZ']},
    ]
    bbox_station_service._update_scrttv_channels(stations)
    cfg_path = root / 'etc' / 'scrttv.cfg'
    content = cfg_path.read_text()
    # HH beats BH for TX.PB01
    assert 'TX.PB01.*.HH?' in content
    assert 'TX.PB01.*.BH?' not in content
    # EH beats BH for 4O.AT01
    assert '4O.AT01.*.EH?' in content
    assert '4O.AT01.*.BH?' not in content
    # exactly one entry per station (comma-separated)
    assert content.count('TX.PB01.') == 1
    assert content.count('4O.AT01.') == 1
    # must NOT contain legacy 3-part entries
    assert 'TX.PB01.HH?' not in content
    assert '4O.AT01.EH?' not in content
    # idempotent: calling again should not change content
    bbox_station_service._update_scrttv_channels(stations)
    assert content == cfg_path.read_text()


def test_fix_scrttv_channel_format(tmp_path):
    """fix_scrttv_channel_format should migrate 3-part entries to 4-part and
    collapse multiple band patterns per station down to the highest-priority one."""
    import os
    from examples import bbox_station_service
    root = tmp_path / "scroot"
    root.mkdir()
    os.environ['SEISCOMP_ROOT'] = str(root)
    etc = root / 'etc'
    etc.mkdir()
    cfg_path = etc / 'scrttv.cfg'
    # TX.PB01 has HH? and BH? (3-part); HH? should win
    # 4O.AT01 already 4-part HH? – unchanged
    cfg_path.write_text(
        'streams.codes = TX.PB01.HH?, TX.PB01.BH?, 4O.AT01.*.HH?\n'
    )
    result = bbox_station_service.fix_scrttv_channel_format(dry_run=False)
    assert result['migrated'] == 2   # two 3-part entries converted
    assert result['collapsed'] == 1  # BH? discarded in favour of HH?
    content = cfg_path.read_text()
    # HH? survives; BH? has been collapsed away
    assert 'TX.PB01.*.HH?' in content
    assert 'TX.PB01.*.BH?' not in content
    assert '4O.AT01.*.HH?' in content
    # exactly one entry per station
    assert content.count('TX.PB01.') == 1
    assert content.count('4O.AT01.') == 1
    assert '4O.AT01.*.HH?' in content
    # dry_run should not write
    cfg_path.write_text('streams.codes = TX.OLD.HH?\n')
    r2 = bbox_station_service.fix_scrttv_channel_format(dry_run=True)
    assert r2['dry_run'] is True
    assert r2['migrated'] == 1
    # file should be unchanged
    assert 'TX.OLD.HH?' in cfg_path.read_text()

