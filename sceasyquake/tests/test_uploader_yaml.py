import os
import tempfile
from sceasyquake.uploader import PickUploader


def test_send_pick_writes_yaml(tmp_path):
    out = tmp_path / 'picks'
    pu = PickUploader(out_dir=str(out), source='test')
    res = pu.send_pick(network='XX', station='STA', location='', channel='BHZ', time='2025-12-14T00:00:00', probability=0.9)
    assert os.path.exists(res)
    with open(res) as f:
        s = f.read()
    assert 'station' in s
    assert 'probability' in s
