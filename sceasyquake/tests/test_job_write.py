import os
import tempfile
import yaml
from pathlib import Path

from subprocess import run, PIPE


def test_write_job(tmp_path):
    job_dir = tmp_path / 'jobs'
    job_dir.mkdir()
    cmd = ['python', 'sceasyquake/bin/sceasyquake-client.py', 'job', '--job-dir', str(job_dir), '--function', 'detection_continuous', '--params', '{"dirname":"20250101"}']
    p = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    assert p.returncode == 0
    files = list(job_dir.glob('*.yaml'))
    assert len(files) == 1
    with open(files[0]) as f:
        job = yaml.safe_load(f)
    assert job['function'] == 'detection_continuous'
    assert job['params']['dirname'] == '20250101'
