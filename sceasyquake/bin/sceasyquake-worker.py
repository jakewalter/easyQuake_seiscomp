#!/usr/bin/env python3
"""Simple worker that watches a jobs directory for YAML jobs and runs easyQuake."""

import argparse
import time
import yaml
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def run_job(job_path, python_cmd='python'):
    with open(job_path) as f:
        job = yaml.safe_load(f)

    func = job.get('function')
    params = job.get('params', {})

    # For robustness, we run in a subprocess that imports easyQuake and calls
    # the requested function. This avoids keeping heavy ML libs loaded in the
    # worker process and keeps separation from SeisComP client.
    script = f"""
import yaml
from easyQuake import *
import json
job = {params!r}
# call the selected function
if '{func}' == 'detection_continuous':
    detection_continuous(**job)
elif '{func}' == 'detection_association_event':
    detection_association_event(**job)
else:
    raise SystemExit('Unknown function: {func}')
"""

    proc = subprocess.run([python_cmd, '-c', script], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Job {job_path} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    else:
        print(f"Job {job_path} completed. Output:\n{proc.stdout}")

    # move job to processed
    dst = job_path + '.done'
    os.rename(job_path, dst)


class JobHandler(FileSystemEventHandler):
    def __init__(self, python_cmd):
        self.python_cmd = python_cmd

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.yaml') or event.src_path.endswith('.yml'):
            print(f'New job file: {event.src_path}')
            run_job(event.src_path, python_cmd=self.python_cmd)


def main():
    parser = argparse.ArgumentParser(description='sceasyquake worker')
    parser.add_argument('--job-dir', required=True, help='Directory to watch for job YAML files')
    parser.add_argument('--python-cmd', default='python', help='Python command to use (e.g., conda env activation wrapper)')
    args = parser.parse_args()

    if not os.path.isdir(args.job_dir):
        os.makedirs(args.job_dir, exist_ok=True)

    event_handler = JobHandler(args.python_cmd)
    observer = Observer()
    observer.schedule(event_handler, args.job_dir, recursive=False)
    observer.start()
    print(f'Watching {args.job_dir} for jobs...')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()
