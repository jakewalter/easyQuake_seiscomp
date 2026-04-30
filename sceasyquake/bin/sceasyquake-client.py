#!/usr/bin/env python3
"""Simple client to create job YAMLs for the sceasyquake worker.

Modes:
- CLI mode: write a job YAML with specified function and params
- (Optional) Messaging mode: subscribe to SeisComP messages and create jobs automatically
"""

import argparse
import os
import yaml
import json

try:
    # SeisComP 7 Python bindings expose the `seiscomp` package. Use the
    # Client submodule (Application class) for messaging integration.
    import seiscomp.Client as sc_client  # type: ignore
    HAS_SC = True
except Exception:
    HAS_SC = False


def write_job(job_dir, function, params):
    os.makedirs(job_dir, exist_ok=True)
    ts = int(__import__('time').time())
    fname = os.path.join(job_dir, f'job_{function}_{ts}.yaml')
    with open(fname, 'w') as f:
        yaml.safe_dump({'function': function, 'params': params}, f)
    print(f'Wrote job to {fname}')


class SCClient:
    def __init__(self, job_dir, subscribe='origin'):
        self.job_dir = job_dir
        self.subscribe = subscribe

    def start(self):
        if not HAS_SC:
            raise RuntimeError('SeisComP python bindings not available')
        # Minimal client: subscribe to origins and create a job to run association
        # Implementing a full SeisComP client is out of scope for the MVP; this is
        # a placeholder showing where to attach code that listens to the messaging
        # system and writes job files.
        print('SeisComP bindings are available; a real implementation would subscribe to messages here.')


def main():
    parser = argparse.ArgumentParser(description='sceasyquake client - create jobs for easyQuake')
    sub = parser.add_subparsers(dest='cmd')

    cli = sub.add_parser('job', help='Create a job YAML from CLI')
    cli.add_argument('--job-dir', required=True)
    cli.add_argument('--function', required=True, choices=['detection_continuous', 'detection_association_event'])
    cli.add_argument('--params', help='JSON string with parameters for the easyQuake function')
    cli.add_argument('--params-file', help='Path to JSON or YAML file with params')

    listen = sub.add_parser('listen', help='(Optional) Subscribe to SeisComP messaging to generate jobs')
    listen.add_argument('--job-dir', required=True)
    listen.add_argument('--subscribe', default='origin')

    args = parser.parse_args()

    if args.cmd == 'job':
        if args.params_file:
            with open(args.params_file) as f:
                try:
                    params = json.load(f)
                except Exception:
                    import yaml as _yaml
                    params = _yaml.safe_load(f)
        elif args.params:
            params = json.loads(args.params)
        else:
            params = {}
        write_job(args.job_dir, args.function, params)

    elif args.cmd == 'listen':
        if not HAS_SC:
            print('SeisComP python bindings are not available; listen mode requires them.')
            return
        client = SCClient(args.job_dir, subscribe=args.subscribe)
        client.start()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
