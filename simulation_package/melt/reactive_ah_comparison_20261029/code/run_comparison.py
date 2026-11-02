"""Launch a single matched task without changing the study runner or outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hoomd
from common import ROOT, manifest, task_index, require_validation, sha256, output_dir, require_hoomd_build


def command(task, m):
    cmd = [sys.executable, '-u', str(ROOT/'code/matched_runner.py'),
        '--model', task['model'], '--initial-state', task['initial_state'],
        '--epsilon', str(task['epsilon']), '--replicate', str(task['replicate']),
        '--seed', str(task['seed']), '--output-root', str(ROOT/'outputs'/task['model'])]
    for name, value in m['simulation_options'].items():
        cmd += [f'--{name}', str(value)]
    return cmd+['--resume']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    m = manifest()
    task = m['tasks'][task_index(args.task_id)]
    cmd = command(task, m)
    if args.dry_run:
        print(json.dumps(cmd, indent=2))
        return 0
    require_validation()
    assert json.loads((ROOT/'validation/workflow.json').read_text())['status']=='passed'
    require_hoomd_build(hoomd.version.git_sha1, m)
    path = Path(task['initial_state'])
    initial = json.loads(path.with_suffix('.json').read_text())
    assert initial['status'] == 'complete' and initial['sha256'] == sha256(path)
    assert initial['pair'] == m['pairs'][task['pair_id']]
    root = output_dir(task)
    root.mkdir(parents=True, exist_ok=True)
    receipt = {'task': task, 'initial_sha256': initial['sha256'],
               'command': cmd, 'manifest_sha256': sha256(ROOT/'manifest.json')}
    provenance = root/'comparison_input.json'
    if provenance.exists():
        assert json.loads(provenance.read_text()) == receipt
    else:
        provenance.write_text(json.dumps(receipt, indent=2)+'\n')
    print('Running', task, flush=True)
    return subprocess.run(cmd, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
