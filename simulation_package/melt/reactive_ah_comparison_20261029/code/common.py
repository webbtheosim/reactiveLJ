"""Campaign paths, immutable input checks and task selection."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    m = json.loads((ROOT/'manifest.json').read_text())
    m['root'] = str(ROOT)
    for key in ('pairs', 'tasks'):
        for task in m[key]:
            task['initial_state'] = str(ROOT / task['initial_state'])
    return m


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def task_index(value):
    return value if value is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])


def require_validation():
    report = json.loads((ROOT/'validation/force_cpu_gpu.json').read_text())
    assert report['status'] == 'passed' and 'gpu' in report['devices']
    assert sha256(next((ROOT/'code').glob('_reactive_ah*.so'))) == report['extension_sha256']
    verify_sources()


def verify_sources():
    """Check the published force and frozen reference files before production."""
    for item in json.loads((ROOT/'source_checksums.json').read_text()):
        if sha256(ROOT / item['path']) != item['sha256']:
            raise RuntimeError(f"Reference input changed: {item['path']}")


def output_dir(task):
    return ROOT/'outputs'/task['model']/f"eps_{task['epsilon']:g}"/f"rep_{task['replicate']:03d}"
