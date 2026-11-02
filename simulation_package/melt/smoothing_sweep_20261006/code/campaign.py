"""Resolve campaign paths relative to the manifest, never the working directory."""
import json
from pathlib import Path


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    for key in ('manifest_path', 'runner_snapshot', 'output_root', 'analysis_root'):
        data[key] = str(path.parent / data[key])
    for entry in data['source_snapshots']:
        entry['snapshot'] = str(path.parent / entry['snapshot'])
    return data
