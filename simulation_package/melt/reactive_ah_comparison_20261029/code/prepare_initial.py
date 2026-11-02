"""Save one shared unsticky melt state for each AH/LJ replicate pair."""
import argparse
import json
import os
from pathlib import Path
import sys
import hoomd
from common import ROOT, manifest, task_index, require_validation, sha256, require_hoomd_build
sys.path.insert(0, str(ROOT/'code/reference'))
import run_reactive_lj as reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-id', type=int)
    args = parser.parse_args()
    require_validation()
    m = manifest()
    require_hoomd_build(hoomd.version.git_sha1, m)
    task = m['pairs'][task_index(args.pair_id)]
    cfg = reference.SimulationConfig()
    for name, value in m['baseline_configuration'].items():
        setattr(cfg, name, value)
    path = Path(task['initial_state'])
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix('.json')
    if path.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        assert old['sha256'] == sha256(path) and old['pair'] == task
        print(f'Shared initial state already complete: {path}', flush=True)
        return
    if path.exists():
        raise RuntimeError(f'{path} has no provenance record')
    sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=task['seed'])
    L = reference.compute_box_length(cfg.n_chains*cfg.chain_length, cfg.density)
    sim.create_state_from_snapshot(reference.build_snapshot(cfg, task['seed'], L))
    nlist = hoomd.md.nlist.Cell(buffer=cfg.nlist_buffer)
    sim.operations.integrator = reference.build_integrator(cfg, nlist, reactive_lj_enabled=True)
    sim.operations.updaters.append(hoomd.md.update.ZeroMomentum(
        hoomd.trigger.Periodic(cfg.zero_momentum_period)))
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=cfg.temperature)
    sim.run(cfg.unsticky_equil_steps)
    temporary = path.with_suffix('.partial.gsd')
    hoomd.write.GSD.write(state=sim.state, filename=str(temporary), mode='wb')
    os.replace(temporary, path)
    receipt.write_text(json.dumps({'pair': task, 'sha256': sha256(path),
        'step': int(sim.timestep), 'status': 'complete',
        'hoomd_git_sha1': hoomd.version.git_sha1}, indent=2)+'\n')
    print(f'Saved shared initial state: {path}', flush=True)


if __name__ == '__main__':
    main()
