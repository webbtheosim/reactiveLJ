"""Validate the add-on force before the matched melt array may run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import hoomd
from reactive_ah import ReactiveAH
from potentials import total_energy, pair_energy, weakening, RM

ROOT = Path(__file__).resolve().parents[1]


def make_sim(pos, model='ReactiveAH', device='cpu', p=4., epsilon=3., types=None):
    dev = hoomd.device.GPU() if device == 'gpu' else hoomd.device.CPU()
    sim = hoomd.Simulation(device=dev, seed=113)
    snap = hoomd.Snapshot()
    snap.configuration.box = [12, 12, 12, 0, 0, 0]
    snap.particles.N = len(pos)
    snap.particles.types = ['sticky', 'inert']
    snap.particles.position[:] = pos
    if types is not None:
        snap.particles.typeid[:] = types
    sim.create_state_from_snapshot(snap)
    cls = ReactiveAH if model == 'ReactiveAH' else hoomd.md.many_body.ReactiveLJ
    force = cls(hoomd.md.nlist.Cell(buffer=.3, exclusions=[]), 'sticky', 1., epsilon,
                weakening_exponent=p)
    sim.operations.integrator = hoomd.md.Integrator(dt=.0005,
        methods=[hoomd.md.methods.ConstantVolume(hoomd.filter.All())], forces=[force])
    sim.always_compute_pressure = True
    sim.run(0)
    return sim, force


def check_close(actual, expected, *, rtol=2.e-4, atol=2.e-4):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    devices = ['cpu', 'gpu'] if args.gpu else ['cpu']
    report = {'status': 'running', 'devices': devices, 'checks': []}
    # Probe both AH branches, the coordination taper, mixed types and PBC.
    cases = [np.array([[0., 0, 0], [1.05, 0, 0], [.2, 1.36, .1]]),
             np.array([[0., 0, 0], [1.22, 0, 0], [.35, 1.31, .15], [1.6, 1.1, .2]]),
             np.array([[5.7, 0, 0], [-5.25, 0, 0], [5.9, 1.36, .1]])]
    reference_results = {}
    for device in devices:
        for model in ['ReactiveAH', 'ReactiveLJ']:
            for idx, pos in enumerate(cases):
                types = np.zeros(len(pos), dtype=int)
                if idx == 1:
                    types[-1] = 1
                sim, force = make_sim(pos, model, device, types=types)
                energy = float(force.energy)
                check_close(energy, total_energy(pos, model, epsilon=3, types=types),
                            rtol=2.e-6, atol=2.e-5)
                h = 2.e-5
                numerical = np.zeros_like(pos)
                for i in range(len(pos)):
                    for d in range(3):
                        plus, minus = pos.copy(), pos.copy()
                        plus[i, d] += h
                        minus[i, d] -= h
                        numerical[i, d] = -(total_energy(plus, model, epsilon=3, types=types)
                                            -total_energy(minus, model, epsilon=3, types=types))/(2*h)
                check_close(force.forces, numerical)
                check_close(force.forces.sum(axis=0), 0, atol=2.e-5)
                # Diagonal virials equal minus the derivative under homogeneous strain.
                virial = force.virials.sum(axis=0)
                for d, component in enumerate([0, 3, 5]):
                    plus, minus = pos.copy(), pos.copy()
                    plus[:, d] *= 1+h
                    minus[:, d] *= 1-h
                    bp, bm = np.full(3, 12.), np.full(3, 12.)
                    bp[d] *= 1+h; bm[d] *= 1-h
                    numerical_virial = -(total_energy(plus, model, epsilon=3, types=types, box=bp)
                                         -total_energy(minus, model, epsilon=3, types=types, box=bm))/(2*h)
                    check_close(virial[component], numerical_virial)
                key = f'{model}_{idx}'
                values = np.r_[energy, force.forces.ravel(), virial]
                if device == 'cpu':
                    reference_results[key] = values
                else:
                    check_close(values, reference_results[key])
                report['checks'].append(f'{device} {key}: energy, force, momentum, virial')
                del sim, force
        # Exact p=0 LJ limit, including a crowded configuration.
        sim1, f1 = make_sim(cases[0], 'ReactiveAH', device, p=0)
        sim2, f2 = make_sim(cases[0], 'ReactiveLJ', device, p=0)
        check_close(f1.energy, f2.energy, rtol=2.e-6, atol=2.e-5)
        check_close(f1.forces, f2.forces)
        report['checks'].append(f'{device}: p=0 agrees with installed ReactiveLJ')
        del sim1, sim2, f1, f2
        # Bounded dimer trajectory probes a stable minimum without outer-cutoff crossings.
        sim, force = make_sim([[0., 0, 0], [RM+.01, 0, 0]], device=device, epsilon=18.)
        thermo = hoomd.md.compute.ThermodynamicQuantities(hoomd.filter.All())
        sim.operations.computes.append(thermo)
        sim.run(0)
        e0 = float(force.energy+thermo.kinetic_energy)
        deviations = []
        for _ in range(10):
            sim.run(500)
            deviations.append(abs(float(force.energy+thermo.kinetic_energy)-e0))
        assert max(deviations) < 2.e-4, deviations
        report['checks'].append(f'{device}: NVE dimer max absolute drift {max(deviations):.6g}')
        del sim, force
    W = weakening(np.array([0., 1., 4.]))
    check_close(pair_energy(RM-1.e-8, W, 'ReactiveAH'),
                pair_energy(RM+1.e-8, W, 'ReactiveAH'), rtol=1.e-10, atol=1.e-12)
    report['isolated_regularization_W'] = float(W[0])
    report['checks'].append('Energy joins continuously at the LJ minimum')
    from common import verify_sources
    verify_sources()
    report['checks'].append('Published force and frozen reference files unchanged')
    (ROOT/'validation').mkdir(exist_ok=True)
    report['status'] = 'passed'
    report['extension_sha256'] = hashlib.sha256(next((ROOT/'code').glob('_reactive_ah*.so')).read_bytes()).hexdigest()
    path = ROOT/'validation'/('force_cpu_gpu.json' if args.gpu else 'force_cpu.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
