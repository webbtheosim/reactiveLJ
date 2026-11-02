"""Exercise the complete GPU runner, restart, and GSD analysis on a small melt."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import hoomd
import gsd.hoomd
from common import ROOT, manifest, require_validation, sha256
sys.path.insert(0, str(ROOT/'code/reference'))
import run_reactive_lj as reference
import analyze_comparison as analysis


def main():
    require_validation()
    testroot = ROOT/'validation/workflow'
    testroot.mkdir(exist_ok=True)
    cfg=reference.SimulationConfig(n_chains=20, unsticky_equil_steps=2000,
        reactive_equil_steps=20000, sticky_burnin_steps=2000, production_steps=10000,
        frame_steps=1000, virial_log_steps=1000, msd_particles=1)
    initial=testroot/'initial.gsd'
    if not initial.exists():
        sim=hoomd.Simulation(device=hoomd.device.GPU(),seed=301)
        L=reference.compute_box_length(cfg.n_chains*cfg.chain_length,cfg.density)
        sim.create_state_from_snapshot(reference.build_snapshot(cfg,301,L))
        sim.operations.integrator=reference.build_integrator(cfg,hoomd.md.nlist.Cell(buffer=.4),True)
        sim.operations.updaters.append(hoomd.md.update.ZeroMomentum(hoomd.trigger.Periodic(100)))
        sim.state.thermalize_particle_momenta(hoomd.filter.All(),kT=1.)
        sim.run(2000)
        hoomd.write.GSD.write(state=sim.state,filename=str(initial),mode='wb')
        del sim
    small=manifest()
    for key in list(small['baseline_configuration']):
        small['baseline_configuration'][key]=getattr(cfg,key)
    small['simulation_options']['production-steps']=10000
    (testroot/'manifest.json').write_text(json.dumps(small,indent=2)+'\n')
    rows=[]
    for model in ['ReactiveLJ','ReactiveAH']:
        out=testroot/model/'eps_18/rep_001'
        cmd=[sys.executable,'-u',str(ROOT/'code/matched_runner.py'), '--model',model,
            '--initial-state',str(initial),'--epsilon','18','--replicate','1','--seed','301',
            '--output-root',str(testroot/model),'--n-chains','20','--unsticky-equil-steps','2000',
            '--reactive-equil-steps','20000','--sticky-burnin-steps','2000',
            '--production-steps','10000','--frame-steps','1000','--virial-log-steps','1000',
            '--msd-particles','1','--production-chunk-steps','2000','--resume']
        subprocess.run(cmd,check=True)
        # Re-entering a finished checkpoint must neither rerun nor append duplicate frames.
        digest=sha256(out/'trajectory.gsd')
        subprocess.run(cmd,check=True)
        assert sha256(out/'trajectory.gsd')==digest
        task={'model':model,'epsilon':18,'replicate':1,'seed':301,'pair_id':0,
              'initial_state':str(initial)}
        (out/'comparison_input.json').write_text(json.dumps({
            'initial_sha256':sha256(initial),'manifest_sha256':sha256(testroot/'manifest.json')}))
        analysis.manifest=lambda: small
        analysis.output_dir=lambda task: out
        analysis.ROOT=testroot
        row=analysis.analyze_task(task)
        assert -1<=row['gelation_mean']<=2
        assert row['n_frames']==10 and np.isfinite(row['rate_assoc'])
        with gsd.hoomd.open(str(out/'checkpoint.gsd'),'r') as f:
            assert np.all(np.isfinite(f[-1].particles.position))
            assert np.all(np.isfinite(f[-1].particles.velocity))
        rows.append(row)
    report={'status':'passed','checks':['Shared initial melt, both GPU force definitions',
        'Full force ramp and timestep ramp, including dt=0.005 at epsilon=18',
        'Production checkpoint and completed-run restart',
        'Saved GSD frame count, analysis metadata, gelation and turnover'], 'smoke_results':rows}
    (ROOT/'validation/workflow.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
