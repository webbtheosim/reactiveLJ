"""Analyze only completed add-on trajectories and render manuscript-style SVGs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/reactive_ah_mpl_{os.getuid()}')
os.environ.setdefault('XDG_CACHE_HOME', f'/tmp/reactive_ah_cache_{os.getuid()}')
from pathlib import Path
import sys
import warnings
import gsd.hoomd
import numpy as np
from common import ROOT, manifest, output_dir, sha256, require_validation
sys.path.insert(0, str(ROOT/'code/reference'))
from analysis_utils import (MultiTauSetCorrelationAccumulator, compute_r_thresh,
    find_sticker_neighbor_pairs, fit_exponential_semilog_linear_region)
from potentials import weight
from plot_style import plt, setup, format_axes, save, COLORS, MARKERS
from plot_potentials import main as plot_potentials

COORD_BINS = np.r_[np.arange(0, 8.001, .25), np.inf]


def mean_sem(values):
    values = np.asarray(values, float)
    return np.mean(values, axis=0), np.std(values, axis=0, ddof=1)/np.sqrt(len(values))


def turnover_counts(current, previous, previous_open):
    new = np.setdiff1d(current, previous, assume_unique=True)
    n = len(previous_open)
    associative = np.count_nonzero(~previous_open[new//n] | ~previous_open[new % n])
    return int(associative), int(len(new)-associative)


def fit(time, curve):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(fit_exponential_semilog_linear_region(time, curve))


def fit_status(tau, interval, span):
    if not np.isfinite(tau) or tau <= 0:
        return 'unresolved'
    if tau < interval:
        return 'below_saved_interval'
    if tau > span:
        return 'beyond_observed_span'
    return 'within_observed_window'


def check_metadata(task, meta, m):
    expected = dict(m['baseline_configuration'])
    expected.update(model=task['model'], reactive_epsilon=task['epsilon'],
                    replicate=task['replicate'], seed=task['seed'],
                    smooth_elbow=task['model']=='ReactiveLJ',
                    smooth_kappa=.05 if task['model']=='ReactiveLJ' else 0., smooth_beta=1.)
    for key, value in expected.items():
        actual = meta.get(key)
        ok = actual == value if value is None or isinstance(value, (str, bool)) else (
            actual is not None and np.isclose(actual, value, rtol=1.e-10, atol=1.e-12))
        if not ok:
            raise RuntimeError(f'{task}: unexpected {key}={actual}, expected {value}')
    assert meta['run_status']=='complete'
    assert meta['production_completed_steps']==m['simulation_options']['production-steps']


def analyze_task(task):
    m = manifest()
    path = output_dir(task)
    meta = json.loads((path/'metadata.json').read_text())
    check_metadata(task, meta, m)
    receipt = json.loads((path/'comparison_input.json').read_text())
    assert receipt['initial_sha256']==sha256(task['initial_state'])
    assert receipt['manifest_sha256']==sha256(ROOT/'manifest.json')
    trajectory = path/'trajectory.gsd'
    expected_frames = meta['production_steps']//meta['frame_steps']
    interval = meta['dt']*meta['frame_steps']
    correlation = MultiTauSetCorrelationAccumulator(expected_frames-1)
    previous = previous_open = None
    counts = np.zeros(2, dtype=np.int64)
    exposure = 0.
    pc = 1/(meta['stickers_per_chain']-1)
    steps, gelation, paired, distributions, weighted = [], [], [], [], []
    with gsd.hoomd.open(str(trajectory), 'r') as frames:
        if len(frames)!=expected_frames:
            raise RuntimeError(f'{trajectory}: {len(frames)} frames, expected {expected_frames}')
        for frame in frames:
            n = frame.particles.N
            assert n==meta['n_chains']*meta['stickers_per_chain']
            steps.append(int(frame.configuration.step))
            i, j, r = find_sticker_neighbor_pairs(frame.particles.position,
                float(frame.configuration.box[0]), 1.5*meta['reactive_sigma'])
            bonded = r < compute_r_thresh(meta['reactive_sigma'])
            bi, bj = i[bonded], j[bonded]
            degree = np.bincount(np.r_[bi, bj], minlength=n)
            is_open = degree==0
            ids = np.sort(np.minimum(bi, bj).astype(np.int64)*n+np.maximum(bi, bj))
            fraction = float(np.mean(~is_open))
            paired.append(fraction)
            gelation.append((fraction-pc)/pc)
            distributions.append(np.bincount(np.minimum(degree, 6), minlength=7)/n)
            c = np.bincount(np.r_[i, j], weights=np.tile(weight(r), 2), minlength=n)
            weighted.append(np.histogram(c, COORD_BINS)[0]/n)
            correlation.update(ids)
            if previous is not None:
                counts += turnover_counts(ids, previous, previous_open)
                exposure += np.count_nonzero(previous_open)*interval
            previous, previous_open = ids, is_open.copy()
    steps = np.asarray(steps)
    assert np.all(np.diff(steps)==meta['frame_steps'])
    # The frozen runner writes at absolute multiples of frame_steps after production starts.
    start = meta['production_target_final_timestep']-meta['production_steps']
    assert steps[0]==start+meta['frame_steps'] and steps[-1]==meta['production_target_final_timestep']
    length = correlation.valid_length()
    times = correlation.lag_indices[:length]*interval
    cs = correlation.correlation()[:length]
    tau = fit(times, cs)
    span = (steps[-1]-steps[0])*meta['dt']
    rates = counts/exposure if exposure else np.full(2, np.nan)
    row = {k: task[k] for k in ['model','epsilon','replicate','seed','pair_id']}
    row.update(n_frames=len(steps), interval=interval, observed_span=span,
        gelation_mean=float(np.mean(gelation)), phi_pair=float(np.mean(paired)),
        gelation_first_half=float(np.mean(gelation[:len(steps)//2])),
        gelation_second_half=float(np.mean(gelation[len(steps)//2:])),
        rate_assoc=float(rates[0]), rate_dissoc=float(rates[1]),
        associative_events=int(counts[0]), dissociative_events=int(counts[1]),
        open_sticker_time=float(exposure), tau_s=tau,
        tau_status=fit_status(tau, interval, span), trajectory_sha256=sha256(trajectory),
        initial_sha256=receipt['initial_sha256'])
    cache = ROOT/'results/cache'
    cache.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache/f"{task['model']}_eps_{task['epsilon']}_rep_{task['replicate']:03d}.npz",
        row=json.dumps(row), time=times, cs=cs, production_time=(steps-start)*meta['dt'],
        gelation=gelation, phi_pair=paired, degree=np.mean(distributions, axis=0),
        weighted=np.mean(weighted, axis=0))
    print(f"Analyzed {task['model']} eps={task['epsilon']} rep={task['replicate']}", flush=True)
    return row


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(m):
    groups, summaries = {}, []
    all_rows = []
    for model in COLORS:
        for eps in m['epsilon_values']:
            curves = []
            for rep in range(1, m['n_replicates']+1):
                with np.load(ROOT/'results/cache'/f'{model}_eps_{eps}_rep_{rep:03d}.npz') as f:
                    curves.append({k: f[k].copy() for k in f.files})
            rows = [json.loads(str(v['row'])) for v in curves]
            all_rows.extend(rows)
            for c in curves[1:]:
                np.testing.assert_array_equal(c['time'], curves[0]['time'])
                np.testing.assert_array_equal(c['production_time'], curves[0]['production_time'])
            group = {'rows': rows, 'time': curves[0]['time'],
                     'production_time': curves[0]['production_time']}
            summary = {'model': model, 'epsilon': eps, 'n_replicates': len(rows),
                       'interval': rows[0]['interval'], 'observed_span': rows[0]['observed_span']}
            for metric in ['gelation_mean','phi_pair','rate_assoc','rate_dissoc']:
                avg, sem = mean_sem([r[metric] for r in rows])
                summary[metric]=float(avg); summary[metric+'_sem']=float(sem)
            for metric in ['cs','gelation','degree','weighted']:
                data = np.stack([r[metric] for r in curves])
                group[metric], group[metric+'_sem'] = mean_sem(data)
            # Resample entire independent replicate curves, preserving time correlations.
            tau = fit(group['time'], group['cs'])
            rng = np.random.default_rng(20261029+eps)
            cs = np.stack([c['cs'] for c in curves])
            boot = np.array([fit(group['time'], np.mean(cs[rng.integers(0, len(cs), len(cs))], axis=0))
                             for _ in range(500)])
            finite = boot[np.isfinite(boot) & (boot>0)]
            summary.update(tau_s=tau, tau_s_bootstrap_sem=float(np.std(finite, ddof=1)) if len(finite)>1 else np.nan,
                tau_s_bootstrap_n_finite=int(len(finite)),
                tau_status=fit_status(tau, summary['interval'], summary['observed_span']))
            groups[model, eps] = group
            summaries.append(summary)
            write_csv(ROOT/'results'/f'{model}_eps_{eps}_correlation.csv',
                [{'time_tau_LJ':float(t),'C_s':float(c),'sem':float(s)}
                 for t,c,s in zip(group['time'],group['cs'],group['cs_sem'])])
    write_csv(ROOT/'results/replicate_results.csv', all_rows)
    write_csv(ROOT/'results/summary.csv', summaries)
    # Paired contrasts are calculated from replicate differences, not from pooled frames.
    differences = []
    for eps in m['epsilon_values']:
        lj, ah = groups['ReactiveLJ',eps]['rows'], groups['ReactiveAH',eps]['rows']
        assert all(a['initial_sha256']==b['initial_sha256'] for a,b in zip(lj,ah))
        row = {'epsilon':eps, 'n_pairs':len(lj)}
        for metric in ['gelation_mean','phi_pair','rate_assoc','rate_dissoc']:
            avg, sem=mean_sem([a[metric]-b[metric] for a,b in zip(ah,lj)])
            row[metric+'_AH_minus_LJ']=float(avg)
            row[metric+'_paired_sem']=float(sem)
        differences.append(row)
    write_csv(ROOT/'results/paired_differences.csv',differences)
    return groups, summaries


def errorbar(ax, x, y, sem, model, **kwargs):
    ax.errorbar(x, y, yerr=sem, color=COLORS[model], marker=MARKERS[model],
                markersize=3.5, capsize=2, elinewidth=.7, markeredgewidth=.7,
                label=model, **kwargs)


def plot_results(groups, summaries, m):
    setup()
    out = ROOT/'results/figures'
    out.mkdir(exist_ok=True)
    epsilons = m['epsilon_values']
    categories = np.arange(len(epsilons))
    bar_width = .36
    tau_r = m['tau_R0']
    from matplotlib.ticker import LogLocator, NullFormatter
    fig, axes = plt.subplots(1,3,figsize=(7,2.8))
    fig.subplots_adjust(left=.08,right=.985,bottom=.23,top=.80,wspace=.48)
    for model_index, model in enumerate(COLORS):
        rows = [next(r for r in summaries if r['model']==model and r['epsilon']==eps)
                for eps in epsilons]
        positions = categories + (model_index-.5)*bar_width
        for ax,key in zip(axes,['gelation_mean','rate_assoc','rate_dissoc']):
            scale=1 if key=='gelation_mean' else tau_r
            ax.bar(positions,np.array([r[key] for r in rows])*scale,width=bar_width,
                yerr=np.array([r[key+'_sem'] for r in rows])*scale,
                color=COLORS[model],edgecolor='black',linewidth=.5,label=model,
                capsize=2,error_kw={'elinewidth':.7,'capthick':.7})
    for i,(ax,title) in enumerate(zip(axes,['Degree of gelation','Associative-type','Dissociative-type'])):
        ax.set(xticks=categories,xticklabels=epsilons,
               xlabel=r'$\varepsilon/\varepsilon_0$',title=title)
        format_axes(ax,chr(97+i))
    axes[0].axhline(1,color='.4',linestyle='--',linewidth=.7)
    axes[0].set(ylabel=r'$\mathcal{E}$',ylim=(0.,2.1))
    axes[0].yaxis.label.set_math_fontfamily('cm')
    for ax,key in zip(axes[1:],['rate_assoc','rate_dissoc']):
        ax.set_ylabel(r'$\nu_\alpha\tau_R^{(0)}$')
        values=np.array([r[key] for r in summaries])*tau_r
        if np.all(values>0):
            low=10**np.floor(np.log10(values.min()/1.5))
            high=10**np.ceil(np.log10(values.max()*1.5))
            ax.set(yscale='log',ylim=(low,high))
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            # Keep exact zero rates visible rather than dropping them on a log axis.
            ax.set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(),frameon=False,
               loc='upper center',bbox_to_anchor=(.5,1.015),ncol=2)
    save(fig,out/'gelation_turnover')

    for kind in ['gelation_time','pair_persistence','sticker_coordination','weighted_coordination']:
        fig,axes=plt.subplots(1,3,figsize=(7,2.9),sharey=True)
        fig.subplots_adjust(left=.10,right=.985,bottom=.23,top=.79,wspace=.18)
        for i,(ax,eps) in enumerate(zip(axes,epsilons)):
            for model in COLORS:
                g=groups[model,eps]
                if kind=='gelation_time':
                    x,y,se=g['production_time']/tau_r,g['gelation'],g['gelation_sem']
                elif kind=='pair_persistence':
                    x,y,se=g['time']/tau_r,g['cs'],g['cs_sem']
                elif kind=='sticker_coordination':
                    x,y,se=np.arange(7),g['degree'],g['degree_sem']
                else:
                    # Final overflow bin (C>=8) is stored in the cache and shown at 8.25.
                    x=np.r_[(COORD_BINS[:-2]+COORD_BINS[1:-1])/2,8.25]
                    y,se=g['weighted'],g['weighted_sem']
                if kind in ['gelation_time','pair_persistence']:
                    ax.plot(x,y,color=COLORS[model],label=model)
                    ax.fill_between(x,np.maximum(y-se,1.e-10) if kind=='pair_persistence' else y-se,
                                    y+se,color=COLORS[model],alpha=.2,linewidth=0)
                else:
                    errorbar(ax,x,y,se,model)
            ax.set_title(rf'$\varepsilon/\varepsilon_0={eps}$')
            format_axes(ax,chr(97+i))
            if kind=='gelation_time': ax.set(xlabel=r'$t/\tau_R^{(0)}$',ylim=(-1.1,2.1))
            elif kind=='pair_persistence':
                positive=np.concatenate([g['cs'][g['cs']>0] for g in groups.values()])
                lower=10**np.floor(np.log10(positive.min())) if len(positive) else 1.e-6
                ax.set(xlabel=r'$t/\tau_R^{(0)}$',yscale='log',ylim=(lower,1.1))
            elif kind=='sticker_coordination':
                ax.set(xlabel='Bonded neighbors',xticks=[0,1,2,3,4,5,6],
                       xticklabels=['0','1','2','3','4','5','6+'],ylim=(-.02,1.02))
            else: ax.set(xlabel=r'Weighted coordination, $C_i$',xlim=(-.1,8.5))
        axes[0].set_ylabel({'gelation_time':r'Degree of gelation, $\mathcal{E}$',
            'pair_persistence':r'Pair correlation, $C_s(t)$',
            'sticker_coordination':'Sticker fraction',
            'weighted_coordination':'Fraction per bin'}[kind])
        if kind=='gelation_time':
            axes[0].yaxis.label.set_math_fontfamily('cm')
        fig.legend(*axes[0].get_legend_handles_labels(),frameon=False,
                   loc='upper center',bbox_to_anchor=(.5,1.015),ncol=2)
        save(fig,out/kind)

    fig,ax=plt.subplots(figsize=(3.5,3.2))
    fig.subplots_adjust(left=.23,right=.97,bottom=.18,top=.76)
    for model_index, model in enumerate(COLORS):
        rows=[r for r in summaries if r['model']==model]
        for row in rows:
            if not np.isfinite(row['tau_s']) or row['tau_s']<=0: continue
            status=row['tau_status']
            position=epsilons.index(row['epsilon'])+(model_index-.5)*bar_width
            ax.bar(position,row['tau_s']/tau_r,width=bar_width,
                yerr=row['tau_s_bootstrap_sem']/tau_r,color=COLORS[model],
                edgecolor='black',linewidth=.5,capsize=2,
                error_kw={'elinewidth':.7,'capthick':.7})
            if status!='within_observed_window':
                marker='v' if status=='below_saved_interval' else '^'
                ax.plot(position,row['tau_s']/tau_r,marker=marker,linestyle='none',
                        color='black',markerfacecolor='white',markersize=4,zorder=5)
    ax.axhline(500/tau_r,color='.4',ls=':',lw=.7)
    ax.axhline(49500/tau_r,color='.4',ls='--',lw=.7)
    ax.set(xlabel=r'$\varepsilon/\varepsilon_0$',ylabel=r'Fitted persistence, $\tau_s/\tau_R^{(0)}$',
           xticks=categories,xticklabels=epsilons,yscale='log')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLORS[m],edgecolor='black',linewidth=.5,label=m) for m in COLORS],
              frameon=False,loc='lower center',bbox_to_anchor=(.5,1.04))
    format_axes(ax)
    save(fig,out/'persistence_times')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    m=manifest()
    if not args.plots_only:
        require_validation()
    (ROOT/'results').mkdir(exist_ok=True)
    if not args.plots_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(analyze_task,m['tasks']))
    groups,summaries=summarize(m)
    plot_results(groups,summaries,m)
    plot_potentials(ROOT/'results/figures')
    (ROOT/'results/completion.json').write_text(json.dumps({
        'status':'complete','n_simulations':len(m['tasks']),
        'manifest_sha256':sha256(ROOT/'manifest.json'),
        'uncertainty':'SEM over four independent replicate means or curves; persistence uses 500 replicate bootstrap draws.',
        'gelation':'(phi_pair - 1/(f-1))/(1/(f-1)), f=4; manuscript mean-field indicator.',
        'coordination':'Integer bonded-neighbor counts and smooth Ci are both reported. Histograms average framewise fractions within each replicate.',
        'turnover':'Saved-frame apparent rates, normalized by prior open-sticker exposure; unresolved intermediate events are not counted.',
        'persistence':'Endpoint pair-state correlation. Hollow down/up triangles denote fits below the 500 tau saved interval or above the 49,500 tau observed span.',
        'energy_landscape':'Total three-sticker energy relative to the isolated AB dimer plus a separated C. Energies divided by epsilon; identical normalized maps at all three epsilon values.',
        'figures':sorted(p.name for p in (ROOT/'results/figures').glob('*.svg')),
        },indent=2)+'\n')
    print('Complete; SVGs saved in', ROOT/'results/figures',flush=True)


if __name__=='__main__': main()
