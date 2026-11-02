"""Independent, static three-sticker barrier check for the manuscript potential.

No production or manuscript files are modified. Distances, energies and forces
are checked against the installed HOOMD ReactiveLJ force before path searches.
The search uses all three pair distances, not a fixed-dimer approach path.
"""
from pathlib import Path
import argparse, gc, hashlib, json, os, sys, time
import numpy as np
from scipy import ndimage, optimize
from numba import njit

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
RM = 2.0**(1.0/6.0)
RC = 1.5
SHIFT = 4.0*(RC**-12-RC**-6)
DEPTH = 1.0+SHIFT
ELBOW = ((1.0+np.sqrt(1.0+SHIFT))/2.0)**(-1.0/6.0)

@njit(cache=True)
def weight(r, inner):
    if r >= RC: return 0.0
    if r <= inner: return 1.0
    return 0.5*(1.0+np.cos(np.pi*(r-inner)/(RC-inner)))

@njit(cache=True)
def lj(r):
    inv6=r**-6
    return 4.0*(inv6*inv6-inv6)-SHIFT

@njit(cache=True)
def dlj(r):
    return 24.0*(-2.0*r**-13+r**-7)

@njit(cache=True)
def pair(r, raw, p, kappa=0.05, beta=1.0):
    if r >= RC: return 0.0
    excess=0.5*(raw+np.sqrt(raw*raw+1.e-12))
    w=(1.0+excess)**(-p)
    u=lj(r)
    ans=max(u,0.0)+w*min(u,0.0)
    delta=kappa*(1.0-w)**beta
    r1=ELBOW-delta; r2=ELBOW+delta
    if delta>1.e-6 and 2.0*delta>1.e-6 and r1<r<r2:
        t=(r-r1)/(2.0*delta)
        h00=2*t**3-3*t*t+1; h10=t**3-2*t*t+t
        h01=-2*t**3+3*t*t; h11=t**3-t*t
        ans=h00*lj(r1)+h10*(2*delta)*dlj(r1)+h01*w*lj(r2)+h11*(2*delta)*w*dlj(r2)
    return ans

@njit(cache=True)
def energy(x,p,inner,kappa=0.05,beta=1.0):
    a,b,c=x
    wa=weight(a,inner); wb=weight(b,inner); wc=weight(c,inner)
    return pair(a,wb+wc,p,kappa,beta)+pair(b,wa+wc,p,kappa,beta)+pair(c,wa+wb,p,kappa,beta)

@njit(cache=True)
def energy_grid(axis,p,inner,kappa=0.05,beta=1.0):
    n=len(axis); out=np.empty((n,n,n),np.float64)
    ws=np.array([weight(r,inner) for r in axis])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                out[i,j,k]=(pair(axis[i],ws[j]+ws[k],p,kappa,beta)
                           +pair(axis[j],ws[i]+ws[k],p,kappa,beta)
                           +pair(axis[k],ws[i]+ws[j],p,kappa,beta))
    return out

@njit(cache=True)
def path_bfs(e, threshold, source, target):
    n=e.shape[0]; size=n*n*n
    parent=np.full(size,-1,np.int32)
    queue=np.empty(size,np.int32)
    start=source[0]*n*n+source[1]*n+source[2]
    goal=target[0]*n*n+target[1]*n+target[2]
    queue[0]=start; parent[start]=start; head=0; tail=1
    flat=e.ravel()
    while head<tail and parent[goal]<0:
        u=queue[head];head+=1
        i=u//(n*n); j=(u//n)%n; k=u%n
        for step in range(6):
            v=-1
            if step==0 and i>0: v=u-n*n
            if step==1 and i+1<n: v=u+n*n
            if step==2 and j>0: v=u-n
            if step==3 and j+1<n: v=u+n
            if step==4 and k>0: v=u-1
            if step==5 and k+1<n: v=u+1
            if v>=0 and parent[v]<0 and flat[v]<=threshold:
                parent[v]=u;queue[tail]=v;tail+=1
    if parent[goal]<0:raise ValueError('No connected path')
    count=1;u=goal
    while u!=start: count+=1;u=parent[u]
    path=np.empty((count,3),np.int32);u=goal
    for q in range(count-1,-1,-1):
        path[q,0]=u//(n*n);path[q,1]=(u//n)%n;path[q,2]=u%n
        u=parent[u]
    return path

def positions(x):
    a,b,c=np.asarray(x,float)
    kx=(a*a+b*b-c*c)/(2*a)
    ky=np.sqrt(max(0.0,b*b-kx*kx))
    return np.array([[0.,0.,0.],[a,0.,0.],[kx,ky,0.]])

def distances(pos):
    return np.array([np.linalg.norm(pos[0]-pos[1]),np.linalg.norm(pos[0]-pos[2]),np.linalg.norm(pos[1]-pos[2])])

def validate():
    import hoomd
    rng=np.random.default_rng(142)
    records=[]
    for p in [0.,.6,1.,2.,4.,8.]:
        for inner in [1.15,1.2,1.3]:
            probes=[np.array([RM,1.5,1.5]),np.array([ELBOW,1.36,1.39]),rng.uniform(1.04,1.48,3)]
            for case,x in enumerate(probes):
                pos=positions(x)
                device=hoomd.device.CPU();device.notice_level=0
                sim=hoomd.Simulation(device=device,seed=113)
                snap=hoomd.Snapshot();snap.configuration.box=[12,12,12,0,0,0]
                snap.particles.N=3;snap.particles.types=['sticky'];snap.particles.position[:]=pos
                sim.create_state_from_snapshot(snap)
                force=hoomd.md.many_body.ReactiveLJ(hoomd.md.nlist.Cell(buffer=.3,exclusions=[]),
                    'sticky',1.,1.,r_cut=RC,weakening_inner=inner,weakening_outer=RC,
                    weakening_exponent=p,smooth_elbow=True,smooth_kappa=.05,smooth_beta=1.)
                sim.operations.integrator=hoomd.md.Integrator(dt=.0005,
                    methods=[hoomd.md.methods.ConstantVolume(hoomd.filter.All())],forces=[force])
                sim.run(0)
                predicted=float(energy(x,p,inner));actual=float(force.energy)
                error=abs(predicted-actual)
                np.testing.assert_allclose(predicted,actual,rtol=3.e-11,atol=3.e-11)
                record={'p':p,'inner':inner,'case':case,'distances':x.tolist(),'energy':actual,'energy_abs_error':error}
                if case==2:
                    num=np.zeros_like(pos);h=2.e-6
                    for i in range(3):
                        for d in range(3):
                            plus=pos.copy();minus=pos.copy();plus[i,d]+=h;minus[i,d]-=h
                            num[i,d]=-(energy(distances(plus),p,inner)-energy(distances(minus),p,inner))/(2*h)
                    record['force_max_abs_error']=float(np.max(np.abs(force.forces-num)))
                    np.testing.assert_allclose(force.forces,num,rtol=3.e-6,atol=3.e-7)
                records.append(record)
                del sim,force,device
    # The lower boundary excludes configurations whose repulsion exceeds the
    # greatest possible attraction of the two remaining pairs. Each pair's
    # smoothed branch is additionally sampled to check that it stays above -U.
    extrema=[]
    for p in [0.,.6,.8,1.,1.5,2.,4.,8.]:
        emin=min(pair(r,raw,p) for r in np.linspace(.94,1.5,801) for raw in np.linspace(0,2,101))
        extrema.append({'p':p,'min_pair_energy':float(emin)})
        assert emin>=-DEPTH-1.e-10
    assert lj(.95)-2*DEPTH>0
    report={'status':'passed','hoomd_version':hoomd.version.version,'cases':records,
            'max_energy_abs_error':max(r['energy_abs_error'] for r in records),
            'max_force_abs_error':max(r.get('force_max_abs_error',0) for r in records),
            'pair_minimum_checks':extrema,'repulsive_boundary_lower_bound':float(lj(.95)-2*DEPTH)}
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HOOMD validation passed',len(records),'energy cases; max error',report['max_energy_abs_error'],flush=True)

def grid_search(p,inner,h,save_path=False):
    start_time=time.perf_counter()
    # Insert the exact dimer minimum and coordination-window boundary.
    axis=np.unique(np.r_[np.linspace(.95,RC,round((RC-.95)/h)+1),RM,inner])
    n=len(axis); ri=int(np.argmin(abs(axis-RM)))
    src=(ri,n-1,n-1); dst=(n-1,ri,n-1)
    e=energy_grid(axis,p,inner)
    dimer=float(energy(np.array([RM,RC,RC]),p,inner));U=-dimer
    labels=np.empty(e.shape,np.int32)
    structure=ndimage.generate_binary_structure(3,1)
    def connected(threshold,structure=structure):
        ndimage.label(e<=threshold,structure=structure,output=labels)
        return labels[src]>0 and labels[src]==labels[dst]
    low=dimer;high=0.
    assert connected(high)
    if connected(low):high=low
    else:
        for _ in range(29):
            mid=(low+high)/2
            if connected(mid):high=mid
            else:low=mid
    minloc=np.unravel_index(np.argmin(e),e.shape)
    result={'p':p,'inner':inner,'h':h,'n_axis':n,'vertices':int(e.size),
        'dimer_energy':dimer,'barrier_energy':high-dimer,'barrier_over_U':(high-dimer)/U,
        'threshold':high,'threshold_bracket_width':high-low,'grid_minimum_energy':float(e[minloc]),
        'grid_minimum_over_U':float(e[minloc]/U),'grid_minimum_distances':axis[np.array(minloc)].tolist()}
    if save_path:
        inds=path_bfs(e,high+2.e-12,src,dst)
        path=axis[inds]
        values=np.array([energy(x,p,inner) for x in path])
        peak=int(values.argmax())
        dense=np.vstack([a+(b-a)*t for a,b in zip(path[:-1],path[1:]) for t in np.linspace(0,1,9)])
        dense=np.vstack([dense,path[-1]])
        dense_e=np.array([energy(x,p,inner) for x in dense])
        result.update({'path_nodes':len(path),'path_peak_distances':path[peak].tolist(),
                       'dense_path_barrier_over_U':float((dense_e.max()-dimer)/U)})
        tag=f'p{p:g}_rin{inner:g}_h{h:g}'
        np.savez_compressed(ROOT/(tag+'_path.npz'),distances=path,energy=values,dense_distances=dense,dense_energy=dense_e)
        # A broader 26-neighbor connection is a separate discretization check.
        s26=ndimage.generate_binary_structure(3,3)
        lo=dimer;hi=high
        if connected(lo,s26):hi=lo
        else:
            for _ in range(25):
                mid=(lo+hi)/2
                if connected(mid,s26):hi=mid
                else:lo=mid
        result['barrier_26_neighbors_over_U']=float((hi-dimer)/U)
    result['seconds']=time.perf_counter()-start_time
    print(json.dumps(result),flush=True)
    del e,labels;gc.collect()
    return result

def run(mode):
    result_path=ROOT/'grid_results.json'
    results=json.loads(result_path.read_text()) if result_path.exists() else []
    if mode=='screen':cases=[(p,r,.01) for r in [1.3,1.2,1.15] for p in [.6,.8,1.,1.5,2.,4.,8.]]
    elif mode=='refine':cases=[(p,1.3,h) for h in [.005,.0025] for p in [2.,4.,8.]]
    elif mode=='fine':cases=[(p,1.3,.00125) for p in [2.,4.,8.]]
    else:raise ValueError(mode)
    for p,r,h in cases:
        if any(row['p']==p and row['inner']==r and row['h']==h for row in results):continue
        results.append(grid_search(p,r,h,save_path=(mode!='screen')))
        result_path.write_text(json.dumps(results,indent=2)+'\n')
    keys=['p','inner','h','barrier_over_U','grid_minimum_over_U','seconds']
    (ROOT/'grid_results.csv').write_text(','.join(keys)+'\n'+'\n'.join(','.join(str(row[k]) for k in keys) for row in results)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['validate','screen','refine','fine','all'])
    args=parser.parse_args()
    if args.mode in ('validate','all'):validate()
    if args.mode=='all':
        for mode in ['screen','refine']:run(mode)
    elif args.mode!='validate':run(args.mode)
