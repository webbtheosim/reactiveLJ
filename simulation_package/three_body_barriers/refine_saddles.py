"""Continuous saddle refinement after the full three-distance grid search."""
import json, sys
from pathlib import Path
import numpy as np
from scipy import optimize
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_barriers import ROOT, RM, RC, DEPTH, ELBOW, energy, weight, lj, dlj, grid_search

def gradient(x,p,inner):
    # The candidate saddles lie outside the elbow interval. This derivative
    # includes the coordination contribution from both other pair energies.
    x=np.asarray(x,float)
    assert np.min(x)>ELBOW+.05
    w=np.array([weight(r,inner) for r in x])
    dw=np.where((x>inner)&(x<RC),-.5*np.pi/(RC-inner)*np.sin(np.pi*(x-inner)/(RC-inner)),0.)
    raw=np.array([w[1]+w[2],w[0]+w[2],w[0]+w[1]])
    sq=np.sqrt(raw*raw+1.e-12);ce=.5*(raw+sq)
    W=(1.+ce)**(-p);dW=-p*W/(1.+ce)*.5*(1.+raw/sq)
    u=np.array([lj(r) if r<RC else 0. for r in x])
    du=np.array([dlj(r) if r<RC else 0. for r in x])
    g=W*du
    for i in range(3):
        g[i]+=dw[i]*sum(dW[j]*u[j] for j in range(3) if j!=i)
    return g

def refine(p,inner,guess):
    sol=optimize.least_squares(lambda x:gradient(x,p,inner),guess,
        bounds=(np.full(3,ELBOW+.05001),np.full(3,RC-1.e-9)),
        xtol=1.e-14,ftol=1.e-14,gtol=1.e-13,max_nfev=1000)
    x=sol.x;g=gradient(x,p,inner)
    assert np.max(abs(g))<1.e-8,(p,inner,sol)
    h=2.e-6
    numeric=np.array([(energy(x+np.eye(3)[i]*h,p,inner)-energy(x-np.eye(3)[i]*h,p,inner))/(2*h) for i in range(3)])
    np.testing.assert_allclose(g,numeric,atol=5.e-8,rtol=1.e-6)
    h=1.e-5
    H=np.column_stack([(gradient(x+np.eye(3)[i]*h,p,inner)-gradient(x-np.eye(3)[i]*h,p,inner))/(2*h) for i in range(3)])
    eig,vec=np.linalg.eigh(.5*(H+H.T))
    assert sum(eig<-1.e-5)==1,(p,inner,x,eig)
    dimer=energy(np.array([RM,RC,RC]),p,inner)
    descents=[]
    for sign in [-1,1]:
        # These endpoints help identify whether the saddle leads to a dimer
        # basin or to the two-bond intermediate along the exchange path.
        q=optimize.minimize(lambda y:energy(y,p,inner),x+sign*.003*vec[:,0],
            method='L-BFGS-B',bounds=[(1.08,RC)]*3,
            options={'ftol':1.e-14,'gtol':1.e-9,'maxiter':3000,'maxls':100})
        descents.append({'distances':q.x.tolist(),'energy_over_U':float(q.fun/-dimer),'success':bool(q.success)})
    return {'p':p,'inner':inner,'distances':x.tolist(),'saddle_energy':float(energy(x,p,inner)),
            'barrier_over_U':float((energy(x,p,inner)-dimer)/-dimer),
            'gradient_max_abs':float(max(abs(g))),'hessian_eigenvalues':eig.tolist(),
            'descending_endpoints':descents}

if __name__=='__main__':
    records=[];grid_checks=[]
    for inner in [1.3,1.2,1.15]:
        for p in [1.,1.5,2.,4.,8.]:
            fn=ROOT/f'p{p:g}_rin{inner:g}_h0.0025_path.npz'
            if fn.exists():
                d=np.load(fn);x=d['distances'][np.argmax(d['energy'])]
            else:
                row=grid_search(p,inner,.005,save_path=True)
                grid_checks.append(row);x=np.array(row['path_peak_distances'])
            r=refine(p,inner,x);records.append(r)
            print(json.dumps(r),flush=True)
            (ROOT/'saddle_results.json').write_text(json.dumps(records,indent=2)+'\n')
    base=next(r for r in records if r['p']==1.5 and r['inner']==1.3)
    def residual(v):
        x=v[:3];p=v[3];d=energy(np.array([RM,RC,RC]),p,1.3)
        return np.r_[gradient(x,p,1.3),(energy(x,p,1.3)-d)/-d-.5]
    sol=optimize.least_squares(residual,np.r_[base['distances'],1.5],
        bounds=(np.r_[np.full(3,1.08),1.4],np.r_[np.full(3,RC-1.e-9),1.7]),
        xtol=1.e-14,ftol=1.e-14,gtol=1.e-13)
    assert max(abs(residual(sol.x)))<1.e-9
    mapping={'p_for_half_U':float(sol.x[3]),'saddle_distances':sol.x[:3].tolist()}
    (ROOT/'half_U_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (ROOT/'supplementary_grid_checks.json').write_text(json.dumps(grid_checks,indent=2)+'\n')
    print(json.dumps(mapping),flush=True)
