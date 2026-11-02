"""Sample the verified three-sticker potential for the manuscript figure."""
from pathlib import Path
import contextlib, csv, io, json, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_barriers import ROOT, grid_search
from refine_saddles import refine

path=ROOT/'figure_barriers.json'
rows=json.loads(path.read_text()) if path.exists() else []
ps=np.unique(np.r_[np.linspace(0,9,73),.6,.8])
known=json.loads((ROOT/'saddle_results.json').read_text())
for inner in [1.15,1.2,1.3]:
    for p in ps:
        p=float(p)
        if any(r['p']==p and r['inner']==inner for r in rows):continue
        with contextlib.redirect_stdout(io.StringIO()):
            grid=grid_search(p,inner,.01,save_path=True)
        value=grid['barrier_over_U'];method='grid';saddle=None
        if value<1.e-9:
            value=0.;method='zero-barrier grid path'
        else:
            for attempt in range(2):
                try:
                    saddle=refine(p,inner,np.array(grid['path_peak_distances']))
                    assert abs(saddle['barrier_over_U']-grid['dense_path_barrier_over_U'])<.0015
                    value=saddle['barrier_over_U'];method='refined index-one saddle'
                    break
                except (AssertionError,ValueError):
                    if attempt==0:
                        with contextlib.redirect_stdout(io.StringIO()):
                            grid=grid_search(p,inner,.005,save_path=True)
                    else:
                        # Resolve narrow channel transitions without assuming
                        # that a particular stationary point controls the path.
                        with contextlib.redirect_stdout(io.StringIO()):
                            fine=grid_search(p,inner,.0025,save_path=True)
                        assert abs(fine['dense_path_barrier_over_U']-grid['dense_path_barrier_over_U'])<.001
                        grid=fine;value=fine['dense_path_barrier_over_U']
                        method='converged continuous grid path'
        for anchor in known:
            if anchor['p']==p and anchor['inner']==inner:
                assert abs(value-anchor['barrier_over_U'])<2.e-7,(p,inner,value,anchor)
        row={'p':p,'inner':inner,'barrier_over_U':value,'method':method,
             'grid_h':grid['h'],'grid_barrier_over_U':grid['barrier_over_U'],
             'continuous_grid_barrier_over_U':grid['dense_path_barrier_over_U'],
             'saddle':saddle}
        rows.append(row)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2)+'\n');tmp.replace(path)
        if len(rows)%15==0:print('Completed',len(rows),'of',3*len(ps),'parameter points',flush=True)
rows.sort(key=lambda r:(r['inner'],r['p']))
for inner in [1.15,1.2,1.3]:
    values=np.array([r['barrier_over_U'] for r in rows if r['inner']==inner])
    assert np.all(np.diff(values)>-1.e-4),(inner,'nonmonotonic curve')
with (ROOT/'figure_barriers.csv').open('w',newline='') as f:
    keys=['p','inner','barrier_over_U','method','grid_h','grid_barrier_over_U','continuous_grid_barrier_over_U']
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
print('Finished',len(rows),'checked barrier values',flush=True)
