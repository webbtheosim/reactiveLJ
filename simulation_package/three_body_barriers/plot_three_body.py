"""Draw the manuscript figure from the checked barrier cache and analytic f(z)."""
from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parent
STYLE=ROOT.parent/'melt/reactive_ah_comparison_20261029/code'
sys.path.insert(0,str(STYLE))
from plot_style import plt, setup, format_axes
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

setup()
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,
                     'svg.hashsalt':'reactivelj-three-body-20261102'})
rows=json.loads((ROOT/'figure_barriers.json').read_text())
assert len(rows)==225,'The complete calculation must finish before plotting.'
fig,axes=plt.subplots(1,2,figsize=(7.,3.4))
fig.subplots_adjust(left=.085,right=.985,bottom=.26,top=.91,wspace=.34)
ax=axes[0]
colors={1.15:'#d55e00',1.2:'#009e73',1.3:'#0072b2'}
for inner in [1.15,1.2,1.3]:
    data=sorted((r for r in rows if r['inner']==inner),key=lambda r:r['p'])
    p=np.array([r['p'] for r in data]);y=np.array([r['barrier_over_U'] for r in data])
    assert len(p)==75 and p[0]==0 and p[-1]==9
    ax.plot(p,y,color=colors[inner],lw=1.2,label=rf'$r_\mathrm{{in}}={inner:.2f}\sigma$')
    for val in [2.,4.,8.]:
        point=next(r for r in data if r['p']==val)
        ax.plot(val,point['barrier_over_U'],'o',mfc='white',mec=colors[inner],ms=4.2,mew=.9,zorder=5)
p=np.linspace(0,9,901)
ax.plot(p,1-2**(1-p),color='.25',ls='--',lw=.9,label=r'Two-bond $\Delta E_2/U$')
ax.axhline(1.,color='.35',ls=':',lw=.9,label='Full dissociation')
ax.axhline(0.,color='.75',lw=.5,zorder=0)
ax.set(xlim=(0,9),ylim=(-1.05,1.07),xlabel=r'Weakening exponent, $p$',ylabel=r'Energy / $U$')
ax.set_xticks([0,2,4,6,8]);ax.set_yticks([-1,-.5,0,.5,1])
ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.legend(frameon=False,loc='lower right',handlelength=1.5,labelspacing=.25,borderpad=.1)
format_axes(ax,'a')

ax=axes[1]
z=np.linspace(1,8,1401)
ps=[0.,.5,.6,.8,1.,2.]
colors=['#222222','#0072b2','#56b4e9','#009e73','#d55e00','#cc79a7']
cached={'z':z}
for p,color in zip(ps,colors):
    y=z*(2*z-1)**(-p);cached[f'f_p{p:g}']=y
    ax.plot(z,y,color=color,lw=1.2,label=rf'$p={p:g}$')
    if .5<p<1:
        minimum=1/(2*(1-p));fmin=minimum*(2*minimum-1)**(-p)
        ax.plot(minimum,fmin,'^',ms=5.,color=color,mec='black',mew=.4,zorder=6)
ax.set_yscale('log')
ax.set(xlim=(1,8),ylim=(.025,10),xlabel=r'Coordination, $z$',
       ylabel=r'Binding-energy magnitude, $f(z)$')
ax.set_xticks([1,2,4,6,8]);ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)));ax.yaxis.set_minor_formatter(NullFormatter())
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,
          handlelength=1.3,columnspacing=1.0,labelspacing=.25,borderaxespad=0)
format_axes(ax,'b')

output=ROOT/'three_body_barrier_valence'
for suffix in ['.svg','.pdf']:
    fig.savefig(output.with_suffix(suffix),transparent=True)
fig.savefig(output.with_suffix('.png'),dpi=220,facecolor='white')
np.savez_compressed(ROOT/'coordination_curves.npz',**cached)
plt.close(fig)
print('Saved SVG, PDF, PNG, and analytic coordination curves.')
