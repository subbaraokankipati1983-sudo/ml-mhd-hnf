"""Figure 11 (appendix): calibrated uncertainty of the Nu surrogate and closure-exponent sweep."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'serif'})
c=json.load(open('stage8.json')); s=json.load(open('stage7.json'))
fig,ax=plt.subplots(1,3,figsize=(9.2,2.7))
lv=[0.5,0.7,0.8,0.9,0.95]; ax[0].plot([.45,1],[.45,1],'k:',lw=1)
for t,m,col in (('Nu','o','tab:blue'),('Sh','s','tab:red')):
    y=[c[t][str(l)][0] for l in lv]; e=[c[t][str(l)][1] for l in lv]; ax[0].errorbar(lv,y,e,fmt=m+'-',ms=4,color=col,capsize=2,label=r'$%s_x$'%t)
ax[0].set_xlabel('Nominal coverage'); ax[0].set_ylabel('Empirical coverage'); ax[0].legend(frameon=False); ax[0].set_title('(a) Split-conformal calibration',fontsize=8)
o=c['optima']; sd=[r['seed'] for r in o]
for r in o:
    ax[1].plot([r['seed']]*2,[r['lo'],r['hi']],color='0.6',lw=3,solid_capstyle='butt')
    ax[1].plot(r['seed'],r['pred'],'_',color='k',ms=8)
    ax[1].plot(r['seed'],r['solver'],'o',ms=4,color='tab:green' if r['inside'] else 'tab:red')
ax[1].set_xlabel('DE seed'); ax[1].set_ylabel(r'$Nu_x$'); ax[1].set_xticks(sd); ax[1].set_title('(b) Solver values vs 90% interval',fontsize=8)
ps=sorted(s,key=float); x=[float(p) for p in ps]
ax[2].plot(x,[s[p]['Nu_at_ref'] for p in ps],'o-',color='tab:blue',label='reference design, re-solved')
ax[2].plot(x,[s[p]['best']['Nu'] for p in ps],'s--',color='tab:orange',label='re-optimized (seed 1)')
ax[2].set_xlabel(r'Closure exponent $p$ in $k_d\propto|f^{\prime}|^{p}$'); ax[2].set_ylabel(r'$Nu_x$'); ax[2].legend(frameon=False,fontsize=7); ax[2].set_title('(c) Closure-exponent sweep',fontsize=8)
fig.tight_layout()
for e in ('pdf','png'): fig.savefig('../figures/Figure_11.'+e,dpi=300)
