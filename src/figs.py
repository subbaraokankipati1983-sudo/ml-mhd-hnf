from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pickle
from cycler import cycler
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"serif","mathtext.fontset":"stix","font.size":10,"axes.labelsize":11,"legend.fontsize":9,"legend.frameon":False,"axes.linewidth":0.8,"lines.linewidth":1.6,
 "axes.prop_cycle":cycler(color=["#000000","#D55E00","#0072B2","#009E73","#CC79A7","#E69F00"])+cycler(linestyle=["-","--","-.",":",(0,(5,1,1,1)),(0,(3,1,1,1,1,1))])})
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures')+os.sep; os.makedirs(O,exist_ok=True)
NEWNAME={'Figure_10':'Figure_1','Figure_1':'Figure_2','Figure_2':'Figure_3','Figure_11':'Figure_4','Figure_8':'Figure_5a','Figure_9':'Figure_5b','Figure_3':'Figure_6a','Figure_4':'Figure_6b','Figure_5':'Figure_7','Figure_6':'Figure_8','Figure_7':'Figure_9'}  # numbering follows order of appearance in the manuscript
def save(fig,n):
    n=NEWNAME.get(n,n)
    fig.savefig(O+n+'.pdf',bbox_inches='tight'); fig.savefig(O+n+'.png',dpi=600,bbox_inches='tight'); plt.close(fig)
BASE=dict(Ha=1,K=0.5,S=0.2,fw=1.5,D=1,Sr=0.5,Du=0.3,phi=0.06)
def prof(over,einf=8):
    d={**BASE,**over}; x=[d[k] for k in NAMES]; r=run(x,einf=einf,nodes=400,tol=1e-8); eta=np.linspace(0,6 if einf<=8 else einf,300); return eta,r['sol'].sol(eta)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for Ha in (0,1,3,5):
    e,y=prof(dict(Ha=Ha)); ax[0].plot(e,y[1],label=f"Ha={Ha}"); ax[1].plot(e,y[5],label=f"Ha={Ha}")
ax[0].set_xlabel(r"$\eta$");ax[0].set_ylabel(r"$f'(\eta)$");ax[1].set_xlabel(r"$\eta$");ax[1].set_ylabel(r"$\theta(\eta)$");ax[0].legend();ax[1].legend();fig.tight_layout();save(fig,'Figure_3')
fig,ax=plt.subplots(1,2,figsize=(10,4))
for K in (0,0.5,1,2,3):
    e,y=prof(dict(K=K)); ax[0].plot(e,y[1],label=f"K={K}"); ax[1].plot(e,y[3],label=f"K={K}")
ax[0].set_xlabel(r"$\eta$");ax[0].set_ylabel(r"$f'(\eta)$");ax[1].set_xlabel(r"$\eta$");ax[1].set_ylabel(r"$h(\eta)$");ax[0].legend();ax[1].legend();fig.tight_layout();save(fig,'Figure_4')
fig,ax=plt.subplots(figsize=(6,4.2))
for D in (0.1,0.3,0.5,1,1.5,2):
    e,y=prof(dict(D=D)); ax.plot(e,y[5],label=f"D={D}")
ax.set_xlabel(r"$\eta$");ax.set_ylabel(r"$\theta(\eta)$");ax.legend();fig.tight_layout();save(fig,'Figure_5')
fig,ax=plt.subplots(figsize=(6,4.2))
for Sr in (0,0.5,1,1.5):
    e,y=prof(dict(Sr=Sr)); ax.plot(e,y[7],label=f"Sr={Sr}")
ax.set_xlabel(r"$\eta$");ax.set_ylabel(r"$\chi(\eta)$");ax.legend();fig.tight_layout();save(fig,'Figure_6')
phis=np.linspace(0,0.08,12); cf=[];nu=[]
for p in phis:
    d={**BASE,'phi':p}; r=run([d[k] for k in NAMES],einf=8,tol=1e-8); cf.append(r['Cf']); nu.append(r['Nu'])
np.save('phi_sweep.npy',np.c_[phis,cf,nu])
fig,a1=plt.subplots(figsize=(6,4.2)); a2=a1.twinx()
a1.plot(phis,cf,'s-',color='tab:red'); a2.plot(phis,nu,'o-',color='tab:blue')
a1.set_xlabel(r"$\phi=\phi_1+\phi_2$");a1.set_ylabel(r"$C_f$",color='tab:red');a2.set_ylabel(r"$Nu_x$",color='tab:blue');fig.tight_layout();save(fig,'Figure_7')
# parity + importance
a=np.load('lhs_raw.npy'); ok=a[:,-1]==1; kr0=np.array([props(p/2,p/2)['kr'] for p in a[:,7]]); Dd=a[ok&(a[:,5]*a[:,6]<kr0)]; X=Dd[:,:8]; ship=pickle.load(open('ship.pkl','rb'))
fig,axs=plt.subplots(2,2,figsize=(9,8))
for ax,(j,t) in zip(axs.flat,enumerate(['Cf','Cn','Nu','Sh'])):
    z=ship[t].predict(X); p=np.sinh(z) if t in('Nu','Sh') else z; y=Dd[:,8+j]
    ax.scatter(y,p,s=8,alpha=.5); lo,hi=min(y.min(),p.min()),max(y.max(),p.max()); ax.plot([lo,hi],[lo,hi],'k--',lw=1)
    ax.set_xlabel(f"BVP solver {t}"); ax.set_ylabel(f"Surrogate {t}")
fig.tight_layout();save(fig,'Figure_1')
imp=ship['Nu'].feature_importances_; od=np.argsort(imp)
fig,ax=plt.subplots(figsize=(6,4)); ax.barh(np.array(NAMES)[od],imp[od],color='seagreen'); ax.set_xlabel('Relative importance (Nu surrogate)'); fig.tight_layout(); save(fig,'Figure_2')
# sobol + pareto composite figs 8, 9
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,t in zip(axs,['Nu','Sh']):
    d=np.load(f'sobol_{t}.npz'); od=np.argsort(d['ST']); ax.barh(d['names'][od],d['ST'][od],xerr=d['ST_conf'][od],color='#3b7dd8')
    ax.set_xlabel('Total-order Sobol index $S_T$ (Jansen estimator)'); ax.set_title(r'$\mathit{%s}_x$'%t)
fig.tight_layout();save(fig,'Figure_8')
v=np.load('pareto_ver.npy'); fig,ax=plt.subplots(figsize=(6,5)); s=ax.scatter(v[:,8],v[:,10],c=v[:,9],cmap='viridis',s=60)
ax.set_xlabel(r'$Nu_x$');ax.set_ylabel(r'$|Sh_x|$');cb=fig.colorbar(s,ax=ax);cb.set_label(r'$|C_f|$');fig.tight_layout();save(fig,'Figure_9')
# feasibility map (Figure 10): Sr-Du plane with the admissibility curves
kr_all=np.array([props(p/2,p/2)['kr'] for p in a[:,7]]); bad=ok&(a[:,5]*a[:,6]>=kr_all)
fig,ax=plt.subplots(figsize=(6.4,4.6))
good=ok&~bad
ax.scatter(a[good,5],a[good,6],s=7,c='#0072B2',label='converged, admissible')
ax.scatter(a[~ok,5],a[~ok,6],s=16,c='#D55E00',marker='x',label='not converged')
ax.scatter(a[bad,5],a[bad,6],s=40,facecolors='none',edgecolors='k',marker='s',label='converged, inadmissible')
sr=np.linspace(0.3,1.5,200)
for ph,ls in ((0,'-'),(0.08,'--')): ax.plot(sr,props(ph/2,ph/2)['kr']/sr,'k',ls=ls,lw=1.2,label=r'Du Sr $=A_4$, $\phi=%g$'%ph)
ax.set_xlim(0,1.5); ax.set_ylim(0,1.15); ax.set_xlabel('Sr'); ax.set_ylabel('Du')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=2,fontsize=8); fig.tight_layout(); save(fig,'Figure_10')
# Figure 11: best solver-verified Nu against the cap on suction
R=[json.loads(l) for l in open('de_results.jsonl')]
pts=[(c,[r for r in R if r['tag']==t and r['seed']==1][0]['Nu']) for t,c in (('cap10',1.0),('cap15',1.5),('cap20',2.0),('cap25',2.5),('ref',3.0))]
vb=json.load(open('val.json'))['baseline']['Nu']
fig,ax=plt.subplots(figsize=(6,4.2)); ax.plot([p[0] for p in pts],[p[1] for p in pts],'o-',color='#0072B2'); ax.axhline(vb,color='k',ls='--',lw=1)
ax.text(1.02,vb+0.8,'baseline',fontsize=9); ax.set_xlabel(r'Upper limit on $f_w$'); ax.set_ylabel(r'Best solver-verified $Nu_x$'); ax.set_ylim(0,None); fig.tight_layout(); save(fig,'Figure_11')
print('figs ok')
