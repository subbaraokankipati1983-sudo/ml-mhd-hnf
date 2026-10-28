from common import *
from scipy.optimize import differential_evolution
import sys
def de(seed, bounds=OPT, wcf=0.15, wsh=0.05, fixed=None, extra=None, maxiter=15, pop=6, dscap=None):
    extra=extra or {}; cnt=[0]
    b=[tuple(r) for r in bounds]
    def f(x):
        x=np.array(x)
        if fixed:
            for k,v in fixed.items(): x[NAMES.index(k)]=v
        if dscap is not None and x[NAMES.index('Du')]*x[NAMES.index('Sr')]>dscap*props(x[7]/2,x[7]/2)['kr']: return 1e3
        r=run(x,tol=1e-5,**extra); cnt[0]+=1
        if r is None or not np.isfinite(r['Nu']): return 1e3
        return -J(r,wcf,wsh)
    res=differential_evolution(f,b,seed=seed,popsize=pop,maxiter=maxiter,tol=1e-5,mutation=(0.4,1.2),recombination=0.7,polish=False)
    x=np.array(res.x)
    if fixed:
        for k,v in fixed.items(): x[NAMES.index(k)]=v
    for einf,nodes in [(12,1000),(16,1200),(20,1200)]:
        r=run(x,einf=einf,nodes=nodes,tol=1e-8,**extra)
        if r is not None: break
    out=dict(seed=seed,x=x.tolist(),evals=cnt[0])
    if r: out.update(Cf=r['Cf'],Cn=r['Cn'],Nu=r['Nu'],Sh=r['Sh'],J=J(r,wcf,wsh),rms=r['rms'])
    return out
if __name__=='__main__':
    tag=sys.argv[1]; seed=int(sys.argv[2])
    cfg={'ref':{},'w2cf':dict(wcf=0.30),'whcf':dict(wcf=0.075),'w2sh':dict(wsh=0.10),'whsh':dict(wsh=0.025),
         'D0':dict(fixed={'D':0.0},bounds=np.array([[0,5],[0,3],[0,0.4],[0.5,3],[0.0,0.0],[0,1.0],[0,0.8],[0,0.08]])),
         'shear':dict(extra=dict(closure='shear')),'nubulk':dict(extra=dict(nu_disp=False)),
         'cap10':dict(bounds=np.array([[0,5],[0,3],[0,0.4],[0.5,1.0],[0.25,2],[0,1.0],[0,0.8],[0,0.08]])),
         'cap15':dict(bounds=np.array([[0,5],[0,3],[0,0.4],[0.5,1.5],[0.25,2],[0,1.0],[0,0.8],[0,0.08]])),
         'cap20':dict(bounds=np.array([[0,5],[0,3],[0,0.4],[0.5,2.0],[0.25,2],[0,1.0],[0,0.8],[0,0.08]])),
         'cap25':dict(bounds=np.array([[0,5],[0,3],[0,0.4],[0.5,2.5],[0.25,2],[0,1.0],[0,0.8],[0,0.08]])),
         'cftotal':dict(extra=dict(cf_total=True)),
         'wide':dict(bounds=np.array([[0,5],[0,3],[0,0.6],[0.5,4],[0.25,2],[0,1.5],[0,1.15],[0,0.08]])),
         'wide_ds90':dict(bounds=np.array([[0,5],[0,3],[0,0.6],[0.5,4],[0.25,2],[0,1.5],[0,1.15],[0,0.08]]),dscap=0.9),
         'wide_ds50':dict(bounds=np.array([[0,5],[0,3],[0,0.6],[0.5,4],[0.25,2],[0,1.5],[0,1.15],[0,0.08]]),dscap=0.5),
         'wide_ds70':dict(bounds=np.array([[0,5],[0,3],[0,0.6],[0.5,4],[0.25,2],[0,1.5],[0,1.15],[0,0.08]]),dscap=0.7),
         'wide_ds80':dict(bounds=np.array([[0,5],[0,3],[0,0.6],[0.5,4],[0.25,2],[0,1.5],[0,1.15],[0,0.08]]),dscap=0.8),
         'wide_ds95':dict(bounds=np.array([[0,5],[0,3],[0,0.6],[0.5,4],[0.25,2],[0,1.5],[0,1.15],[0,0.08]]),dscap=0.95),
         'phifloor':dict(bounds=np.array([[0,5],[0,3],[0,0.4],[0.5,3],[0.25,2],[0,1.0],[0,0.8],[0.02,0.08]]))}[tag]
    t=time.time(); o=de(seed,**cfg); o['tag']=tag; o['sec']=time.time()-t
    open('de_results.jsonl','a').write(json.dumps(o)+'\n'); print(json.dumps(o))
