from common import *
import model, pickle
from scipy.stats import qmc
ship=pickle.load(open('ship.pkl','rb'))
de=[json.loads(l) for l in open('de_results.jsonl')]
ref=[o for o in de if o['tag']=='ref' and o['seed']==1][0]
xo=np.array(ref['x']); out={'xo':xo.tolist()}
base=dict(Ha=1,K=0.5,S=0.2,fw=1.5,D=1,Sr=0.5,Du=0.3,phi=0.06)
xb=np.array([base[k] for k in NAMES])
def hi(x,**e): return run(x,einf=12,nodes=1000,tol=1e-8,**e)
rb=hi(xb); ro=hi(xo); out['base']=[rb['Nu'],rb['Cf'],rb['Sh']]; out['opt']=[ro['Cf'],ro['Cn'],ro['Nu'],ro['Sh']]; out['ratio']=ro['Nu']/rb['Nu']
# decomposition (order: fw, Du, Sr, K, Ha&S, D, phi)
steps=[('fw',['fw']),('Du',['Du']),('Sr',['Sr']),('K',['K']),('Ha,S',['Ha','S']),('D',['D']),('phi',['phi'])]
x=xb.copy(); dec=[('baseline',rb['Nu'])]
for nm,ks in steps:
    for k in ks: x[NAMES.index(k)]=xo[NAMES.index(k)]
    dec.append((nm,hi(x)['Nu']))
out['decomp']=dec
# local sensitivity
sens={}
for k in ('D','fw','K','Du'):
    sens[k]=[]
    for p in (-.10,-.05,0,.05,.10):
        xx=xo.copy(); xx[NAMES.index(k)]*=1+p; r=hi(xx); sens[k].append((p,r['Nu'],J(r)))
out['sens']=sens
# property robustness at optimum: Einstein viscosity, HC conductivity
def alt(props_patch):
    orig=model.props
    def p2(a,b):
        P=orig(a,b); P.update(props_patch(a,b,P)); return P
    model.props=p2
    try: r=hi(xo)
    finally: model.props=orig
    return r
def ein(a,b,P): P=dict(A1=(1+2.5*a)*(1+2.5*b)); return P
def hc(a,b,P):
    n=6.0; kf,k1,k2=model.K['f'],model.K['cu'],model.K['al']
    knf=kf*(k1+(n-1)*kf-(n-1)*a*(kf-k1))/(k1+(n-1)*kf+a*(kf-k1))
    kh=knf*(k2+(n-1)*knf-(n-1)*b*(knf-k2))/(k2+(n-1)*knf+b*(knf-k2)); return dict(kr=kh/kf)
for nm,fn in (('einstein',ein),('hc6',hc)):
    r=alt(fn); out[nm]=[r['Cf'],r['Nu'],r['Sh']]
# alternative closures / wall-flux definitions at optimum (re-solve only)
for nm,e in (('shear_closure',dict(closure='shear')),('nu_bulk',dict(nu_disp=False)),('v24_model',dict(kfactor=False,A0micro=False,nu_disp=False))):
    r=hi(xo,**e); out[nm]=[r['Nu'],r['Cf'],r['Sh']] if r else None
# Sobol (Jansen) on surrogates over OPT box
def predict(X,t): 
    z=ship[t].predict(X); return np.sinh(z) if t in('Nu','Sh') else z
def sobol(t,n=4096,seed=0,nb=100):
    rng=np.random.default_rng(seed); A=qmc.scale(qmc.Sobol(16,seed=seed).random(n)[:,:8],OPT[:,0],OPT[:,1])
    B=qmc.scale(qmc.Sobol(16,seed=seed).random(n)[:,8:],OPT[:,0],OPT[:,1])
    fA=predict(A,t); var=np.var(np.r_[fA,predict(B,t)]); ST=[];conf=[]; ABs=[]
    for i in range(8):
        C=A.copy(); C[:,i]=B[:,i]; fC=predict(C,t); ABs.append(fC)
    for i in range(8):
        d=0.5*(fA-ABs[i])**2; ST.append(d.mean()/var)
        bs=[(0.5*(fA[k]-ABs[i][k])**2).mean()/var for k in (rng.integers(0,n,n) for _ in range(nb))]; conf.append(1.96*np.std(bs))
    return np.array(ST),np.array(conf)
out['sobol']={}
for t in('Nu','Sh'):
    runs=[sobol(t,4096,s)[0] for s in (1,2,3)]
    ST,cf=sobol(t,8192,0); out['sobol'][t]=dict(ST=ST.tolist(),conf=cf.tolist(),seed_spread=np.array(runs).tolist())
    np.savez(f'sobol_{t}.npz',names=np.array(NAMES),ST=ST,ST_conf=cf)
# Pareto: 16384 Sobol points on surrogates, non-dominated, then solver-verify 40 spread points
P=qmc.scale(qmc.Sobol(8,seed=5).random(16384),OPT[:,0],OPT[:,1])
M=np.c_[predict(P,'Nu'),np.abs(predict(P,'Cf')),np.abs(predict(P,'Sh'))]
nd=[]
order=np.argsort(-M[:,0])
best=[]
for i in order:
    if not any((M[j,1]<=M[i,1] and M[j,2]<=M[i,2]) for j in best): best.append(i)
nd=np.array(best); print('nondominated surrogate pts',len(nd))
sel=nd[np.linspace(0,len(nd)-1,min(40,len(nd))).astype(int)]
ver=[]
for i in sel:
    r=hi(P[i])
    if r: ver.append(list(P[i])+[r['Nu'],abs(r['Cf']),abs(r['Sh'])])
ver=np.array(ver)
keep=[k for k in range(len(ver)) if not any((ver[j,8]>=ver[k,8] and ver[j,9]<=ver[k,9] and ver[j,10]<=ver[k,10] and j!=k and (ver[j,8]>ver[k,8] or ver[j,9]<ver[k,9] or ver[j,10]<ver[k,10])) for j in range(len(ver)))]
out['pareto']=ver[keep].tolist(); out['pareto_n_eval']=len(ver)
np.save('pareto_ver.npy',ver[keep])
json.dump(out,open('stage5.json','w'),indent=1,default=float)
print(json.dumps({k:out[k] for k in ('opt','ratio','base','decomp','einstein','hc6','shear_closure','nu_bulk','v24_model')},default=float))
