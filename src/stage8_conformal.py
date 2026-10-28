"""Split-conformal prediction intervals for the GBT surrogate of Nu_x (arcsinh scale), and a coverage test at the solver-verified optima.
Question: can a calibrated surrogate flag the designs it cannot be trusted on? Calibration: 60/20/20 train/calibrate/test, 30 random splits."""
from common import *
import json
from sklearn.ensemble import GradientBoostingRegressor
a = np.load('lhs_raw.npy'); ok = a[:,-1]==1
X = a[ok,:8]; kr = np.array([props(p/2,p/2)['kr'] for p in X[:,7]]); keep = X[:,5]*X[:,6] < kr
X = X[keep]; Y = a[ok][keep][:,8:12]
res = {}
for j,t in ((2,'Nu'),(3,'Sh')):
    z = np.arcsinh(Y[:,j]); levels = [0.5,0.7,0.8,0.9,0.95]; cov = {l:[] for l in levels}; wid = {l:[] for l in levels}
    for s in range(30):
        rng = np.random.default_rng(s); idx = rng.permutation(len(X)); n = len(X)
        tr, ca, te = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]
        m = GradientBoostingRegressor(n_estimators=300,max_depth=4,learning_rate=0.06,subsample=0.9,random_state=s).fit(X[tr],z[tr])
        sc = np.sort(np.abs(z[ca]-m.predict(X[ca])))
        for l in levels:
            q = sc[min(len(sc)-1,int(np.ceil((len(sc)+1)*l))-1)]
            p = m.predict(X[te]); lo,hi = np.sinh(p-q), np.sinh(p+q)
            cov[l].append(float(np.mean((Y[te,j]>=lo)&(Y[te,j]<=hi)))); wid[l].append(float(np.median(hi-lo)))
    res[t] = {str(l):(float(np.mean(cov[l])),float(np.std(cov[l])),float(np.mean(wid[l]))) for l in levels}
    print(t,res[t],flush=True)
# optima check: train on all, calibrate on 25% held out, evaluate the ten reference optima
rng = np.random.default_rng(0); idx = rng.permutation(len(X)); ca = idx[:len(X)//4]; tr = idx[len(X)//4:]
m = GradientBoostingRegressor(n_estimators=600,max_depth=4,learning_rate=0.04,subsample=0.9,random_state=0).fit(X[tr],np.arcsinh(Y[tr,2]))
sc = np.sort(np.abs(np.arcsinh(Y[ca,2])-m.predict(X[ca]))); l=0.9; q = sc[min(len(sc)-1,int(np.ceil((len(sc)+1)*l))-1)]
R = [json.loads(x) for x in open('de_results.jsonl')]; R = [r for r in R if r['tag']=='ref']
rows=[]
for r in R:
    x = np.array(r['x']); p = m.predict(x[None])[0]; lo,hi = float(np.sinh(p-q)),float(np.sinh(p+q))
    rows.append(dict(seed=r['seed'],solver=r['Nu'],pred=float(np.sinh(p)),lo=lo,hi=hi,inside=bool(lo<=r['Nu']<=hi)))
res['optima']=rows; res['q90_arcsinh']=float(q)
print('optima inside:',sum(r['inside'] for r in rows),'/',len(rows)); [print(r) for r in rows]
json.dump(res,open('stage8.json','w'),indent=1)
