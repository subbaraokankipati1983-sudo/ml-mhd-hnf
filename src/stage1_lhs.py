from common import *
N=2000
X=qmc.scale(qmc.LatinHypercube(d=8,seed=20260).random(N),SAMP[:,0],SAMP[:,1])
import os
rows=list(np.load('lhs_raw.npy')) if os.path.exists('lhs_raw.npy') else []
t0=time.time()
for i,x in enumerate(X):
    if i<len(rows): continue
    t=time.time(); r=run(x)
    ok = r is not None and np.isfinite(r['Nu'])
    rows.append(list(x)+([r['Cf'],r['Cn'],r['Nu'],r['Sh'],1] if ok else [np.nan]*4+[0]))
    if i%100==0:
        print(i,round(time.time()-t0),'s; conv so far',int(np.nansum([q[-1] for q in rows])),flush=True)
        np.save('lhs_raw.npy',np.array(rows))
np.save('lhs_raw.npy',np.array(rows)); print('done',time.time()-t0, flush=True)
