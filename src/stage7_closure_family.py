"""Closure-exponent family: k_d = D k_hnf |f'|^p. p=1 is the adopted closure (exact similarity reduction). Re-optimizes each p (seed 1) and
re-solves the reference design, to show how much of the result is a property of the closure exponent."""
from common import *
from stage3_de import de
import json
ref = [json.loads(l) for l in open('de_results.jsonl')]
xref = np.array([r for r in ref if r['tag']=='ref' and r['seed']==1][0]['x'])
out = {}
os.environ['MAXN']='8000'
rr=[r for r in ref if r['tag']=='ref' and r['seed']==1][0]
out['1.0']=dict(Nu_at_ref=float(rr['Nu']),best=rr,sec=0.0)
for p in [1.5, 2.0, 3.0]:
    ex = dict(closure='power', pexp=p)
    r0 = run(xref, tol=1e-6, **ex)
    t = time.time(); o = de(1, extra=ex)
    out[str(p)] = dict(Nu_at_ref=None if r0 is None else float(r0['Nu']), best=o, sec=time.time()-t)
    print(p, out[str(p)]['Nu_at_ref'], o.get('Nu'), o['x'], flush=True)
json.dump(out, open('stage7.json','w'), indent=1)
