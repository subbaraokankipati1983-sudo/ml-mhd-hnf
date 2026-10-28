"""Independent Chebyshev-Newton check of the solve_bvp results at the baseline, the reference optimum and random LHS points."""
from common import *
from cheb_check import solve_cheb
base = dict(Ha=1, K=0.5, S=0.2, fw=1.5, D=1, Sr=0.5, Du=0.3, phi1=0.03, phi2=0.03, **MODEL)
de = [json.loads(l) for l in open('de_results.jsonl')]; xo = [o for o in de if o['tag'] == 'ref' and o['seed'] == 1][0]['x']
cases = {'baseline': base, 'optimum': pdict(xo)}
a = np.load('lhs_raw.npy'); ok = np.where(a[:, -1] == 1)[0]; rng = np.random.default_rng(27)
kr = np.array([props(p/2, p/2)['kr'] for p in a[:, 7]]); adm = a[:, 5]*a[:, 6] < 0.9*kr
pool = [i for i in ok if adm[i]]; pick = rng.choice(pool, 40, replace=False)
for i in pick: cases[f'lhs{i}'] = pdict(a[i, :8])
out = {}; worst = 0
for k, par in cases.items():
    r = solve(par, einf=16, nodes=1500, tol=1e-9, max_nodes=100000)
    best = None
    for L, N in ((14, 90), (16, 130), (18, 160)):
        c = solve_cheb(par, L=L, N=N)
        if c['ok'] and c['resnorm'] < 1e-6: best = (c, L, N); break
    if r is None or best is None: out[k] = None; continue
    c = best[0]; rel = max(abs(c[q]-r[q])/max(1, abs(r[q])) for q in ('Cf', 'Cn', 'Nu', 'Sh'))
    out[k] = dict(rel=float(rel), bvp=[float(r[q]) for q in ('Cf', 'Cn', 'Nu', 'Sh')], cheb=[float(c[q]) for q in ('Cf', 'Cn', 'Nu', 'Sh')], L=best[1], N=best[2])
    if k.startswith('lhs'): worst = max(worst, rel)
out['n_lhs_compared'] = sum(1 for k, v in out.items() if k.startswith('lhs') and v); out['max_rel_lhs'] = float(worst)
json.dump(out, open('cheb.json', 'w'), indent=1); print(out['baseline'], out['optimum'], out['n_lhs_compared'], worst)
