"""Admissibility accounting for the 2000-point LHS sample and the surrogate ablation without inadmissible converged cases."""
from common import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
a = np.load('lhs_raw.npy'); N = len(a); ok = a[:, -1] == 1
X = a[:, :8]; kr = np.array([props(p/2, p/2)['kr'] for p in X[:, 7]]); DS = X[:, 5]*X[:, 6]
viol = DS >= kr
out = dict(N=int(N), converged=int(ok.sum()), failed=int((~ok).sum()), failed_all_violate=bool(np.all(viol[~ok])),
           n_violate=int(viol.sum()), conv_and_violate=int((ok & viol).sum()), conv_and_admissible=int((ok & ~viol).sum()),
           admissible_failed=int((~ok & ~viol).sum()))
# determinant profile along each converged solution
mins, flips, maxs = [], [], []
for i in np.where(ok)[0]:
    par = pdict(X[i]); r = solve(par, tol=1e-6, max_nodes=30000)
    if r is None: mins.append(np.nan); maxs.append(np.nan); flips.append(False); continue
    eta, det = det_profile(r, par); mins.append(float(det.min())); maxs.append(float(det.max())); flips.append(bool(det.min() < 0 < det.max()))
mins = np.array(mins); maxs = np.array(maxs); flips = np.array(flips); idx = np.where(ok)[0]
out['admissible_min_det'] = float(np.nanmin(mins[~viol[idx]])); out['admissible_det_sign_change'] = int(flips[~viol[idx]].sum())
out['violating_converged'] = [dict(row=int(i), DuSr=float(DS[i]), kr=float(kr[i]), min_det=float(mins[list(idx).index(i)]), max_det=float(maxs[list(idx).index(i)]), negative_everywhere=bool(maxs[list(idx).index(i)] < 0),
                                  sign_change=bool(flips[list(idx).index(i)])) for i in idx[viol[idx]]]
# surrogate ablation
D = a[ok]; Xc = D[:, :8]; Y = D[:, 8:12]; keep = ~viol[ok]
def fwd(y, t): return np.arcsinh(y) if t in ('Nu', 'Sh') else y
def inv(z, t): return np.sinh(z) if t in ('Nu', 'Sh') else z
def gbt(): return GradientBoostingRegressor(n_estimators=600, max_depth=4, learning_rate=0.04, subsample=0.9, random_state=0)
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
abl = {}
for j, t in enumerate(['Cf', 'Cn', 'Nu', 'Sh']):
    for nm, mask in (('all_converged', np.ones(len(Xc), bool)), ('admissible_only', keep)):
        r2 = []
        for tr, te in rkf.split(Xc[mask]):
            xs, ys = Xc[mask], Y[mask, j]
            m = gbt().fit(xs[tr], fwd(ys[tr], t)); r2.append(r2_score(ys[te], inv(m.predict(xs[te]), t)))
        abl[f'{t}|{nm}'] = (float(np.mean(r2)), float(np.std(r2)))
    print(t, abl[f'{t}|all_converged'], abl[f'{t}|admissible_only'], flush=True)
out['ablation'] = abl
json.dump(out, open('stage6.json', 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'ablation'}, indent=1))
