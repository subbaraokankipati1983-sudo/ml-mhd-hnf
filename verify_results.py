"""Checks the archived results against the values quoted in the manuscript (V28)."""
import json, numpy as np, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
v = json.load(open(f'{R}/val.json')); o = json.load(open(f'{R}/stage5.json')); s = json.load(open(f'{R}/stage2.json'))
de = [json.loads(l) for l in open(f'{R}/de_results.jsonl')]
d = np.loadtxt(f'{R}/lhs_dataset.csv', delimiter=',', skiprows=1)
ref = [r for r in de if r['tag'] == 'ref']; nu = np.array([r['Nu'] for r in ref])
checks = [
 ('baseline Nu', v['baseline']['Nu'], 12.161, 5e-4), ('baseline Cf', v['baseline']['Cf'], -2.5709, 5e-4),
 ('reference optimum Nu', o['opt'][2], 42.202, 5e-4), ('gain factor', o['ratio'], 3.47, 5e-3),
 ('LHS converged', d[:, -1].sum(), 1872, 0), ('LHS sampled', len(d), 2000, 0),
 ('ten-seed Nu mean', nu.mean(), 43.5, 2e-3), ('ten-seed Nu min', nu.min(), 38.8, 2e-3), ('ten-seed Nu max', nu.max(), 45.7, 2e-3),
 ('Nu CV R2', s['cv_gbt']['Nu']['r2'][0], 0.945, 1e-3), ('Crane f\'\'(0)', v['crane'], -1.0, 2e-6),
 ('wide-box Nu', [r for r in de if r['tag'] == 'wide'][0]['Nu'], 99.1, 1e-3),
 ('D=0 Nu', [r for r in de if r['tag'] == 'D0'][0]['Nu'], 77.0, 1e-3),
 ('shear closure Nu', [r for r in de if r['tag'] == 'shear'][0]['Nu'], 34.2, 1e-3),
 ('bulk-flux Nu', [r for r in de if r['tag'] == 'nubulk'][0]['Nu'], 34.05, 1e-3),
 ('margin cap 0.5 Nu', [r for r in de if r['tag'] == 'wide_ds50'][0]['Nu'], 38.4, 2e-3),
 ('margin cap 0.95 Nu', [r for r in de if r['tag'] == 'wide_ds95'][0]['Nu'], 86.5, 2e-3),
 ('suction cap 1.5 Nu', [r for r in de if r['tag'] == 'cap15'][0]['Nu'], 21.95, 1e-3),
]
S7 = json.load(open(f'{R}/stage7.json')); S8 = json.load(open(f'{R}/stage8.json'))
checks += [('closure p=2 reference-design Nu', S7['2.0']['Nu_at_ref'], 42.16, 1e-3),
           ('conformal 90% coverage Nu', S8['Nu']['0.9'][0], 0.90, 3e-2),
           ('optima inside 90% interval (of 10)', sum(r['inside'] for r in S8['optima']), 8, 1e-12)]
bad = 0
for n, got, want, tol in checks:
    ok = abs(got - want) <= max(tol, 1e-12) * max(1, abs(want)); bad += not ok
    print(('OK  ' if ok else 'FAIL'), n, got, want)
raise SystemExit(bad)
