"""Clean-clone smoke test (a few seconds): re-solves three cases with the shipped solver and
compares them with the archived/paper values. It does NOT regenerate the whole study
(use reproduce_all.py for that, about 45 min). Run from the repository root:  python quick_check.py"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from common import run, NAMES
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
base = [1, 0.5, 0.2, 1.5, 1.0, 0.5, 0.3, 0.06]
ref = [json.loads(l) for l in open(f'{R}/de_results.jsonl')]
ref = [r for r in ref if r['tag'] == 'ref' and r['seed'] == 1][0]
checks = []
r = run(base, einf=10, nodes=400, tol=1e-8); checks.append(('baseline Nu', r['Nu'], 12.1614, 1e-4)); checks.append(('baseline Cf', r['Cf'], -2.5709, 1e-4))
crane = run([0, 0, 0, 0, 0, 0, 0, 0], einf=10, nodes=400, tol=1e-8, A0micro=True)
checks.append(("Crane f''(0) (Cf, A1=1)", crane['Cf'], -1.0, 1e-5))
r = run(ref['x'], einf=12, nodes=1000, tol=1e-8); checks.append(('reference optimum Nu', r['Nu'], ref['Nu'], 1e-5))
checks.append(('reference optimum Nu vs paper', r['Nu'], 42.20, 1e-3))
bad = 0
for n, got, want, tol in checks:
    ok = abs(got - want) <= tol * max(1, abs(want)); bad += not ok
    print('OK  ' if ok else 'FAIL', n, round(got, 6), want)
print('quick_check:', 'passed' if not bad else f'{bad} failed'); raise SystemExit(bad)
