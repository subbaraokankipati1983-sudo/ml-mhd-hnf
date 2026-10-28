"""Regenerates the V33 pipeline (about 45 min on two CPU cores). Run from the repository root."""
import subprocess, sys, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
for _f in ['de_results.jsonl']:  # stage3_de.py appends: start from an empty file so a re-run in a populated checkout is not doubled
    if os.path.exists(_f): os.remove(_f)
steps = [['stage1_lhs.py'], ['stage2_sur.py'], ['stage4_val.py'], ['stage6_admissibility.py']]
steps += [['stage3_de.py', 'ref', str(s)] for s in range(1, 11)]
steps += [['stage3_de.py', t, '1'] for t in ['w2cf','whcf','w2sh','whsh','D0','shear','nubulk','wide','wide_ds50','wide_ds70','wide_ds80','wide_ds90','wide_ds95','phifloor','cap10','cap15','cap20','cap25']]
steps += [['cheb_verify.py'], ['stage5.py'], ['stage7_closure_family.py'], ['stage8_conformal.py'], ['stage9_figs.py'], ['figs.py'], ['mktables.py']]
for st in steps:
    print('running', st, flush=True); subprocess.check_call([sys.executable] + st)
import shutil
os.makedirs('../results', exist_ok=True)
for f in ['de_results.jsonl','stage2.json','stage5.json','stage6.json','val.json','cheb.json','caps.json','stage7.json','stage8.json']:
    if os.path.exists(f): shutil.copy(f, '../results/' + f)
import numpy as np
_a = np.load('lhs_raw.npy'); np.savetxt('../results/lhs_dataset.csv', _a, delimiter=',', header='Ha,K,S,fw,D,Sr,Du,phi,Cf,Cn,Nu,Sh,converged', comments='', fmt='%.10g')
print('copied archived outputs to results/; now run: python ../verify_results.py')
