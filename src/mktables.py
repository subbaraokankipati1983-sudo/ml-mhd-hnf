"""Builds tables.json: every number and table body quoted in the manuscript, taken from the archived results."""
import json, numpy as np
from model import props
s = json.load(open('stage2.json')); v = json.load(open('val.json')); o = json.load(open('stage5.json'))
a6 = json.load(open('stage6.json')); ch = json.load(open('cheb.json'))
R = [json.loads(l) for l in open('de_results.jsonl')]
a = np.load('lhs_raw.npy')
def f(x, n=3): return (f"{x:.{n}f}".replace('-', '$-$')) if x < 0 else f"{x:.{n}f}"
def tag(t, seed=1): return [r for r in R if r['tag'] == t and r['seed'] == seed][0]
T = {}
# ---------- validation tables
rows = []
for k, (e, n) in {'10_400': (10, 400), '14_400': (14, 400), '10_800': (10, 800), '14_800': (14, 800)}.items():
    q = v['mesh'][k]; rows.append(f"$\\eta_\\infty={e}$, {n} nodes & {f(q['Cf'],6)} & {f(q['Cn'],6)} & {f(q['Nu'],6)} & {f(q['Sh'],6)} \\\\")
T['T_mesh'] = '\n'.join(rows)
def sci(x):
    if x == 0 or x < 1e-15: return '$<10^{-15}$'
    m, e = f"{x:.1e}".split('e'); return f"${m}\\times10^{{{int(e)}}}$"
T['T_pavlov'] = '\n'.join(f"{h} & {f(d[0],6)} & {f(d[1],6)} & {sci(d[2])} \\\\" for h, d in v['pavlov'].items())
rkref = {'1': 1.333346, '3': 2.509682, '10': 4.796226}
T['T_rk'] = '\n'.join(f"{p} & {v['rk'][p]:.6f} & {rkref[p]:.6f} & {sci(abs(v['rk'][p]-rkref[p]))} \\\\" for p in rkref)
T['crane'] = f"{v['crane']:.6f}".replace('-', '$-$')
P3 = v['props03']
for k in ('A0', 'A1', 'A2', 'A3'): T[k] = f"{P3[k]:.4f}"
T['A4'] = f"{P3['kr']:.4f}"
T['cheb_base'] = '$'+f"{ch['baseline']['rel']:.0e}".replace('e-0', 'e-').replace('e-', '\\times10^{-') + '}$'
T['cheb_opt'] = '$'+f"{ch['optimum']['rel']:.0e}".replace('e-0', 'e-').replace('e-', '\\times10^{-') + '}$'
T['cheb_n'] = str(ch['n_lhs_compared']); T['cheb_max'] = '$'+f"{ch['max_rel_lhs']:.0e}".replace('e-0', 'e-').replace('e-', '\\times10^{-') + '}$'
# ---------- LHS accounting
X = a[:, :8]; ok = a[:, -1] == 1; kr = np.array([props(p/2, p/2)['kr'] for p in X[:, 7]]); ratio = X[:, 5]*X[:, 6]/kr
T['N'] = str(len(a)); T['nconv'] = str(int(ok.sum())); T['nfail'] = str(int((~ok).sum())); T['nadm'] = str(int((ratio < 1).sum()))
T['nadm_conv'] = str(int((ok & (ratio < 1)).sum())); T['nviol_conv'] = str(int((ok & (ratio >= 1)).sum()))
b1 = (ratio >= 1) & (ratio < 1.2); b2 = ratio >= 1.2
T['band1_n'] = str(int(b1.sum())); T['band1_fail'] = str(int((b1 & ~ok).sum())); T['band2_n'] = str(int(b2.sum())); T['band2_fail'] = str(int((b2 & ~ok).sum()))
vc = a6['violating_converged']; T['viol_ratio_lo'] = f"{min(q['DuSr']/q['kr'] for q in vc):.2f}"; T['viol_ratio_hi'] = f"{max(q['DuSr']/q['kr'] for q in vc):.2f}"
T['viol_flips'] = str(sum(q['negative_everywhere'] for q in vc)); T['adm_min_det'] = f"{a6['admissible_min_det']:.4f}"; T['adm_flips'] = str(a6['admissible_det_sign_change'])
T['viol_Dmax'] = f"{max(a[q['row'], 4] for q in vc):.2f}"
# ---------- surrogates
tn = {'Cf': '$C_f$', 'Cn': '$C_n$', 'Nu': '$\\text{Nu}_x$', 'Sh': '$\\text{Sh}_x$'}
c = s['cv_gbt']
T['T_cv'] = '\n'.join(f"{tn[t]} & ${c[t]['r2'][0]:.3f}\\pm{c[t]['r2'][1]:.3f}$ & ${c[t]['rmse'][0]:.3f}\\pm{c[t]['rmse'][1]:.3f}$ & ${c[t]['mae'][0]:.3f}\\pm{c[t]['mae'][1]:.3f}$ & ${c[t]['mape'][0]:.1f}\\%\\pm{c[t]['mape'][1]:.1f}\\%$ \\\\" for t in tn)
T['ntrain'] = str(s['train']); T['r2nu'] = f"{c['Nu']['r2'][0]:.2f}"; T['r2sh'] = f"{c['Sh']['r2'][0]:.2f}"; T['r2cf'] = f"{c['Cf']['r2'][0]:.3f}"
rows = []
for t in tn:
    best_te = max(s['t10'][f'{t}|{m}']['test']['r2'] for m in ('GBT', 'RandomForest', 'ANN')); best_cv = max(s['t10'][f'{t}|{m}']['cv_r2'][0] for m in ('GBT', 'RandomForest', 'ANN'))
    for i, m in enumerate(('GBT', 'RandomForest', 'ANN')):
        d = s['t10'][f'{t}|{m}']; r2 = f"{d['test']['r2']:.3f}"; cv = f"{d['cv_r2'][0]:.3f}\\pm{d['cv_r2'][1]:.3f}"
        if d['test']['r2'] == best_te: r2 = f"\\textbf{{{r2}}}"
        cvs = f"$\\mathbf{{{cv}}}$" if d['cv_r2'][0] == best_cv else f"${cv}$"
        first = f"\\multirow{{3}}{{*}}{{{tn[t]}}}" if i == 0 else ""
        rows.append(f"{first} & {m if m != 'ANN' else 'ANN (MLP)'} & {r2} & {d['test']['rmse']:.3f} & {d['test']['mape']:.1f}\\% & {cvs} \\\\")
    if t != 'Sh': rows.append("\\midrule")
T['T_t10'] = '\n'.join(rows)
T['nu_test_gbt'] = f"{s['t10']['Nu|GBT']['test']['r2']:.3f}"; T['nu_cv_gbt'] = f"${s['t10']['Nu|GBT']['cv_r2'][0]:.3f}\\pm{s['t10']['Nu|GBT']['cv_r2'][1]:.3f}$"
fi = s['fi_nu']; T['fi_fw'] = f"{fi['fw']:.2f}"; T['fi_D'] = f"{fi['D']:.2f}"; T['fi_Sr'] = f"{fi['Sr']:.2f}"; T['fi_Du'] = f"{fi['Du']:.2f}"; T['fi_other'] = f"{max(fi[k] for k in ('Ha','K','S','phi')):.3f}"
ab = a6['ablation']
rows = []
for t in ('Cf', 'Cn', 'Nu', 'Sh'):
    x1, x2 = ab[f'{t}|admissible_only'], ab[f'{t}|all_converged']
    rows.append(f"{tn[t]} & ${x1[0]:.3f}\\pm{x1[1]:.3f}$ & ${x2[0]:.3f}\\pm{x2[1]:.3f}$ & ${x1[0]-x2[0]:+.3f}$ \\\\".replace('+-', '$-$').replace('$+', '$+'))
T['T_abl'] = '\n'.join(rows)
T['abl_nu_diff'] = f"{abs(ab['Nu|admissible_only'][0]-ab['Nu|all_converged'][0]):.3f}"; T['abl_sh_diff'] = f"{abs(ab['Sh|admissible_only'][0]-ab['Sh|all_converged'][0]):.3f}"
# ---------- optimum
xo = np.array(o['xo']); ref = tag('ref'); T['xo_Ha'] = f"{xo[0]:.3f}"; T['xo_K'] = f"{xo[1]:.3f}"; T['xo_S'] = f"{xo[2]:.3f}"; T['xo_fw'] = f"{xo[3]:.3f}"; T['xo_D'] = f"{xo[4]:.3f}"; T['xo_Sr'] = f"{xo[5]:.3f}"; T['xo_Du'] = f"{xo[6]:.3f}"; T['xo_phi'] = f"{xo[7]:.4f}"
T['opt_Cf'] = f(o['opt'][0], 3); T['opt_Cn'] = f(o['opt'][1], 3); T['opt_Nu'] = f"{o['opt'][2]:.3f}"; T['opt_Sh'] = f(o['opt'][3], 3); T['opt_res'] = f"{ref['rms']:.1e}".replace('e-0', 'e-').replace('e-', '\\times10^{-') + '}'
T['nu_base'] = f"{o['base'][0]:.2f}"; T['nu_opt'] = f"{o['opt'][2]:.1f}"; T['ratio'] = f"{o['ratio']:.2f}"; T['ratio1'] = f"{o['ratio']:.1f}"
OPT = np.array([[0, 5], [0, 3], [0, 0.4], [0.5, 3], [0.25, 2], [0, 1.0], [0, 0.8], [0, 0.08]])
pos = [(xo[i]-OPT[i, 0])/(OPT[i, 1]-OPT[i, 0]) for i in range(8)]
T['pos_fw'] = f"{100*pos[3]:.0f}"; T['pos_Du'] = f"{100*pos[6]:.0f}"; T['pos_Sr'] = f"{100*pos[5]:.0f}"; T['pos_D'] = f"{100*pos[4]:.1f}"
T['ds_ratio'] = f"{xo[5]*xo[6]/props(xo[7]/2, xo[7]/2)['kr']:.2f}"
T['pen_cf'] = f"{0.15*abs(o['opt'][0]):.2f}"; T['pen_sh'] = f"{0.05*abs(o['opt'][3]):.2f}"
dec = o['decomp']; b = dec[0][1]
lab = {'fw': f'+ suction $f_w$: $1.5\\to{xo[3]:.3f}$', 'Du': f'+ Dufour Du: $0.3\\to{xo[6]:.3f}$', 'Sr': f'+ Soret Sr: $0.5\\to{xo[5]:.3f}$', 'K': f'+ micropolarity $K$: $0.5\\to{xo[1]:.3f}$', 'Ha,S': '+ Ha, $S$ (optimized values)', 'D': f'+ dispersion $D$: $1\\to{xo[4]:.3f}$', 'phi': f'+ loading $\\phi$: $0.06\\to{xo[7]:.4f}$'}
rows = [f"Baseline & {b:.2f} & --- \\\\"] + [f"{lab[nm]} & {val:.2f} & ${100*(val/b-1):+.0f}\\%$ \\\\" for nm, val in dec[1:]]
T['T_decomp'] = '\n'.join(rows)
d = dict(dec[1:]); T['dec_fw'] = f"{d['fw']:.1f}"; T['dec_cross'] = f"{d['Sr']:.1f}"; T['dec_D'] = f"{d['D']:.1f}"; T['dec_phi_gain'] = f"{100*(dec[-1][1]/d['D']-1):.0f}"
# ---------- weights
rows = []; W = {}
for t, l in (('ref', 'Base (0.15, 0.05)'), ('w2cf', '$2\\times C_f$ (0.30, 0.05)'), ('whcf', '$0.5\\times C_f$ (0.075, 0.05)'), ('w2sh', '$2\\times\\text{Sh}$ (0.15, 0.10)'), ('whsh', '$0.5\\times\\text{Sh}$ (0.15, 0.025)')):
    q = tag(t); x = q['x']; W[t] = q['Nu']
    rows.append(f"{l} & {x[3]:.3f} & {x[4]:.3f} & {x[7]:.4f} & {f(q['Cf'],3)} & {f(q['Cn'],3)} & {q['Nu']:.3f} & {f(q['Sh'],3)} \\\\")
T['T_weights'] = '\n'.join(rows)
T['w_lo'] = f"{min(W.values()):.1f}"; T['w_hi'] = f"{max(W.values()):.1f}"; T['wmin'] = f"{100*(min(W.values())/W['ref']-1):+.1f}"; T['wmax'] = f"{100*(max(W.values())/W['ref']-1):+.1f}"
wx = [tag(t)['x'] for t in ('ref', 'w2cf', 'whcf', 'w2sh', 'whsh')]
T['w_fw_lo'] = f"{min(x[3] for x in wx):.2f}"; T['w_fw_hi'] = f"{max(x[3] for x in wx):.2f}"; T['w_D_lo'] = f"{min(x[4] for x in wx):.2f}"; T['w_D_hi'] = f"{max(x[4] for x in wx):.2f}"
S = o['sens']
T['T_sens'] = '\n'.join(f"${int(round(100*S['D'][i][0])):+d}\\%$ & {S['D'][i][1]:.3f} & {S['D'][i][2]:.3f} & {S['fw'][i][1]:.3f} & {S['fw'][i][2]:.3f} \\\\".replace('$+0\\%$', '$0\\%$') for i in range(5))
T['k_var'] = f"{100*(max(x[1] for x in S['K'])-min(x[1] for x in S['K']))/S['K'][2][1]:.2f}"; T['du_lo'] = f"{S['Du'][0][1]:.1f}"; T['du_hi'] = f"{S['Du'][4][1]:.1f}"
# ---------- seeds
rf = [r for r in R if r['tag'] == 'ref']; XS = np.array([r['x'] for r in rf]); nu = np.array([r['Nu'] for r in rf]); Jv = np.array([r['J'] for r in rf])
def st(z, d=3): return f"{z.mean():.{d}f} & {z.std(ddof=1):.{d}f} & {z.min():.{d}f} & {z.max():.{d}f}"
rows = [f"$\\text{{Nu}}_x$ & {st(nu)} \\\\", f"$J$ & {st(Jv)} \\\\"]
for nm, i, dd in (('$f_w$', 3, 3), ('$D$', 4, 4), ('Du', 6, 3), ('Sr', 5, 3), ('$K$', 1, 3), ('Ha', 0, 3), ('$S$', 2, 3), ('$\\phi$', 7, 4)):
    z = XS[:, i]; rows.append(f"{nm} & {z.mean():.{dd}f} & {z.std(ddof=1):.{dd}f} & {z.min():.{dd}f} & {z.max():.{dd}f} \\\\")
T['T_seeds'] = '\n'.join(rows)
T['nu_mean'] = f"{nu.mean():.1f}"; T['nu_smin'] = f"{nu.min():.1f}"; T['nu_smax'] = f"{nu.max():.1f}"; T['cvJ'] = f"{100*Jv.std(ddof=1)/Jv.mean():.1f}"
bs = max(rf, key=lambda r: r['J']); T['bestseed'] = str(bs['seed']); T['bestNu'] = f"{bs['Nu']:.2f}"; T['bestJ'] = f"{bs['J']:.2f}"; T['best_pct'] = f"{100*(bs['Nu']/ref['Nu']-1):.0f}"
T['fw_cv'] = f"{100*XS[:,3].std(ddof=1)/XS[:,3].mean():.0f}"; T['D_cv'] = f"{100*XS[:,4].std(ddof=1)/XS[:,4].mean():.0f}"; T['Du_cv'] = f"{100*XS[:,6].std(ddof=1)/XS[:,6].mean():.0f}"
T['phi_small'] = ('all' if (XS[:, 7] < 0.02).all() else str(int((XS[:, 7] < 0.02).sum()))); T['phi_max'] = f"{XS[:,7].max():.3f}"
T['seed_fw_lo'] = f"{XS[:,3].min():.2f}"; T['seed_Sr_lo'] = f"{XS[:,5].min():.2f}"; T['seed_Du_lo'] = f"{XS[:,6].min():.2f}"
# ---------- stress tests
def g(t): q = tag(t); return q
T['nu_D0'] = f"{g('D0')['Nu']:.1f}"; T['nu_shear'] = f"{g('shear')['Nu']:.1f}"; T['nu_phifloor'] = f"{g('phifloor')['Nu']:.1f}"
T['nu_bulk_opt'] = f"{g('nubulk')['Nu']:.1f}"; T['D_bulk'] = f"{g('nubulk')['x'][4]:.3f}"; T['D_shear'] = f"{g('shear')['x'][4]:.3f}"
T['nu_hc6'] = f"{o['hc6'][1]:.2f}"; T['hc6_pct'] = f"{100*(o['hc6'][1]/o['opt'][2]-1):+.1f}"; T['nu_ein'] = f"{o['einstein'][1]:.2f}"; T['ein_rel'] = f"{abs(o['einstein'][1]/o['opt'][2]-1):.0e}"
T['nu_bulk_pt'] = f"{o['nu_bulk'][0]:.1f}"; T['bulk_pct'] = f"{100*(1-o['nu_bulk'][0]/o['opt'][2]):.0f}"; T['nu_v24_pt'] = f"{o['v24_model'][0]:.1f}"; T['v24_pct'] = f"{100*abs(o['v24_model'][0]/o['opt'][2]-1):.0f}"
T['nu_shear_pt'] = f"{o['shear_closure'][0]:.1f}"
T['phifloor_pct'] = f"{100*(1-g('phifloor')['Nu']/ref['Nu']):.0f}"
def stress(lbl, t, treat): q = g(t); return f"{lbl} & {treat} & {q['Nu']:.2f} & {q['x'][4]:.3f} \\\\"
rows = [stress('Reference ($k_d\\propto f^{\\prime}$, flux with $k_{hnf}+k_d$)', 'ref', 're-optimized'), stress('No dispersion ($D=0$)', 'D0', 're-optimized'),
        stress('Shear-based dispersion, $k_d=D\\,k_{hnf}|f^{\\prime\\prime}|$', 'shear', 're-optimized'), stress('Loading floor $\\phi\\ge0.02$', 'phifloor', 're-optimized'),
        stress('Molecular-only wall flux, $-A_4\\theta^{\\prime}(0)$ (not adopted)', 'nubulk', 're-optimized'),
        f"Hamilton--Crosser conductivity ($n=6$) & re-solved & {o['hc6'][1]:.2f} & {xo[4]:.3f} \\\\", f"Einstein viscosity & re-solved & {o['einstein'][1]:.2f} & {xo[4]:.3f} \\\\",
        f"Shear-based dispersion at the reference point & re-solved & {o['shear_closure'][0]:.2f} & {xo[4]:.3f} \\\\"]
T['T_stress'] = '\n'.join(rows)
# ---------- boxes and caps
w = tag('wide'); wk = props(w['x'][7]/2, w['x'][7]/2)['kr']; T['wide_Nu'] = f"{w['Nu']:.1f}"; T['wide_fw'] = f"{w['x'][3]:.2f}"; T['wide_D'] = f"{w['x'][4]:.3f}"; T['wide_Sr'] = f"{w['x'][5]:.2f}"; T['wide_Du'] = f"{w['x'][6]:.2f}"; T['wide_phi'] = f"{w['x'][7]:.3f}"; T['wide_Sh'] = f"{abs(w['Sh']):.1f}"; T['wide_ds'] = f"{w['x'][5]*w['x'][6]/wk:.2f}"
w2 = tag('wide_ds90'); k2 = props(w2['x'][7]/2, w2['x'][7]/2)['kr']; T['wds_Nu'] = f"{w2['Nu']:.1f}"; T['wds_fw'] = f"{w2['x'][3]:.2f}"; T['wds_ds'] = f"{w2['x'][5]*w2['x'][6]/k2:.2f}"; T['wds_D'] = f"{w2['x'][4]:.3f}"; T['wds_phi'] = f"{w2['x'][7]:.3f}"

rows=[]
for tg,cap in [('wide_ds50','0.5'),('wide_ds70','0.7'),('wide_ds80','0.8'),('wide_ds90','0.9'),('wide_ds95','0.95'),('wide','none')]:
    r=tag(tg); kk=props(r['x'][7]/2,r['x'][7]/2)['kr']
    rows.append(f"{cap} & {r['x'][3]:.2f} & {r['x'][4]:.2f} & {r['x'][5]*r['x'][6]/kk:.2f} & {r['x'][7]:.3f} & {r['Nu']:.1f} & {r['Sh']:.2f} \\\\")
T['T_margin']='\n'.join(rows)
T['m50_Nu']=f"{tag('wide_ds50')['Nu']:.1f}"; T['m70_Nu']=f"{tag('wide_ds70')['Nu']:.1f}"; T['m95_Nu']=f"{tag('wide_ds95')['Nu']:.1f}"
rows = []; capdat = []
for t, cap in (('cap10', 1.0), ('cap15', 1.5), ('cap20', 2.0), ('cap25', 2.5), ('ref', 3.0)):
    q = tag(t); rows.append(f"{cap:.1f} & {q['x'][3]:.3f} & {q['x'][4]:.3f} & {q['x'][7]:.4f} & {q['Nu']:.2f} & {f(q['Cf'],3)} & {f(q['Sh'],2)} \\\\"); capdat.append((cap, q['Nu'], q['x'][3]))
T['T_caps'] = '\n'.join(rows); json.dump(capdat, open('caps.json', 'w'))
T['cap10_Nu'] = f"{tag('cap10')['Nu']:.1f}"; T['cap15_Nu'] = f"{tag('cap15')['Nu']:.1f}"; T['cap20_Nu'] = f"{tag('cap20')['Nu']:.1f}"; T['cap25_Nu'] = f"{tag('cap25')['Nu']:.1f}"
T['cap10_gain'] = f"{tag('cap10')['Nu']/o['base'][0]:.1f}"; T['cap15_gain'] = f"{tag('cap15')['Nu']/o['base'][0]:.1f}"
T['cap15_pct'] = f"{100*tag('cap15')['Nu']/ref['Nu']:.0f}"
# ---------- sobol / pareto
so = o['sobol']; nm = ['Ha', 'K', 'S', 'fw', 'D', 'Sr', 'Du', 'phi']
T['sob_nu_fw'] = f"{so['Nu']['ST'][3]:.2f}"; T['sob_nu_D'] = f"{so['Nu']['ST'][4]:.2f}"; T['sob_nu_Sr'] = f"{so['Nu']['ST'][5]:.2f}"; T['sob_nu_Du'] = f"{so['Nu']['ST'][6]:.2f}"
T['sob_nu_other'] = f"{max(so['Nu']['ST'][i] for i in (0,1,2,7)):.2f}"
T['sob_sh_Sr'] = f"{so['Sh']['ST'][5]:.2f}"; T['sob_sh_D'] = f"{so['Sh']['ST'][4]:.2f}"; T['sob_sh_fw'] = f"{so['Sh']['ST'][3]:.2f}"; T['sob_sh_Du'] = f"{so['Sh']['ST'][6]:.2f}"
sp = np.array(so['Nu']['seed_spread']); T['sob_rep'] = f"{np.abs(sp - np.array(so['Nu']['ST'])).max():.2f}"
Pv = np.array(o['pareto']); T['par_n'] = str(len(Pv)); T['par_eval'] = str(o['pareto_n_eval']); i = np.argmax(Pv[:, 8]); T['par_maxNu'] = f"{Pv[i,8]:.1f}"; T['par_maxNu_Sh'] = f"{Pv[i,10]:.1f}"
lowsh = Pv[Pv[:, 10] < 1]; T['par_lowSh_Nu'] = f"{lowsh[:,8].max():.1f}" if len(lowsh) else 'n/a'
T['par_cf_lo'] = f"{Pv[:,9].min():.1f}"; T['par_cf_hi'] = f"{Pv[:,9].max():.1f}"
# ---------- results section
T['T_Ha'] = '\n'.join(f"{h} & {f(q['Cf'],4)} & {f(q['Cn'],4)} & {q['Nu']:.4f} & {f(q['Sh'],4)} \\\\" for h, q in v['Ha'].items())
T['T_Dtab'] = '\n'.join(f"{dd} & {f(q['Cf'],4)} & {f(q['Cn'],4)} & {q['Nu']:.4f} & {f(q['Sh'],4)} & {f(q['thp'],4)} \\\\" for dd, q in v['Dtab'].items())
H = v['Ha']; T['Ha_cf0'] = f"{abs(H['0']['Cf']):.2f}"; T['Ha_cf5'] = f"{abs(H['5']['Cf']):.2f}"; T['Ha_nu_pct'] = f"{100*(1-H['5']['Nu']/H['0']['Nu']):.1f}"
Dt = v['Dtab']; T['D_nu_lo'] = f"{min(q['Nu'] for q in Dt.values()):.2f}"; T['D_nu_hi'] = f"{max(q['Nu'] for q in Dt.values()):.2f}"; T['D_nu_spread'] = f"{100*(max(q['Nu'] for q in Dt.values())/min(q['Nu'] for q in Dt.values())-1):.1f}"
T['D_thp_01'] = f"{abs(Dt['0.1']['thp']):.2f}"; T['D_thp_2'] = f"{abs(Dt['2']['thp']):.2f}"
Sr_ = v['Sr']; T['Sh_sr0'] = f"{Sr_['0']['Sh']:.2f}"; T['Sh_sr15'] = f"{Sr_['1.5']['Sh']:.2f}"; T['Nu_sr0'] = f"{Sr_['0']['Nu']:.2f}"; T['Nu_sr15'] = f"{Sr_['1.5']['Nu']:.2f}"
ps = np.load('phi_sweep.npy'); T['phi_cf0']=f"{abs(ps[0,1]):.2f}"; T['phi_cf8']=f"{abs(ps[-1,1]):.2f}"; T['phi_nu0']=f"{ps[0,2]:.2f}"; T['phi_nu8']=f"{ps[-1,2]:.2f}"
T['abl_max']=f"{max(abs(ab[t+'|admissible_only'][0]-ab[t+'|all_converged'][0]) for t in ('Cf','Cn','Nu','Sh')):.3f}"
T['sensD_lo']=f"{min(x[1] for x in S['D']):.1f}"; T['sensD_hi']=f"{max(x[1] for x in S['D']):.1f}"; T['sensfw_lo']=f"{min(x[1] for x in S['fw']):.1f}"; T['sensfw_hi']=f"{max(x[1] for x in S['fw']):.1f}"
json.dump(T, open('tables.json', 'w'), indent=1); print(len(T), 'keys')
