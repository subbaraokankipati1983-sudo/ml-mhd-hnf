"""Graphical abstract (1328 x 531 px aspect), drawn from the archived results."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from model import props
here = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(here, '..', 'figures') + os.sep
plt.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42, 'font.size': 15})
a = np.load('lhs_raw.npy'); ok = a[:, -1] == 1
R = [json.loads(l) for l in open('de_results.jsonl')]; nu = lambda t: [r for r in R if r['tag'] == t and r['seed'] == 1][0]['Nu']
base = json.load(open('val.json'))['baseline']['Nu']
fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
def box(ax, x, y, w, h, txt, fc, fs=15, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.03', fc=fc, ec='#333', lw=1.5))
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.25)
ax = fig.add_axes([0.0, 0.0, 0.47, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.02, 0.93, 'Solver in the loop', fontsize=19, fontweight='bold', va='center')
box(ax, 0.03, 0.62, 0.42, 0.2, 'DE proposes\n8 design variables', '#dce9f7')
box(ax, 0.03, 0.30, 0.42, 0.2, 'Nonlinear BVP solver\n(ninth order)', '#fbe3c8', bold=True)
box(ax, 0.55, 0.30, 0.42, 0.2, 'Verified Nu, Cf, Sh', '#d7efd9')
box(ax, 0.55, 0.62, 0.42, 0.2, 'ML surrogate:\nscreening only', '#eeeeee')
for (p, q) in (((0.24, 0.62), (0.24, 0.50)), ((0.45, 0.40), (0.55, 0.40)), ((0.76, 0.50), (0.76, 0.62))):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=20, color='#333', lw=2))
ax.text(0.5, 0.12, 'Wall flux uses k$_{hnf}$(1+D)', ha='center', fontsize=15)
ax2 = fig.add_axes([0.575, 0.2, 0.17, 0.6])
kr = np.array([props(p/2, p/2)['kr'] for p in a[:, 7]]); good = ok & (a[:, 5]*a[:, 6] < kr)
ax2.scatter(a[good, 5], a[good, 6], s=3, c='#0072B2'); ax2.scatter(a[~ok, 5], a[~ok, 6], s=12, c='#D55E00', marker='x')
sr = np.linspace(0.3, 1.5, 100); ax2.plot(sr, 1.1/sr, 'k', lw=1.5)
ax2.set_xlim(0, 1.5); ax2.set_ylim(0, 1.15); ax2.set_xlabel('Sr', fontsize=14); ax2.set_ylabel('Du', fontsize=14); ax2.tick_params(labelsize=12)
fig.text(0.51, 0.9, 'Admissible region', fontsize=19, fontweight='bold'); fig.text(0.51, 0.83, 'all failures: Du Sr $\\geq$ k$_{hnf}$/k$_f$', fontsize=13)
ax3 = fig.add_axes([0.82, 0.2, 0.17, 0.6])
labs = ['baseline', 'cap\nf$_w$$\\leq$1.5', 'optimum', 'D = 0']; vals = [base, nu('cap15'), nu('ref'), nu('D0')]
cols = ['#999999', '#0072B2', '#009E73', '#CC79A7']
ax3.bar(range(4), vals, color=cols)
for i, v in enumerate(vals): ax3.text(i, v+1.5, f'{v:.0f}', ha='center', fontsize=13)
ax3.set_xticks(range(4)); ax3.set_xticklabels(labs, fontsize=10); ax3.set_ylabel('Nu$_x$', fontsize=14); ax3.set_ylim(0, 90); ax3.tick_params(axis='y', labelsize=12)
for s in ('top', 'right'): ax3.spines[s].set_visible(False)
fig.text(0.79, 0.9, 'Best verified Nu', fontsize=19, fontweight='bold'); fig.text(0.79, 0.83, 'box-limited, model-bound', fontsize=13)
fig.savefig(O + 'graphical_abstract.pdf'); fig.savefig(O + 'graphical_abstract.png', dpi=200)
from PIL import Image; Image.open(O + 'graphical_abstract.png').convert('RGB').save(O + 'graphical_abstract.tiff', compression='tiff_lzw', dpi=(200, 200))
print(Image.open(O + 'graphical_abstract.png').size)
