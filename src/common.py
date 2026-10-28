import warnings; warnings.filterwarnings("ignore")
import numpy as np; np.seterr(all="ignore")
import numpy as np, json, time, sys, os
from scipy.stats import qmc
from model import solve, props, det_profile
NAMES=['Ha','K','S','fw','D','Sr','Du','phi']
SAMP=np.array([[0,5],[0,3],[0,0.6],[0.5,4],[0.1,2],[0,1.5],[0,1.15],[0,0.08]])
OPT =np.array([[0,5],[0,3],[0,0.4],[0.5,3],[0.25,2],[0,1.0],[0,0.8],[0,0.08]])
MODEL=dict(kfactor=True, A0micro=True, closure='velocity', nu_disp=True)   # V27 model: wall flux includes the dispersion conductivity
def pdict(x, **extra):
    d=dict(zip(NAMES,[float(v) for v in x])); d.update(MODEL); d.update(extra); return d
def run(x, einf=10.0, nodes=400, tol=1e-6, **extra):
    try:
        return solve(pdict(x,**extra), einf=einf, nodes=nodes, tol=tol, max_nodes=int(os.environ.get("MAXN","30000")))
    except Exception as e:
        return None
def J(r, wcf=0.15, wsh=0.05):
    return r['Nu']-wcf*abs(r['Cf'])-wsh*abs(r['Sh'])
