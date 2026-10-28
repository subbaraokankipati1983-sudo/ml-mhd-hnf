"""Collocation BVP solver for the micropolar hybrid-nanofluid similarity system (V27).

Wall heat flux follows the conservation law q_w = -(k_hnf + k_d) dT/dy at y = 0, so the local Nusselt number is
Nu_x = -(k_hnf/k_f) (1 + D g(0)) theta'(0), with g = f' (velocity closure) or |f''| (shear closure).
Set nu_disp=False only to reproduce the earlier bulk-conductivity-only convention (diagnostic, not adopted)."""
import numpy as np
from scipy.integrate import solve_bvp

RHO = dict(f=997.1, cu=8933.0, al=3970.0)
CP  = dict(f=4179.0, cu=385.0, al=765.0)
K   = dict(f=0.613, cu=401.0, al=40.0)
SIG = dict(f=5.5e-6, cu=5.96e7, al=1e-10)

def props(phi1, phi2):
    rnf = (1-phi1)*RHO['f'] + phi1*RHO['cu']; rh = (1-phi2)*rnf + phi2*RHO['al']
    cnf = (1-phi1)*RHO['f']*CP['f'] + phi1*RHO['cu']*CP['cu']
    ch  = (1-phi2)*cnf + phi2*RHO['al']*CP['al']
    knf = K['f']*(K['cu']+2*K['f']-2*phi1*(K['f']-K['cu']))/(K['cu']+2*K['f']+phi1*(K['f']-K['cu']))
    kh  = knf*(K['al']+2*knf-2*phi2*(knf-K['al']))/(K['al']+2*knf+phi2*(knf-K['al']))
    r1 = SIG['cu']/SIG['f']; snf = SIG['f']*(1+3*phi1*(r1-1)/((r1+2)-phi1*(r1-1)))
    r2 = SIG['al']/snf;      sh  = snf*(1+3*phi2*(r2-1)/((r2+2)-phi2*(r2-1)))
    return dict(A0=rh/RHO['f'], A1=1/((1-phi1)**2.5*(1-phi2)**2.5), A2=ch/(RHO['f']*CP['f']),
                A3=sh/SIG['f'], kr=kh/K['f'])

def make_rhs(P, opt):
    A0,A1,A2,A3,kr = P['A0'],P['A1'],P['A2'],P['A3'],P['kr']
    Ha,Kv,S,D,Pr,Sc,Kr,r1,m,Sr,Du = (opt[k] for k in ('Ha','K','S','D','Pr','Sc','Kr','r1','m','Sr','Du'))
    kfac = kr if opt.get('kfactor',False) else 1.0
    inert_h = A0 if opt.get('A0micro',False) else A1
    energy_sign = opt.get('energy_unsteady','full')   # 'full': -S(eta/2 th' + m th)
    closure = opt.get('closure','velocity')
    def rhs(eta, y):
        f,fp,fpp,h,hp,th,thp,ch,chp = y
        fppp = -(A0*(f*fpp-fp**2-S*(eta/2*fpp+fp)) + Kv*hp - A3*Ha*fp)/(A1+Kv)
        hpp = -(inert_h*(f*hp-fp*h-S*(eta/2*hp+1.5*h)) - Kv*(2*h+fpp))/(1+Kv/2)
        # effective conductivity k_eff = k_hnf (1 + D g), g = f' (velocity-based) or |f''| (shear-based)
        if closure=='velocity':   c1 = kfac*(1 + D*fp);        c2 = kfac*D*fpp
        elif closure=='shear':    c1 = kfac*(1 + D*np.abs(fpp)); c2 = kfac*D*np.sign(fpp)*fppp
        elif closure=='power':
            p = opt.get('pexp',1.0); ap = np.abs(fp)+1e-12   # g = |f'|^p, p=1 is the adopted velocity closure
            c1 = kfac*(1 + D*ap**p); c2 = kfac*D*p*ap**(p-1)*np.sign(fp)*fpp
        else:                     c1 = kfac*np.ones_like(fp);  c2 = 0*fp
        if energy_sign=='full': un = -S*(eta/2*thp + m*th)
        else:                   un = -S*eta/2*thp + m*th
        Rt = -(c2*thp + Pr*A2*(f*thp - r1*fp*th + un))
        Rc = -(Sc*(f*chp - Kr*ch - S*eta/2*chp))
        # c1 th'' + Du ch'' = Rt ; Sr th'' + ch'' = Rc
        det = c1 - Du*Sr
        thpp = (Rt - Du*Rc)/det
        chpp = (c1*Rc - Sr*Rt)/det
        return np.vstack([fp,fpp,fppp,hp,hpp,thp,thpp,chp,chpp])
    return rhs

def solve(par, einf=10.0, nodes=400, tol=1e-8, guess=None, max_nodes=200000, **kw):
    """par: dict with Ha,K,S,fw,D,Sr,Du,phi (+optional phi1,phi2,Pr,Sc,Kr,r1,m,n and model options)."""
    d = dict(Pr=6.2,Sc=0.6,Kr=0.5,r1=2.0,m=1.5,n=0.5); d.update(par)
    phi1 = d.get('phi1', d.get('phi',0)/2); phi2 = d.get('phi2', d.get('phi',0)/2)
    P = props(phi1,phi2); rhs = make_rhs(P,d)
    fw,n = d['fw'], d['n']
    def bc(ya,yb):
        return np.array([ya[0]-fw, ya[1]-1, ya[3]+n*ya[2], ya[5]-1, ya[7]-1, yb[1], yb[3], yb[5], yb[7]])
    eta = np.linspace(0,einf,nodes)
    if guess is None:
        e=np.exp(-eta); y0=np.vstack([fw+1-e, e, -e, 0.3*e, -0.3*e, e, -e, e, -e])
    else:
        y0 = guess(eta)
    sol = solve_bvp(rhs,bc,eta,y0,tol=tol,max_nodes=max_nodes)
    if sol.status!=0: return None
    y0p = sol.sol(0)
    kk = P['kr'] if d.get('kfactor',False) else 1.0
    g0 = 1.0 if d.get('closure','velocity') in ('velocity','power') else abs(y0p[2])
    nu_flux = kk*(1 + (d['D']*g0 if d.get('nu_disp',True) else 0.0))
    cfac = P['A1'] + ((1-d['n'])*d['K'] if d.get('cf_total',False) else 0.0)
    out = dict(Cf=cfac*y0p[2], Cn=y0p[4], Nu=-nu_flux*y0p[6], Sh=-y0p[8], thp0=y0p[6], rms=float(np.max(sol.rms_residuals)), sol=sol, P=P)
    return out


def det_profile(out, par):
    """Determinant c1 - Du*Sr of the (theta'', chi'') system along the solution (velocity/shear closure)."""
    d = dict(Sr=par['Sr'], Du=par['Du'], D=par['D']); sol = out['sol']
    kfac = out['P']['kr'] if par.get('kfactor', False) else 1.0
    eta = np.linspace(0, sol.x[-1], 2000); y = sol.sol(eta)
    g = y[1] if par.get('closure','velocity')=='velocity' else np.abs(y[2])
    return eta, kfac*(1 + d['D']*g) - d['Du']*d['Sr']
