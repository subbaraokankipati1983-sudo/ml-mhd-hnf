"""Independent verification: Chebyshev spectral collocation + Newton solver for the same similarity ODE system.
Shares only the property function and the printed equations with model.py; the discretisation and root finder are different (no solve_bvp)."""
import numpy as np
from scipy.optimize import root
from model import props

def cheb(N):
    x = np.cos(np.pi*np.arange(N+1)/N); c = np.r_[2, np.ones(N-1), 2]*(-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1)).T; dX = X - X.T
    D = np.outer(c, 1/c)/(dX + np.eye(N+1)); D -= np.diag(D.sum(axis=1)); return D, x

def solve_cheb(par, L=12.0, N=90, guess=None):
    d = dict(Pr=6.2, Sc=0.6, Kr=0.5, r1=2.0, m=1.5, n=0.5); d.update(par)
    phi1 = d.get('phi1', d.get('phi', 0)/2); phi2 = d.get('phi2', d.get('phi', 0)/2); P = props(phi1, phi2)
    A0, A1, A2, A3, kr = P['A0'], P['A1'], P['A2'], P['A3'], P['kr']
    Ha, K, S, D_, Pr, Sc, Kr, r1, m, Sr, Du, fw, n = (d[k] for k in ('Ha','K','S','D','Pr','Sc','Kr','r1','m','Sr','Du','fw','n'))
    Dm, x = cheb(N); eta = L*(1 - x)/2; D1 = -(2/L)*Dm; D2 = D1@D1; D3 = D2@D1
    M = N + 1
    def res(z):
        f, h, th, ch = z[:M], z[M:2*M], z[2*M:3*M], z[3*M:]
        fp, fpp, fppp = D1@f, D2@f, D3@f; hp, hpp = D1@h, D2@h; thp, thpp = D1@th, D2@th; chp, chpp = D1@ch, D2@ch
        e1 = (A1+K)*fppp + A0*(f*fpp - fp**2 - S*(eta/2*fpp + fp)) + K*hp - A3*Ha*fp
        e2 = (1+K/2)*hpp + A0*(f*hp - fp*h - S*(eta/2*hp + 1.5*h)) - K*(2*h + fpp)
        e3 = kr*(1 + D_*fp)*thpp + kr*D_*fpp*thp + Du*chpp + Pr*A2*(f*thp - r1*fp*th - S*(eta/2*thp + m*th))
        e4 = Sr*thpp + chpp + Sc*(f*chp - Kr*ch - S*eta/2*chp)
        # boundary rows (index 0 is eta=0, index N is eta=L)
        e1[0] = f[0] - fw; e1[1] = fp[0] - 1; e1[N] = fp[N]
        e2[0] = h[0] + n*fpp[0]; e2[N] = h[N]
        e3[0] = th[0] - 1; e3[N] = th[N]
        e4[0] = ch[0] - 1; e4[N] = ch[N]
        return np.r_[e1, e2, e3, e4]
    if guess is None:
        e = np.exp(-eta); z0 = np.r_[fw + 1 - e, 0.3*e, e, e]
    else: z0 = guess(eta)
    sol = root(res, z0, method='hybr', tol=1e-13, options=dict(maxfev=20000))
    z = sol.x; f, h, th, ch = z[:M], z[M:2*M], z[2*M:3*M], z[3*M:]
    fpp0 = (D2@f)[0]; hp0 = (D1@h)[0]; thp0 = (D1@th)[0]; chp0 = (D1@ch)[0]
    nu = -kr*(1 + D_*1.0)*thp0 if d.get('nu_disp', True) else -kr*thp0
    return dict(ok=bool(sol.success), resnorm=float(np.max(np.abs(res(z)))), Cf=A1*fpp0, Cn=hp0, Nu=nu, Sh=-chp0)
