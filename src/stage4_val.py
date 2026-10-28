from common import *
from model import solve
out={}
base=dict(Ha=1,K=0.5,S=0.2,fw=1.5,D=1,Sr=0.5,Du=0.3,phi1=0.03,phi2=0.03,**MODEL)
def q(r): return dict(Cf=r['Cf'],Cn=r['Cn'],Nu=r['Nu'],Sh=r['Sh'],thp=r['thp0'])
# mesh/domain: four combinations
out['mesh']={f'{e}_{n}':q(solve(base,einf=e,nodes=n,tol=1e-8)) for e in (10,14) for n in (400,800)}
# Crane
c=solve(dict(Ha=0,K=0,S=0,fw=0,D=0,Sr=0,Du=0,phi1=0,phi2=0,n=0,**MODEL),einf=12,nodes=800,tol=1e-8); out['crane']=c['sol'].sol(0)[2]
# Pavlov
out['pavlov']={}
for Ha in (0.5,1,2,5):
    r=solve(dict(Ha=Ha,K=0,S=0,fw=0,D=0,Sr=0,Du=0,phi1=0,phi2=0,n=0,**MODEL),einf=14,nodes=800,tol=1e-10)
    fpp=r['sol'].sol(0)[2]; out['pavlov'][Ha]=(fpp,-np.sqrt(1+Ha),abs(fpp+np.sqrt(1+Ha)))
# Reddy-Krishna table 1
out['rk']={}
for Pr in (1,3,10):
    r=solve(dict(Ha=0,K=0,S=0,fw=0,D=0,Sr=0,Du=0,phi1=0,phi2=0,n=0,Pr=Pr,r1=2,**MODEL),einf=20,nodes=1000,tol=1e-10)
    out['rk'][Pr]=-r['thp0']
# Kameswaran-type limit (corrected model: Nu=kr*(-th'(0)), single Cu, phi1=0.05)
out['kam']={}
for D in (0.1,0.5,1,1.5,2):
    r=solve(dict(Ha=0,K=0,Sr=0,Du=0,S=0.2,fw=0,D=D,phi1=0.05,phi2=0,n=0,Pr=6.2,**MODEL),einf=12,nodes=600,tol=1e-8); out['kam'][D]=q(r)
# Reddy-Krishna limit phi=0 D=0
out['rklim']={}
for Ha in (0,1,3):
    r=solve(dict(Ha=Ha,K=0.5,S=0.2,fw=0.5,D=0,Sr=0.5,Du=0.3,phi1=0,phi2=0,n=0.5,**MODEL),einf=12,nodes=600,tol=1e-8); out['rklim'][Ha]=q(r)
# parametric tables at baseline
out['Ha']={Ha:q(solve({**base,'Ha':Ha},einf=10,nodes=400,tol=1e-8)) for Ha in (0,1,3,5)}
out['Dtab']={D:q(solve({**base,'D':D},einf=10,nodes=400,tol=1e-8)) for D in (0.1,0.3,0.5,1,1.5,2)}
out['Sr']={Sr:q(solve({**base,'Sr':Sr},einf=10,nodes=400,tol=1e-8)) for Sr in (0,0.5,1,1.5)}
b=solve(base,einf=10,nodes=400,tol=1e-8); out['baseline']=q(b); out['props03']=b['P']
p5=dict(base,phi1=0.025,phi2=0.025); out['props05']=solve(p5)['P']
# baseline with and without conductivity factor, for transparency
nk=dict(base); nk['kfactor']=False; nk['A0micro']=False; nk['nu_disp']=False; out['v24_baseline']=q(solve(nk,tol=1e-8))
json.dump(out,open('val.json','w'),indent=1,default=float)
print(json.dumps({k:out[k] for k in ('crane','pavlov','rk','baseline','props03','v24_baseline')},default=float,indent=0)[:1800])
