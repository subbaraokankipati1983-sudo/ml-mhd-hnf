from common import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold, train_test_split, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
a=np.load('lhs_raw.npy'); N=len(a); ok=a[:,-1]==1
# admissibility filter: keep converged cases that satisfy Du*Sr < k_hnf/k_f (well-posed far field)
kr=np.array([props(p/2,p/2)['kr'] for p in a[:,7]]); adm=a[:,5]*a[:,6]<kr
nconv=int(ok.sum()); ok=ok&adm; D=a[ok]; X=D[:,:8]; Y=D[:,8:12]
print('sampled',N,'converged',nconv,'admissible training set',int(ok.sum()),'failed',N-nconv)
T=['Cf','Cn','Nu','Sh']
def fwd(y,t): return np.arcsinh(y) if t in('Nu','Sh') else y
def inv(z,t): return np.sinh(z) if t in('Nu','Sh') else z
def gbt(): return GradientBoostingRegressor(n_estimators=600,max_depth=4,learning_rate=0.04,subsample=0.9,random_state=0)
def rf(): return RandomForestRegressor(n_estimators=300,random_state=0,n_jobs=1)
def ann(): return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,64),max_iter=1500,random_state=0,early_stopping=True))
def met(y,p):
    return dict(r2=r2_score(y,p),rmse=mean_squared_error(y,p)**.5,mae=mean_absolute_error(y,p),mape=float(np.mean(np.abs((y-p)/np.where(np.abs(y)<1e-9,1e-9,y)))*100))
res={'N':int(N),'conv':nconv,'train':int(ok.sum())}
# Table 9: repeated 5-fold CV (3 repeats for time), GBT
rkf=RepeatedKFold(n_splits=5,n_repeats=3,random_state=1)
cv={}
for j,t in enumerate(T):
    ms=[]
    for tr,te in rkf.split(X):
        m=gbt().fit(X[tr],fwd(Y[tr,j],t)); p=inv(m.predict(X[te]),t); ms.append(met(Y[te,j],p))
    cv[t]={k:(float(np.mean([q[k] for q in ms])),float(np.std([q[k] for q in ms]))) for k in ms[0]}
    print(t,cv[t]['r2'],flush=True)
res['cv_gbt']=cv
# Table 10: 80/10/10, three families
Xp,Xt,Yp,Yt=train_test_split(X,Y,test_size=0.10,random_state=7)
fam={'GBT':gbt,'RandomForest':rf,'ANN':ann}; t10={}
rkf2=RepeatedKFold(n_splits=5,n_repeats=2,random_state=2)
for t_i,t in enumerate(T):
    for fn,mk in fam.items():
        m=mk().fit(Xp,fwd(Yp[:,t_i],t)); te=met(Yt[:,t_i],inv(m.predict(Xt),t))
        r2s=[]
        for tr,va in rkf2.split(Xp):
            mm=mk().fit(Xp[tr],fwd(Yp[tr,t_i],t)); r2s.append(r2_score(Yp[va,t_i],inv(mm.predict(Xp[va]),t)))
        t10[f'{t}|{fn}']=dict(test=te,cv_r2=(float(np.mean(r2s)),float(np.std(r2s))))
        print(t,fn,round(te['r2'],3),round(np.mean(r2s),3),flush=True)
res['t10']=t10
# shipped models on all converged data
ship={t:gbt().fit(X,fwd(Y[:,j],t)) for j,t in enumerate(T)}
pickle.dump(ship,open('ship.pkl','wb'))
fi=ship['Nu'].feature_importances_; res['fi_nu']=dict(zip(NAMES,map(float,fi))); print(res['fi_nu'])
json.dump(res,open('stage2.json','w'),indent=1)
