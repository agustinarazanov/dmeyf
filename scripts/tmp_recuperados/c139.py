"""Etapa 2: entre los que SE VAN, separar BAJA+2 (paga) de BAJA+1 (no paga).
Train 202103+202104, test out of time 202106. Problema balanceado (~50/50)."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, lightgbm as lgb, competencia as c
from sklearn.metrics import roc_auc_score
t0=time.time()
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
se_van=d["clase_ternaria"].isin(["BAJA+1","BAJA+2"])
tr=d[se_van & d[c.fe.MES].isin([202103,202104])]
te=d[se_van & d[c.fe.MES].eq(202106)]
ytr=(tr["clase_ternaria"]=="BAJA+2").to_numpy().astype(int)
yte=(te["clase_ternaria"]=="BAJA+2").to_numpy().astype(int)
print(f"train {len(tr):,} que se van ({ytr.mean():.1%} son BAJA+2) -> test {len(te):,} ({yte.mean():.1%})")
P=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,
       feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
aucs=[]
for s in c.SEMILLAS:
    m=lgb.train({**P,"seed":s}, lgb.Dataset(tr[pred],label=ytr), num_boost_round=200)
    aucs.append(roc_auc_score(yte, m.predict(te[pred])))
print(f"\nAUC out of time para separar BAJA+2 de BAJA+1: {np.mean(aucs):.4f}  (sd {np.std(aucs):.4f})")
print("  0,50 = indistinguibles | el modelo principal saca ~0,88 separando los que se van del resto")
imp=sorted(zip(m.feature_name(), m.feature_importance("gain")), key=lambda x:-x[1])[:10]
print("\nlo que mas usa para separarlos:")
for n,g in imp: print(f"   {n:32s} {g:10,.0f}")
print(f"\nlisto en {time.time()-t0:.0f}s")
