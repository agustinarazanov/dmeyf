"""Validacion adversaria: ¿que tan distinto es 202108 de los meses de entrenamiento,
y QUE variables lo delatan? No usa etiquetas. Si una variable separa muy bien agosto
del train, el modelo la aprendio en un regimen y la aplica en otro.
Se corre sobre el dataset base y sobre el del FE, para ver si el FE agrega exposicion."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, lightgbm as lgb, competencia as c
from sklearn.model_selection import GroupShuffleSplit
t0=time.time()
P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,
       feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
TRAIN=[202103,202104,202105,202106]
for nom,ds in [("base","competencia_01.parquet"),("fe228","competencia_01_fe.parquet")]:
    d=c.cargar(ds); pred=c.columnas_predictoras(d)
    sub=d[d[c.fe.MES].isin(TRAIN+[202108])]
    y=(sub[c.fe.MES]==202108).astype("int8").to_numpy()
    X=sub[pred]; grupos=sub[c.fe.ID].to_numpy()
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=0.3,random_state=314159).split(X,y,grupos))
    m=lgb.train({**P,"seed":c.SEMILLAS[0]}, lgb.Dataset(X.iloc[tr],label=y[tr]), num_boost_round=200)
    from sklearn.metrics import roc_auc_score
    auc=roc_auc_score(y[te], m.predict(X.iloc[te]))
    imp=pd.DataFrame({"f":m.feature_name(),"g":m.feature_importance("gain")}).sort_values("g",ascending=False)
    print(f"\n{nom} ({len(pred)} col): AUC de separar 202108 del train = {auc:.4f}"
          f"   ({'indistinguible' if auc<0.6 else 'MUY separable: hay drift fuerte'})   [{time.time()-t0:.0f}s]")
    print(f"  las 12 que mas delatan a agosto:")
    tot=imp.g.sum()
    for _,r in imp.head(12).iterrows():
        print(f"    {r.f:38s} {100*r.g/tot:5.1f}% del gain")
