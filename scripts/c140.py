"""Modelo de dos etapas contra el de una, en los dos folds.
  etapa 1: P(se va)            = BAJA+1 u BAJA+2 contra CONTINUA
  etapa 2: P(BAJA+2 | se va)   entrenado SOLO con los que se van
  score    = p1 * p2
Contra: baja2 puro y pesos25 (lo que mandamos hoy)."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, lightgbm as lgb, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
P={"objective":"binary","boosting_type":"gbdt","feature_pre_filter":False,"max_bin":31,
 "num_leaves":45,"learning_rate":0.0077,"min_data_in_leaf":174,"feature_fraction":0.277,
 "bagging_fraction":0.918,"bagging_freq":1,"verbose":-1}
P2=dict(objective="binary",learning_rate=0.03,num_leaves=15,min_data_in_leaf=40,
        feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,verbose=-1)
NBR=250; CAR=c.EXPERIMENTOS/"c140_dosetapas"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
FOLDS=[([202103],202105),([202103,202104],202106)]
res={}
for tr,va in FOLDS:
    sub=d[d[c.fe.MES].isin(tr)]; cl=sub["clase_ternaria"].to_numpy()
    v=d[d[c.fe.MES]==va]; es=(v["clase_ternaria"].to_numpy()=="BAJA+2")
    Xtr=sub[pred]
    y12=np.isin(cl,["BAJA+1","BAJA+2"]).astype(int)          # etapa 1
    mask=y12.astype(bool); X2=sub[pred][mask]; y2=(cl[mask]=="BAJA+2").astype(int)  # etapa 2
    ybaja2=(cl=="BAJA+2").astype(int)
    w25=np.where(cl=="BAJA+1",0.25,1.0)*np.isin(cl,["BAJA+1","BAJA+2"]).astype(float)
    g={n:{k:[] for k in (9000,10000,11000)} for n in ("baja2","pesos25","dos_etapas")}
    for s in sem:
        mb=lgb.train({**P,"seed":s}, lgb.Dataset(Xtr,label=ybaja2), num_boost_round=NBR)
        mp=lgb.train({**P,"seed":s}, lgb.Dataset(Xtr,label=np.isin(cl,["BAJA+1","BAJA+2"]).astype(int),
                      weight=np.where(cl=="BAJA+1",0.25,1.0)), num_boost_round=NBR)
        m1=lgb.train({**P,"seed":s}, lgb.Dataset(Xtr,label=y12), num_boost_round=NBR)
        m2=lgb.train({**P2,"seed":s}, lgb.Dataset(X2,label=y2), num_boost_round=200)
        sc={"baja2":mb.predict(v[pred]), "pesos25":mp.predict(v[pred]),
            "dos_etapas":m1.predict(v[pred])*m2.predict(v[pred])}
        for n,ss in sc.items():
            acum=c.ganancia_acumulada(ss,es)
            for k in g[n]: g[n][k].append(acum[k-1]/1e6)
    res[va]=g
    print(f"  fold {va}: "+" | ".join(f"{n} {np.mean(g[n][10000]):6.1f}M" for n in g)+f"   [{time.time()-t0:.0f}s]")
print("\n=== dos etapas contra cada uno, pareado por semilla, @10.000 ===")
for va in (202105,202106):
    for ref in ("baja2","pesos25"):
        a=np.array(res[va]["dos_etapas"][10000]); b=np.array(res[va][ref][10000])
        print(f"  {va} vs {ref:8s} {a.mean()-b.mean():+6.1f}M  {int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}")
print(f"\nlisto en {time.time()-t0:.0f}s")
