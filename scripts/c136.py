import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time(); d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
NBR=250; CAR=c.EXPERIMENTOS/"c134_mora"; sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
# fold A: entrena 202103 (limpio), valida 202105 (ROTO, como agosto)
X,y,w=c.preparar(d,[202103],"pesos",0.25); X=X[pred]
v=d[d[c.fe.MES]==202105]; real=c.aporte(v["clase_ternaria"].to_numpy()=="BAJA+2")
iny=v["Visa_Finiciomora"].eq(0).to_numpy()
for K in (9000,10000,11000):
    con_ellos,sin_ellos,cuantos=[],[],[]
    for s in sem:
        m=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"crudo_202105_{s}_{c.clave(P,[202103],'pesos',0.25,'crudo',NBR,s)}")
        o=np.argsort(-m.predict(v[pred]))
        top=o[:K]; con_ellos.append(real[top].sum()/1e6); cuantos.append(int(iny[top].sum()))
        limpio=o[~iny[o]][:K]; sin_ellos.append(real[limpio].sum()/1e6)   # saltea inyectados, toma los K siguientes
    a,b=np.array(sin_ellos),np.array(con_ellos)
    print(f"  K={K:6,}  inyectados dentro: {np.mean(cuantos):5.1f}  con ellos {b.mean():6.1f}M  "
          f"sin ellos {a.mean():6.1f}M  dif {a.mean()-b.mean():+5.2f}M  {int((a>b).sum())}/10  "
          f"p={wilcoxon(a,b).pvalue:.3f}")
print(f"listo en {time.time()-t0:.0f}s")
