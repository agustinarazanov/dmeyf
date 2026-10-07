"""hist_min a DISTANCIA 2 con target BAJA+2: la configuracion real de la competencia.
Limitacion conocida e inevitable: a distancia 2 solo se puede validar 202105 y 202106,
y eso obliga a entrenar con marzo/abril, donde las columnas de historia estan casi
vacias. O sea el test esta sesgado EN CONTRA. Se reporta igual."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c148_histmin_d2"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
datos={"base":c.cargar("competencia_01.parquet"),"hist_min":c.cargar("competencia_01_histmin.parquet")}
# cuanta historia tiene cada mes de train en cada fold
for TR,VA in [([202103],202105),([202103,202104],202106)]:
    out={}
    for nom,d in datos.items():
        pred=c.columnas_predictoras(d)
        X,y,w=c.preparar(d,TR,"pesos",0.25); X=X[pred]
        val=d[d[c.fe.MES]==VA]
        es=(val["clase_ternaria"].to_numpy()=="BAJA+2")     # de SU propio val
        out[nom]=np.array([c.ganancia_acumulada(c.entrenar_o_cargar(P,X,y,w,250,s,CAR,
            f"{nom}_{VA}_{s}_{c.clave(P,TR,'pesos',0.25,nom+str(len(pred)),250,s)}")
            .predict(val[pred]),es)[9999]/1e6 for s in sem])
    a,b=out["hist_min"],out["base"]
    nulos=100*datos["hist_min"][datos["hist_min"][c.fe.MES].isin(TR)]["ctrx_quarter__d1"].isna().mean()
    print(f"  train {str(TR):22s} -> {VA}   base {b.mean():6.1f}M | hist_min {a.mean():6.1f}M | "
          f"dif {a.mean()-b.mean():+6.1f}M  {int((a>b).sum()):2d}/10  p={wilcoxon(a,b).pvalue:.3f}"
          f"   ({nulos:.0f}% de las filas de train sin historia)   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
