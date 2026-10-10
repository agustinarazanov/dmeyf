"""Un modelo POR MES, ensamblado por rank, contra un modelo sobre los meses juntos.
Entrenando juntos, las reglas que se contradicen entre meses pelean adentro del mismo
arbol. Entrenando por separado, cada modelo es coherente y se promedian las PREDICCIONES.
Se puede medir en el fold B (train 202103+202104) y en el fold C (train 03+04+05)."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c143_pormes"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
def entrenar(meses,s):
    X,y,w=c.preparar(d,meses,"pesos",0.25)
    return c.entrenar_o_cargar(P,X[pred],y,w,250,s,CAR,
        f"m{'_'.join(map(str,meses))}_{s}_{c.clave(P,meses,'pesos',0.25,'base150',250,s)}")
for TR,VA,tgt in [([202103,202104],202106,"BAJA+2"), ([202103,202104,202105],202107,"BAJA+1")]:
    v=d[d[c.fe.MES]==VA]; es=(v["clase_ternaria"].to_numpy()==tgt); n=es.sum()
    jun,sep=[],[]
    for s in sem:
        sc_j=entrenar(TR,s).predict(v[pred])
        sc_s=c.ensamble_por_rank({m:entrenar([m],s).predict(v[pred]) for m in TR})
        f=lambda sc:100*es[np.argsort(-sc)[:10000]].sum()/n
        jun.append(f(sc_j)); sep.append(f(sc_s))
    a,b=np.array(sep),np.array(jun)
    print(f"  valida {VA} ({tgt}, {n} positivos, train {TR})")
    print(f"    juntos      captura {b.mean():5.2f}%")
    print(f"    uno por mes captura {a.mean():5.2f}%   dif {a.mean()-b.mean():+5.2f} pts  "
          f"{int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
