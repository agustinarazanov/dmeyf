"""La firma 'pierde en mayo / gana en junio' ¿es drift, o es el corte fijo?
Mayo tiene K*=8.092 y junio K*=14.051. Medir los dos a 10.000 mide a mayo
PASADO su optimo y a junio ANTES. Comparo tres formas de medir."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, competencia as c
from scipy.stats import wilcoxon
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c137_combo"; sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
FOLDS=[([202103],202105),([202103,202104],202106)]
S={}
for nom,ds in [("base","competencia_01.parquet"),("todo","competencia_01_todo.parquet")]:
    d=c.cargar(ds); pred=c.columnas_predictoras(d)
    for tr,va in FOLDS:
        X,y,w=c.preparar(d,tr,"pesos",0.25); X=X[pred]
        v=d[d[c.fe.MES]==va]; es=(v["clase_ternaria"].to_numpy()=="BAJA+2")
        S[(nom,va)]=([c.entrenar_o_cargar(P,X,y,w,250,s,CAR,
            f"{nom}_{va}_{s}_{c.clave(P,tr,'pesos',0.25,f'{nom}{len(pred)}',250,s)}").predict(v[pred])
            for s in sem], es)
def cmp(va, f, etq):
    a=np.array([f(s,S[("todo",va)][1]) for s in S[("todo",va)][0]])
    b=np.array([f(s,S[("base",va)][1]) for s in S[("base",va)][0]])
    p=wilcoxon(a,b).pvalue if np.any(a!=b) else 1.0
    print(f"    {va}  {etq:28s} {a.mean()-b.mean():+8.2f}  {int((a>b).sum()):2d}/10  p={p:.3f}")
gan=lambda K: (lambda s,es: c.ganancia_acumulada(s,es)[K-1]/1e6)
cap=lambda K: (lambda s,es: 100*es[np.argsort(-s)[:K]].sum()/es.sum())
pico=lambda s,es: c.ganancia_acumulada(s,es).max()/1e6
print("A) corte FIJO 10.000 en los dos folds  (como veniamos midiendo)")
for _,va in FOLDS: cmp(va, gan(10000), "ganancia M")
print("\nB) cada fold en SU PROPIO optimo  (K* 8.092 en mayo, 14.051 en junio)")
cmp(202105, gan(8092), "ganancia M @K*"); cmp(202106, gan(14051), "ganancia M @K*")
print("\nC) el PICO de cada curva  (sin fijar corte)")
for _,va in FOLDS: cmp(va, pico, "ganancia M en el pico")
print("\nD) CAPTURA a 10.000  (adimensional, no depende de la prevalencia)")
for _,va in FOLDS: cmp(va, cap(10000), "puntos de captura")
