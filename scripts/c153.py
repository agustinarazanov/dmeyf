"""El FE completo de la catedra (z402, 228 columnas) medido bien:
20 archivos de 1 semilla al corte 9.000 (el optimo que acabamos de establecer).
El test anterior estaba viciado: hiperparametros de 150 columnas y folds que no
pueden medir historia. La referencia al mismo corte es c152_corte9000 = 94.2232."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, competencia as c
t0=time.time(); K=9000; MESES=[202103,202104,202105,202106]
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
np.random.seed(c.SEMILLAS[0]); SEM=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
ref=c.cargar("competencia_01.parquet")
for nom,ds in [("c153_fe228","competencia_01_fe.parquet"),
               ("c153_histmin9k","competencia_01_histmin.parquet")]:
    CAR=c.EXPERIMENTOS/nom; (CAR/"envios").mkdir(parents=True,exist_ok=True)
    d=c.cargar(ds); pred=c.columnas_predictoras(d)
    X,y,w=c.preparar(d,MESES,"pesos",0.25); X=X[pred]
    fut=d[d[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); validos=set(ids.tolist())
    print(f"{nom}: {len(pred)} predictoras, {len(X):,} filas de train")
    for i,s in enumerate(SEM,1):
        sc=c.entrenar_o_cargar(P,X,y,w,250,s,CAR,
            f"{nom}_{s}_{c.clave(P,MESES,'pesos',0.25,nom+str(len(pred)),250,s)}").predict(fut[pred])
        sel=c.top_k(sc,ids,K)
        if i==1: c.verificar_seleccion(sel, ref)
        c.escribir_envios(sel, CAR/"envios"/f"{nom}_s{s}.csv", validos)
        print(f"  {i:2d}/20 [{time.time()-t0:.0f}s]",end="\r",flush=True)
    print(f"  listo, 20 archivos de {K:,}   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
