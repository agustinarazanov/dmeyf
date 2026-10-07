"""Bateria de 4, cada una 20 archivos de UNA semilla al corte 11.000.
Un submit cuesta 1 sin importar cuantos archivos lleve, y el bot devuelve la MEDIA
sobre los archivos: el ruido de semilla se divide por raiz(20). Referencia ya medida
sobre la misma particion publica: c107_pesos25 = 91.355, sd 1.4873 (error de la media 0.33)."""
import sys, time, warnings
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
warnings.filterwarnings("ignore")
import numpy as np, optuna, registro, competencia as c
t0=time.time(); K=11_000; MESES=[202103,202104,202105,202106]
BASE={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
st=optuna.load_study(study_name="c122_base", storage=registro.dsn_url())
t22=[t for t in st.trials if t.number==22][0]
EST=["learning_rate","num_leaves","min_data_in_leaf","feature_fraction","bagging_fraction",
     "lambda_l1","lambda_l2","min_gain_to_split","max_bin"]
P22={**BASE,**{k:v for k,v in t22.params.items() if k in EST},"bagging_freq":1}
np.random.seed(c.SEMILLAS[0]); SEM=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
VAR={  # nombre: (dataset, params, meses, por_mes)
 "c150_histmin": ("competencia_01_histmin.parquet", BASE, MESES, False),
 "c150_t22":     ("competencia_01.parquet",         P22,  MESES, False),
 "c150_pormes":  ("competencia_01.parquet",         BASE, MESES, True),
 "c150_sinjunio":("competencia_01.parquet",         BASE, [202103,202104,202105], False),
}
ref=c.cargar("competencia_01.parquet")
for nom,(ds,P,TR,por_mes) in VAR.items():
    CAR=c.EXPERIMENTOS/nom; (CAR/"envios").mkdir(parents=True,exist_ok=True)
    d=c.cargar(ds); pred=c.columnas_predictoras(d)
    fut=d[d[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); validos=set(ids.tolist())
    for i,s in enumerate(SEM,1):
        if por_mes:
            sc=c.ensamble_por_rank({m: c.entrenar_o_cargar(
                *( (P,)+c.preparar(d,[m],"pesos",0.25)[:1]+c.preparar(d,[m],"pesos",0.25)[1:] ),
                250,s,CAR,f"m{m}_{s}") .predict(fut[pred]) for m in TR}) if False else None
            sc={}
            for m in TR:
                X,y,w=c.preparar(d,[m],"pesos",0.25)
                sc[m]=c.entrenar_o_cargar(P,X[pred],y,w,250,s,CAR,
                    f"m{m}_{s}_{c.clave(P,[m],'pesos',0.25,'base150',250,s)}").predict(fut[pred])
            score=c.ensamble_por_rank(sc)
        else:
            X,y,w=c.preparar(d,TR,"pesos",0.25)
            score=c.entrenar_o_cargar(P,X[pred],y,w,250,s,CAR,
                f"{nom}_{s}_{c.clave(P,TR,'pesos',0.25,nom+str(len(pred)),250,s)}").predict(fut[pred])
        sel=c.top_k(score,ids,K)
        if i==1: c.verificar_seleccion(sel, ref)
        c.escribir_envios(sel, CAR/"envios"/f"{nom}_s{s}.csv", validos)
        print(f"  {nom} {i:2d}/20 [{time.time()-t0:.0f}s]",end="\r",flush=True)
    print(f"  {nom:15s} 20 archivos de {K:,} envios listos   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
