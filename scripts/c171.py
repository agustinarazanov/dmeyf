"""TAREA 3: ensamblar base + FE en vez de elegir uno.
Si dos modelos son indistinguibles (y lo son: z=-1,02), promediar rankings es un hedge
gratis que rinde al menos el promedio de los dos y suele rendir mas, porque los errores
no correlacionados se cancelan. Se mide en los TRES folds limpios, en captura."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time(); K=9000
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c171_mezcla"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
DS={"base":"competencia_01.parquet","fe228":"competencia_01_fe.parquet"}
FOLDS=[([202103],202105,"BAJA+2"),([202103,202104],202106,"BAJA+2"),
       ([202103,202104,202105],202107,"BAJA+1")]
for tr,va,tg in FOLDS:
    sc={}; es=None
    for nom,ds in DS.items():
        d=c.cargar(ds); pred=c.columnas_predictoras(d)
        X,y,w=c.preparar(d,tr,"pesos",0.25)
        v=d[d[c.fe.MES]==va]
        if es is None:
            ids_ref=v[c.fe.ID].to_numpy(); es=(v["clase_ternaria"].to_numpy()==tg)
        # alinear por id: los parquets no comparten orden de filas
        orden=v[c.fe.ID].to_numpy().argsort(); inv=np.empty_like(orden); inv[orden]=np.arange(len(orden))
        ref_sort=ids_ref.argsort()
        sc[nom]={s: None for s in sem}
        for s in sem:
            pr=c.entrenar_o_cargar(P,X[pred],y,w,250,s,CAR,
                f"{nom}_{va}_{s}_{c.clave(P,tr,'pesos',0.25,nom+str(len(pred)),250,s)}").predict(v[pred])
            # reordenar al orden de ids_ref
            tmp=np.empty_like(pr); tmp[ref_sort]=pr[orden]
            sc[nom][s]=tmp
    n=es.sum()
    cap=lambda v_: 100*es[np.argsort(-v_)[:K]].sum()/n
    out={}
    for nom in DS: out[nom]=np.array([cap(sc[nom][s]) for s in sem])
    out["mezcla"]=np.array([cap(c.ensamble_por_rank({0:sc["base"][s],1:sc["fe228"][s]})) for s in sem])
    mejor=max(("base","fe228"), key=lambda k: out[k].mean())
    a,b=out["mezcla"],out[mejor]
    print(f"  fold {va} ({tg}): base {out['base'].mean():5.2f}% | fe228 {out['fe228'].mean():5.2f}% | "
          f"mezcla {out['mezcla'].mean():5.2f}%  ->  {a.mean()-b.mean():+5.2f} pts contra el mejor de los dos "
          f"({mejor}), {int((a>b).sum())}/10 p={wilcoxon(a,b).pvalue:.3f}   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
