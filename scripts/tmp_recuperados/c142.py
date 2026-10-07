"""Cosechar c122: tomar lo ESTRUCTURAL de los mejores trials y pisarle el peso a 0,25.
La busqueda limpia exploro bien el espacio de hiperparametros pero con un objetivo
torcido (CV dentro del mes) que la empuja a peso alto. Los folds out of time dicen 0,25.
Revalidacion OOT en los dos folds, contra el base que estamos usando hoy."""
import sys, time, warnings
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
warnings.filterwarnings("ignore")
import numpy as np, optuna, registro, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
s=optuna.load_study(study_name="c122_base", storage=registro.dsn_url())
done=sorted([t for t in s.trials if t.state.name=="COMPLETE"], key=lambda t:-t.value)[:5]
BASE={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
ESTRUCT=["learning_rate","num_leaves","min_data_in_leaf","feature_fraction",
         "bagging_fraction","lambda_l1","lambda_l2","min_gain_to_split","max_bin"]
cfgs={"base_pesos25":BASE}
for i,t in enumerate(done):
    p={**BASE,**{k:v for k,v in t.params.items() if k in ESTRUCT},"bagging_freq":1}
    cfgs[f"t{t.number}_peso25"]=p
    print(f"  t{t.number}: cv={t.value:,.0f}  peso_original={t.params.get('peso_baja1'):.2f} -> 0.25  "
          f"leaves={t.params.get('num_leaves')} mdl={t.params.get('min_data_in_leaf')} ff={t.params.get('feature_fraction'):.2f}")
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
CAR=c.EXPERIMENTOS/"c142_cosecha"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
FOLDS=[([202103],202105),([202103,202104],202106)]
res={n:{} for n in cfgs}
print()
for nom,P in cfgs.items():
    for tr,va in FOLDS:
        X,y,w=c.preparar(d,tr,"pesos",0.25); X=X[pred]
        v=d[d[c.fe.MES]==va]; es=(v["clase_ternaria"].to_numpy()=="BAJA+2")
        g=[]
        for s_ in sem:
            m=c.entrenar_o_cargar(P,X,y,w,250,s_,CAR,f"{nom}_{va}_{s_}_{c.clave(P,tr,'pesos',0.25,'base150',250,s_)}")
            g.append(c.ganancia_acumulada(m.predict(v[pred]),es)[9999]/1e6)
        res[nom][va]=np.array(g)
    print(f"  {nom:16s} 202105 {res[nom][202105].mean():6.1f}M | 202106 {res[nom][202106].mean():6.1f}M | "
          f"media {np.mean([res[nom][202105].mean(),res[nom][202106].mean()]):6.1f}M   [{time.time()-t0:.0f}s]")
print("\n=== contra base_pesos25, pareado por (fold, semilla), @10.000 ===")
ref=np.concatenate([res["base_pesos25"][202105],res["base_pesos25"][202106]])
for nom in cfgs:
    if nom=="base_pesos25": continue
    a=np.concatenate([res[nom][202105],res[nom][202106]])
    print(f"  {nom:16s} {a.mean()-ref.mean():+6.1f}M  {int((a>ref).sum()):2d}/20  p={wilcoxon(a,ref).pvalue:.4f}  "
          f"(mayo {res[nom][202105].mean()-res['base_pesos25'][202105].mean():+6.1f} | "
          f"junio {res[nom][202106].mean()-res['base_pesos25'][202106].mean():+6.1f})")
print(f"\nlisto en {time.time()-t0:.0f}s")
