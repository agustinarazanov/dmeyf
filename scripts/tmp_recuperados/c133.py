import sys, time
from pathlib import Path
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, pandas as pd, fe_panel as fe, competencia as c
t0=time.time(); raiz=Path("/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
PRES=["mprestamos_personales","cprestamos_personales"]
con=fe.conectar()
con.execute(f"""create or replace view hist as select * from read_parquet('{raiz/"data/competencia_01_hist.parquet"}')""")
cols=[r[0] for r in con.sql("describe hist").fetchall()]
montos=[x for x in cols if x.startswith("m") and "__" not in x]
# deflactar los montos CRUDOS por la mediana de su mes; las features historicas ya son relativas
dfl=", ".join(f'median(case when {m}>0 then {m} end) over (partition by foto_mes) as "{m}__d"' for m in montos)
con.execute(f"create or replace table med as select *, {dfl} from hist")
rat=", ".join(f'ratio_seguro({m}, "{m}__d") as "{m}__defl"' for m in montos)
quitar=[f'"{m}__d"' for m in montos] + montos + [x for x in PRES if x in cols and x not in montos]
con.execute(f"create or replace table todo as select * exclude ({', '.join(quitar)}), {rat} from med")
# y fuera tambien la version deflactada del prestamo
con.execute('alter table todo drop column "mprestamos_personales__defl"')
n=con.sql("select count(*) from (describe todo)").fetchone()[0]
con.execute(f"copy todo to '{raiz/'data/competencia_01_todo.parquet'}' (format parquet, compression zstd)")
con.close()
print(f"dataset combinado: {n} columnas")
d=c.cargar("competencia_01_todo.parquet"); pred=c.columnas_predictoras(d)
assert not [x for x in pred if "prestamos_personales" in x], "quedo un prestamo personal"
print(f"{len(pred)} predictoras, prestamos personales: 0")
MESES=[202103,202104,202105,202106]; NBR,CORTE=250,10_000
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c133_todo"; (CAR/"envios").mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0]); sem=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
X,y,w=c.preparar(d,MESES,"pesos",0.25); X=X[pred]
fut=d[d[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); val=set(ids.tolist())
sc={}
for s in sem:
    cl=c.clave(P,MESES,"pesos",0.25,f"todo{len(pred)}",NBR,s)
    sc[s]=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"todo_{s}_{cl}").predict(fut[pred])
    print(f"  {len(sc)}/20 [{time.time()-t0:.0f}s]",end="\r",flush=True)
ens=c.ensamble_por_rank(sc); sel=c.top_k(ens,ids,CORTE)
c.verificar_seleccion(sel, c.cargar("competencia_01.parquet"))
nn=c.escribir_envios(sel,CAR/"envios"/"c133_todo.csv",val)
v=pd.read_parquet(c.EXPERIMENTOS/"c107_pesos25"/"scores_202108.parquet")
ref=set(c.top_k(v.ensamble.to_numpy(), v.numero_de_cliente.to_numpy().astype("int64"), CORTE))
print(f"\nc133_todo.csv: {nn:,} envios, solapamiento con la entrega actual {len(set(sel)&ref)/CORTE:.1%}")
print(f"listo en {time.time()-t0:.0f}s")
