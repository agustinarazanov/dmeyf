import sys, time, numpy as np, pandas as pd
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
import competencia as c
t0=time.time()
d=c.cargar("competencia_01.parquet")
pred=c.columnas_predictoras(d)
MESES=[202103,202104,202105,202106]
def auc_rapido(x, y):
    """AUC via Mann-Whitney sobre rangos. Los NULL van a un grupo propio (rango medio)."""
    s=pd.Series(x); r=s.rank(method="average", na_option="bottom").to_numpy()
    n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan
    return (r[y].sum() - n1*(n1+1)/2) / (n1*n0)
def psi(ref, act, bins=10):
    q=np.unique(np.nanquantile(ref, np.linspace(0,1,bins+1)))
    if len(q)<3: return np.nan
    a,_=np.histogram(act[~np.isnan(act)], bins=q); e,_=np.histogram(ref[~np.isnan(ref)], bins=q)
    a=a/max(a.sum(),1)+1e-10; e=e/max(e.sum(),1)+1e-10
    return float(((a-e)*np.log(a/e)).sum())
filas=[]
sub={m: d[d[c.fe.MES]==m] for m in MESES}
ys={m: (sub[m][c.fe.CLASE].to_numpy()=="BAJA+2") for m in MESES}
for i,v in enumerate(pred):
    if not np.issubdtype(sub[202103][v].dtype, np.number): continue
    a={m: auc_rapido(sub[m][v].to_numpy(dtype="float64"), ys[m]) for m in MESES}
    if any(np.isnan(x) for x in a.values()): continue
    ref=sub[202103][v].to_numpy(dtype="float64")
    p=psi(ref, sub[202106][v].to_numpy(dtype="float64"))
    vals=np.array([a[m] for m in MESES])
    filas.append({"var":v, **{f"auc_{str(m)[-2:]}": a[m] for m in MESES},
                  "rango_auc": vals.max()-vals.min(),
                  "deriva_mar_jun": a[202106]-a[202103],
                  "senal_max": np.abs(vals-0.5).max(), "psi_mar_jun": p})
    if (i+1)%40==0: print(f"  {i+1}/{len(pred)} [{time.time()-t0:.0f}s]", flush=True)
t=pd.DataFrame(filas)
t.to_parquet(c.EXPERIMENTOS/"c128_drift.parquet", index=False)
print(f"\n{len(t)} variables analizadas en {time.time()-t0:.0f}s")
print("\n=== CONCEPT DRIFT: PSI bajo (la distribucion no se mueve) pero el AUC si ===")
cand=t[(t.psi_mar_jun<0.1)&(t.senal_max>0.02)].nlargest(15,"rango_auc")
print(cand[["var","auc_03","auc_04","auc_05","auc_06","rango_auc","deriva_mar_jun","psi_mar_jun"]].round(4).to_string(index=False))
