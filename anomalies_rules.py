import pandas as pd, numpy as np, os
from dotenv import load_dotenv
load_dotenv()

ART = os.getenv("ART_DIR", "artifacts")
kpi = pd.read_csv(f"{ART}/kpi_table.csv")

def z_flag(s, w=4, thr=2.5):
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    z = (s - m) / sd
    return z, z.abs() >= thr

def iqr_flag(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    return (s < low) | (s > high)

out=[]
for (company, metric), g in kpi.groupby(["company","metric"]):
    g = g.sort_values("doc_name")  # naive ordering by doc_name string
    s = g["value"].astype(float)
    z, zf = z_flag(s)
    i = iqr_flag(s)
    for r, zval, zflg, iflg in zip(g.itertuples(), z, zf, i):
        out.append({
            "company": company,
            "metric": metric,
            "doc_name": r.doc_name,
            "value": r.value,
            "z": None if pd.isna(zval) else float(zval),
            "flag_z": bool(zflg) if not pd.isna(zval) else False,
            "flag_iqr": bool(iflg)
        })

df = pd.DataFrame(out)
df.to_csv(f"{ART}/anomaly_flags.csv", index=False)
print("Saved:", f"{ART}/anomaly_flags.csv", "rows:", len(df))
