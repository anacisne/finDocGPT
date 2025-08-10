import os, re, pandas as pd
from dotenv import load_dotenv
load_dotenv()

ART = os.getenv("ART_DIR", "artifacts")
Q = pd.read_parquet(f"{ART}/questions.parquet")

# Example heuristic: keep questions whose text starts with KPI keywords
KPI_KEYS = ["revenue", "net income", "gross margin", "operating margin", "free cash flow", "debt"]
def is_kpi(q):
    ql = q.lower()
    return any(k in ql for k in KPI_KEYS)

kq = Q[Q["question"].apply(is_kpi)].copy()
# This file should be merged with predictions if present
try:
    P = pd.read_csv(f"{ART}/predictions.csv")[["financebench_id","pred_answer"]]
    kq = kq.merge(P, on="financebench_id", how="left")
except FileNotFoundError:
    kq["pred_answer"] = None

def to_num(s):
    if s is None: return None
    s = str(s)
    m = re.findall(r"[-+]?\d[\d,\.]*", s)
    if not m: return None
    return float(m[0].replace(",",""))

rows=[]
for r in kq.itertuples():
    rows.append({
        "company": r.company,
        "doc_name": r.doc_name,
        "question": r.question,
        "metric": next((k for k in KPI_KEYS if k in r.question.lower()), "other"),
        "value": to_num(r.pred_answer) if r.pred_answer else None
    })

df = pd.DataFrame(rows)
df.to_csv(f"{ART}/kpi_table.csv", index=False)
print("Saved:", f"{ART}/kpi_table.csv", "rows:", len(df))
