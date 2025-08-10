import re, math, pandas as pd
from sklearn.metrics import f1_score

Q = "artifacts/questions.parquet"
P = "artifacts/predictions.csv"
OUT = "artifacts/qa_report.csv"

def _to_num(s):
    s = str(s)
    m = re.findall(r"[-+]?\d[\d,\.]*", s)
    if not m: return math.nan
    x = m[0].replace(",", "")
    try: return float(x)
    except: return math.nan

def numeric_close(pred, gold, pct=0.02, abs_tol=1.0):
    p, g = _to_num(pred), _to_num(gold)
    if math.isnan(p) or math.isnan(g): return False
    band = max(abs_tol, pct*abs(g))
    return abs(p-g) <= band

def token_f1(a, b):
    A = str(a).lower().split()
    B = str(b).lower().split()
    if not A and not B: return 1.0
    inter = len(set(A) & set(B))
    if inter == 0: return 0.0
    prec = inter/len(set(A))
    rec  = inter/len(set(B)) if len(set(B)) else 0
    if prec+rec==0: return 0.0
    return 2*prec*rec/(prec+rec)

def main():
    qs = pd.read_parquet(Q)[["financebench_id","answer","question"]]
    pr = pd.read_csv(P)
    df = qs.merge(pr, on="financebench_id", how="left")

    rows=[]
    for r in df.itertuples():
        is_num = any(c.isdigit() for c in str(r.answer))
        if is_num:
            correct = numeric_close(r.pred_answer, r.answer, pct=0.02, abs_tol=1.0)
            f1 = None
            mode = "numeric"
        else:
            f1 = token_f1(r.pred_answer, r.answer)
            correct = f1 == 1.0
            mode = "string"
        rows.append({"financebench_id": r.financebench_id, "mode": mode, "correct": int(correct), "f1": f1})

    rep = pd.DataFrame(rows)
    agg = rep.groupby("mode").agg(acc=("correct","mean"), f1=("f1","mean")).reset_index()
    rep.to_csv(OUT, index=False)
    agg.to_csv("artifacts/qa_summary.csv", index=False)
    print("Saved", OUT, "and artifacts/qa_summary.csv")
    print(agg)

if __name__ == "__main__":
    main()
