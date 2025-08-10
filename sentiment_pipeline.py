import os, pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
load_dotenv()

ART = os.getenv("ART_DIR", "artifacts")
MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")

pages = pd.read_parquet(f"{ART}/pages.parquet")
docs = pages.groupby("doc_name")["page_text"].apply(lambda s: " ".join(s.tolist())).reset_index()

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()

def batch_scores(texts, batch=16, max_len=256):
    out=[]
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=-1).numpy()  # [neg, neu, pos]
        out.extend(probs)
    return np.array(out)

probs = batch_scores(docs["page_text"].tolist(), batch=8)
docs["neg"], docs["neu"], docs["pos"] = probs[:,0], probs[:,1], probs[:,2]
docs.to_parquet(f"{ART}/sentiment.parquet", index=False)
print("Saved:", f"{ART}/sentiment.parquet", "rows:", len(docs))
