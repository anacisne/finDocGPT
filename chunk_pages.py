import os, re, pandas as pd

ART = os.getenv("ART_DIR", "artifacts")
PAGES = os.path.join(ART, "pages.parquet")
OUT = os.path.join(ART, "chunk_store.parquet")

assert os.path.exists(PAGES), f"Missing {PAGES}"
df = pd.read_parquet(PAGES)

rows = []
for _, r in df.iterrows():
    doc = r["doc_name"]
    page = int(r["page"])
    text = (r.get("page_text") or "").strip()

    # prefer paragraph splits; if page is long/monolithic, fall back to 900-char windows
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paras:
        for i in range(0, len(text), 900):
            paras.append(text[i:i+900])

    # enforce max 900 chars per chunk (split paragraphs that are too long)
    maxlen = 900
    for p in paras:
        if len(p) <= maxlen:
            rows.append({"doc_name":doc, "page":page, "chunk_id":len(rows)+1, "chunk_text":p})
        else:
            for i in range(0, len(p), maxlen):
                rows.append({"doc_name":doc, "page":page, "chunk_id":len(rows)+1, "chunk_text":p[i:i+maxlen]})

out_df = pd.DataFrame(rows)
os.makedirs(ART, exist_ok=True)
out_df.to_parquet(OUT, index=False)
print(f"✅ Wrote {OUT} with {len(out_df)} chunks from {len(df)} pages")
