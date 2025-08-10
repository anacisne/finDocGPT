import os, json, time, sys, ast, re
import numpy as np
import pandas as pd
import faiss

# deps
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# paths / params
ART = os.getenv("ART_DIR", "artifacts")
QPARQ = os.path.join(ART, "questions.parquet")
CHUNKS = os.path.join(ART, "chunk_store.parquet")
FAISS_INDEX = os.path.join(ART, "faiss.index")
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "6"))
OUT_CSV = os.path.join(ART, "predictions.csv")
PROMPTS_YAML = "prompts.yaml"

# model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def load_prompts():
    # very small YAML reader without external deps
    if not os.path.exists(PROMPTS_YAML):
        # sensible defaults
        return {
            "system": "Answer only from CONTEXT. If numeric, return a single number with unit. If insufficient, return \"insufficient evidence\". Always include citations as {\"doc\":\"<DOC_NAME>\",\"page\":<PAGE>}.",
            "user_template": "QUESTION:\n{question}\n\nCONTEXT:\n{contexts}\n\nOUTPUT JSON SCHEMA:\n{\n  \"answer\": \"string\",\n  \"citations\": [{\"doc\": \"string\", \"page\": 0}]\n}"
        }
    import io
    # parse minimal yaml
    data = {}
    with open(PROMPTS_YAML, "r", encoding="utf-8") as f:
        content = f.read()
    # naive parse
    sysm = re.search(r"system:\s*\|\s*(.+?)\n\s*\n", content, re.S)
    userm = re.search(r"user_template:\s*\|\s*(.+)$", content, re.S)
    data["system"] = sysm.group(1).strip() if sysm else ""
    data["user_template"] = userm.group(1).strip() if userm else ""
    return data

def ensure_files():
    for p in [QPARQ, CHUNKS, FAISS_INDEX]:
        if not os.path.exists(p):
            sys.exit(f"Missing required file: {p}")

def load_index():
    index = faiss.read_index(FAISS_INDEX)
    return index

def embedder():
    return SentenceTransformer(EMB_MODEL_NAME)

def norm_text(x):
    return (x or "").strip()

def make_context(chunk_df, idxs):
    rows = chunk_df.iloc[idxs][["doc_name","page","chunk_text"]].to_dict("records")
    # context = bullet list of doc/page + text
    ctx_lines = []
    for r in rows:
        ctx_lines.append(f"[{r['doc_name']} p.{r['page']}] {r['chunk_text']}")
    return "\n".join(ctx_lines), rows

def ask_llm(system_prompt, user_prompt):
    # Use Chat Completions; some models only accept default temperature
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ]
    )
    content = resp.choices[0].message.content
    return content

def force_json(s):
    # try json, then python literal, then last-resort brace extract
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            try:
                return ast.literal_eval(m.group(0))
            except Exception:
                return {"answer": s.strip(), "citations":[]}
    return {"answer": s.strip(), "citations":[]}

def main():
    ensure_files()
    prompts = load_prompts()

    # load data
    questions = pd.read_parquet(QPARQ)[["financebench_id","question"]]
    chunks = pd.read_parquet(CHUNKS)
    # sanity
    if "emb_model_name" in chunks.columns:
        # optional metadata; ignore
        pass

    # load index + embedder
    index = load_index()
    model = embedder()

    preds = []
    for i, row in questions.iterrows():
        qid = row["financebench_id"]
        q = norm_text(row["question"])
        if not q:
            continue

        # embed question
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        D, I = index.search(q_emb.astype(np.float32), TOP_K)
        idxs = I[0].tolist()

        # build context
        context_str, ctx_rows = make_context(chunks, idxs)

        # prompt
        user_prompt = prompts["user_template"].format(question=q, contexts=context_str)
        system_prompt = prompts["system"]

        # ask LLM (retry basic)
        for attempt in range(3):
            try:
                out = ask_llm(system_prompt, user_prompt)
                break
            except Exception as e:
                time.sleep(1 + attempt)
        else:
            out = "{\"answer\": \"insufficient evidence\", \"citations\": []}"

        obj = force_json(out)
        ans = obj.get("answer") if isinstance(obj, dict) else str(obj)
        cits = obj.get("citations") if isinstance(obj, dict) else []

        preds.append({
            "financebench_id": qid,
            "question": q,
            "pred_answer": ans,
            "citations": json.dumps(cits, ensure_ascii=False),
            "raw": out
        })

        # simple progress
        if (i+1) % 10 == 0:
            print(f"... answered {i+1}/{len(questions)}")

    df = pd.DataFrame(preds)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV} with {len(df)} rows")

if __name__ == "__main__":
    main()
