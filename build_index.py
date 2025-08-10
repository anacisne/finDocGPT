import os, numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer

ART = os.getenv("ART_DIR", "artifacts")
CHUNKS = os.path.join(ART, "chunk_store.parquet")
EMB = os.path.join(ART, "embeddings.npy")
IDX = os.path.join(ART, "faiss.index")
MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

assert os.path.exists(CHUNKS), f"Missing {CHUNKS}"
df = pd.read_parquet(CHUNKS)
texts = df["chunk_text"].astype(str).tolist()

model = SentenceTransformer(MODEL)
X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
np.save(EMB, X)

index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
faiss.write_index(index, IDX)
print(f"✅ Built {EMB} {X.shape} and {IDX} (ntotal={index.ntotal})")
