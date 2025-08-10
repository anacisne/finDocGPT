# scripts/parse_pdfs.py
import os
import pandas as pd
from pypdf import PdfReader
from pypdf.errors import PdfReadError

ART = os.environ.get("ART_DIR", "artifacts")
PDF_DIR = os.environ.get("PDF_DIR", "financebench-main/pdfs")
os.makedirs(ART, exist_ok=True)

pdf_paths = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
if not pdf_paths:
    raise SystemExit(f"No PDFs found in {PDF_DIR}/ — add files before running.")

rows = []
for path in pdf_paths:
    doc_name = os.path.splitext(os.path.basename(path))[0]
    try:
        reader = PdfReader(path)
        # Try decryption if needed
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                print(f"⚠️  Skipping encrypted PDF {doc_name}: {e}")
                continue
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"⚠️  Could not extract text from {doc_name} page {i}: {e}")
                text = ""
            rows.append({"doc_name": doc_name, "page": i, "page_text": text})
    except PdfReadError as e:
        print(f"⚠️  Could not read PDF {doc_name}: {e}")
    except Exception as e:
        print(f"⚠️  Unexpected error with {doc_name}: {e}")

df = pd.DataFrame(rows)
outp = os.path.join(ART, "pages.parquet")
df.to_parquet(outp, index=False)
print(f"✅ Saved {outp} with {len(df)} rows from {len(pdf_paths)} PDFs")
