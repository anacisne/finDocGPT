import os, glob, traceback
import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pdfminer.high_level import extract_text as pdfminer_extract
from concurrent.futures import ProcessPoolExecutor, as_completed

ART = os.getenv("ART_DIR", "artifacts")
PDF_DIR = os.getenv("PDF_DIR", "financebench-main/pdfs")
LIMIT = int(os.getenv("PDF_LIMIT", "0"))
WORKERS = int(os.getenv("WORKERS", "1"))  # start safe

def parse_one(path: str):
    doc_name = os.path.splitext(os.path.basename(path))[0]
    rows = []
    try:
        reader = PdfReader(path)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                try:
                    pm = pdfminer_extract(path) or ""
                    if pm.strip():
                        rows.append({"doc_name": doc_name, "page": 0, "page_text": pm})
                except Exception:
                    pass
                return rows
        for i, page in enumerate(reader.pages, start=1):
            text = ""
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not text.strip():
                try:
                    pm = pdfminer_extract(path) or ""
                    if pm.strip():
                        rows.append({"doc_name": doc_name, "page": 0, "page_text": pm})
                        return rows
                except Exception:
                    pass
            rows.append({"doc_name": doc_name, "page": i, "page_text": text})
    except PdfReadError:
        try:
            pm = pdfminer_extract(path) or ""
            if pm.strip():
                rows.append({"doc_name": doc_name, "page": 0, "page_text": pm})
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
    return rows

def main():
    os.makedirs(ART, exist_ok=True)
    pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {PDF_DIR}/")
    if LIMIT > 0:
        pdf_paths = pdf_paths[:LIMIT]

    all_rows = []

    if WORKERS <= 1:
        for p in tqdm(pdf_paths, desc="Parsing PDFs", unit="pdf"):
            rows = parse_one(p) or []
            all_rows.extend(rows)
    else:
        workers = max(1, min(WORKERS, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(parse_one, p): p for p in pdf_paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs", unit="pdf"):
                rows = fut.result() or []
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=["doc_name","page","page_text"]).fillna({"page_text":""})
    outp = os.path.join(ART, "pages.parquet")
    df.to_parquet(outp, index=False)
    print(f"✅ Saved {outp} with {len(df)} rows from {len(pdf_paths)} PDFs (workers={WORKERS})")

if __name__ == "__main__":
    main()
