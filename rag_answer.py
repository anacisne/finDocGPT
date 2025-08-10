# scripts/rag_answer.py
import os, json
from openai import OpenAI

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def call_llm(prompt: str) -> dict:
    """
    GPT-5 via Chat Completions. Returns:
      {"answer": str, "citations": [{"doc":..., "page":...}, ...]}
    """
    client = _get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5")

    # Keep system message short; JSON enforced by instruction + post-parse
    system_msg = (
        "You are a financial QA assistant. "
        "Return STRICT JSON with keys: answer (string) and citations (list of objects {doc, page}). "
        "Do not include any other text."
    )

    completion = client.chat.completions.create(
        model=model,
        # GPT-5 rejects non-default temps on some endpoints; omit or set 1
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
    )

    text = completion.choices[0].message.content
    # Parse JSON safely
    try:
        data = json.loads(text)
        if "answer" not in data:
            data["answer"] = ""
        if "citations" not in data or not isinstance(data["citations"], list):
            data["citations"] = []
        return data
    except Exception:
        # Fallback: wrap as JSON if model returned plain text
        return {"answer": text.strip(), "citations": []}
