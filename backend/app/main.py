from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
import os
import re
from datetime import datetime, timedelta, date
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from .receipt_ocr import extract_receipt
from .vector_store import VectorStore

app = FastAPI(title="AI Engineer Test Backend", version="0.9.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VSTORE_PATH = os.environ.get("VSTORE_PATH", "/data/vectors.jsonl")
vdb = VectorStore(VSTORE_PATH)


class UpsertReq(BaseModel):
    id: str
    text: str
    merchant: str | None = None
    date: str | None = None
    price: float | None = None


class QueryReq(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload-receipt")
async def upload_receipt(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    content = await file.read()
    meta, items = extract_receipt(content)

    # Ensure purchase_date is a string
    purchase_date = meta.get("date") or dt.datetime.utcnow().date().isoformat()
    merchant = meta.get("merchant") or "Unknown"

    stored_items = []

    for it in items:
        name = (it.get("name") or "").strip()
        if not name:
            continue

        item_id = f"purchase-{purchase_date}-{name.lower().replace(' ', '-')}"
        
        # Numeric fields defaults
        price = float(it.get("price") or 0)
        qty = float(it.get("qty") or 1)
        unit_price = float(it.get("unit_price") or 0) if it.get("unit_price") is not None else None
        line_total = float(it.get("line_total") or price)

        vdb.upsert(item_id, {
            "text": name,
            "merchant": merchant,
            "date": str(purchase_date),
            "price": price,
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": line_total
        })

        stored_items.append({
            "id": item_id,
            "name": name,
            "price": price,
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": line_total
        })

    return {
        "receipt_date": str(purchase_date),
        "merchant": merchant,
        "total": float(meta.get("total") or 0),
        "currency": meta.get("currency") or "",
        "items": stored_items
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

    sim_matrix = np.dot(a_norm, b_norm.T)
    return sim_matrix.flatten()


def parse_date(s: str):
    """Parse YYYY-MM-DD safely into Python date."""
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def extract_explicit_date(query: str):
    q = query.lower()

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").date()
        except Exception:
            pass

    m = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|"
                  r"september|october|november|december)(\s+\d{4})?\b", q)
    if m:
        day = int(m.group(1))
        month = m.group(2).title()
        year = int(m.group(3).strip()) if m.group(3) else datetime.utcnow().year
        try:
            return datetime.strptime(f"{day} {month} {year}", "%d %B %Y").date()
        except Exception:
            pass

    return None


def extract_date_range(query: str) -> Tuple[date, date] | None:
    q = query.lower()

    m = re.search(r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", q)
    if m:
        try:
            d1 = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            d2 = datetime.strptime(m.group(2), "%Y-%m-%d").date()
            return d1, d2
        except Exception:
            pass

    m = re.search(r"between\s+(\d{1,2}\s+\w+(?:\s+\d{4})?)\s+and\s+(\d{1,2}\s+\w+(?:\s+\d{4})?)", q)
    if m:
        try:
            d1 = datetime.strptime(m.group(1), "%d %B %Y" if re.search(r"\d{4}", m.group(1)) else "%d %B").date()
            d2 = datetime.strptime(m.group(2), "%d %B %Y" if re.search(r"\d{4}", m.group(2)) else "%d %B").date()
            if d1.year == 1900:
                d1 = d1.replace(year=datetime.utcnow().year)
            if d2.year == 1900:
                d2 = d2.replace(year=datetime.utcnow().year)
            return d1, d2
        except Exception:
            pass

    return None


def summarize_answer(query: str, matches: List[dict]) -> str:
    """Summarize purchases with full support for item, merchant, and total expense queries across all date filters."""
    if not matches:
        return "I couldn’t find any matching items."

    q_lower = query.lower()
    today = datetime.utcnow().date()

    def safe_price(val):
        try:
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    # --- Detect item name from query ---
    item_name = None
    item_match = re.search(r"where did i buy (.+?)(?: from| on|$)", q_lower)
    if item_match:
        item_name = item_match.group(1).strip().lower()

    # --- Helper: filter by date ---
    def filter_matches(ms, date_check=None):
        filtered = []
        for m in ms:
            date_val = parse_date(m["metadata"].get("date", ""))
            if date_check and not date_check(date_val):
                continue
            filtered.append(m)
        return filtered

    # --- Cosine similarity for merchant lookup ---
    def best_merchant_by_similarity(item_name, ms):
        candidates = [m for m in ms if m["metadata"].get("merchant")]
        if not candidates:
            return None, None

        texts = [m["metadata"].get("text", "").lower() for m in candidates]
        merchants = [m["metadata"].get("merchant") for m in candidates]

        vectorizer = TfidfVectorizer().fit(texts + [item_name])
        vectors = vectorizer.transform(texts + [item_name])
        query_vec = vectors[-1].toarray()
        item_vecs = vectors[:-1].toarray()

        similarities = cosine_similarity(query_vec, item_vecs).flatten()
        best_idx = similarities.argmax()
        if similarities[best_idx] < 0.2:  # low confidence threshold
            return None, None

        return texts[best_idx], merchants[best_idx]

    # --- Determine date filter ---
    start, end = None, None
    explicit_date = extract_explicit_date(query)
    date_range = extract_date_range(query)
    if date_range:
        start, end = date_range
    elif explicit_date:
        start = end = explicit_date
    elif "yesterday" in q_lower:
        start = end = today - timedelta(days=1)
    elif "last 7 day" in q_lower or "last 7 days" in q_lower:
        start = today - timedelta(days=7)
        end = today

    # --- Apply date filter ---
    if start and end:
        filtered = filter_matches(matches, lambda d: isinstance(d, date) and start <= d <= end)
    else:
        filtered = matches

    # --- Merchant / item lookup ---
    if item_name:
        matched_text, merchant = best_merchant_by_similarity(item_name, filtered)
        if "where" in q_lower:
            if merchant:
                return f"You bought '{matched_text}' from: {merchant}"
            else:
                return f"No purchases of '{item_name}' found in the selected date range."
        else:
            # List items matching the name in date range
            item_matches = [m["metadata"]["text"] for m in filtered if item_name in m["metadata"].get("text", "").lower()]
            if item_matches:
                return f"You bought {', '.join(item_matches)} in the selected date range."
            else:
                return f"No purchases of '{item_name}' found in the selected date range."

    # --- Total expense queries ---
    if "total" in q_lower and "expense" in q_lower:
        total = sum(safe_price(m["metadata"].get("price")) for m in filtered)
        return f"Your total expenses were {total:.2f}."

    # --- Fallback: list items in date range ---
    if filtered:
        return "You bought: " + ", ".join([m["metadata"]["text"] for m in filtered])
    return "No purchases found in the selected date range."




@app.get("/qa")
def qa(q: str = Query(..., description="Natural language question")):
    try:
        matches = vdb.query(q, k=10)
        summary = summarize_answer(q, matches)
        # return {
        #     "query": q,
        #     "answer": summary,
        #     "matches": matches,
        # }
        return summary
    except Exception as e:
        return {"query": q, "answer": f"Vector search failed: {str(e)}"}


@app.post("/vectors/upsert")
def vectors_upsert(req: UpsertReq):
    metadata = {
        "text": req.text,
        "merchant": req.merchant,
        "date": req.date,
        "price": float(req.price) if req.price is not None else None,
    }
    vdb.upsert(req.id, metadata)
    return {"ok": True, "count": vdb.count()}


@app.post("/vectors/query")
def vectors_query(req: QueryReq):
    matches = vdb.query(req.query, k=req.k)
    summary = summarize_answer(req.query, matches)
    return {"answer": summary, "matches": matches}


@app.post("/admin/reset-db")
def reset_db_endpoint():
    vdb.reset()
    return {"ok": True, "message": "Vector store reset"}
