import cv2
import pytesseract
import numpy as np
import re
import datetime as dt
from dateutil import parser as dparser
from typing import Tuple, List, Dict, Any, Optional

# ---------------------------
# CV preprocessing utilities
# ---------------------------
def _deskew(img_gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(img_gray > 0))
    if coords.shape[0] < 10:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def _preprocess_image(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    h, w = img.shape[:2]
    if max(w, h) > 1600:
        scale = 1600 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 2)
    gray = _deskew(gray)
    return gray

# ---------------------------
# Text parsing helpers
# ---------------------------
_money_rx = re.compile(r"([€$£]?)\s*([0-9]+(?:[.,][0-9]{1,2})?)")

# Updated item line patterns to capture qty, name, unit_price, line_total
_item_patterns = [
    re.compile(
        r"^\s*"
        r"(?:(?P<qty>\d+)\s+)?"
        r"(?P<name>.+?)\s+"
        r"(?P<unit_price>[0-9]+(?:[.,][0-9]{1,2})?)\s*\$"
        r"(?P<line_total>[0-9]+(?:[.,][0-9]{1,2})?)\s*$"
    )
]

_date_patterns = [
    re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})"),
    re.compile(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"),
    re.compile(r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{2,4})", re.IGNORECASE),
]

def _is_likely_item_line(line: str) -> bool:
    low = line.lower()
    # skip headers, addresses, or receipt info
    skip_keywords = (
        "receipt", "invoice", "tax", "subtotal", "balance",
        "billed to", "customer", "notes", "thank you", "food receipt"
    )
    if any(k in low for k in skip_keywords):
        return False
    # must contain at least one numeric price at end
    return bool(re.search(r"\d+\.\d{2}\s*$", line))

def _safe_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip().replace(",", ".")
    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s)
    except Exception:
        return None

# ---------------------------
# Main extraction function
# ---------------------------
def extract_receipt(img_bytes: bytes) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gray = _preprocess_image(img_bytes)
    ocr_text = pytesseract.image_to_string(gray)
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    if not lines:
        ocr_text = pytesseract.image_to_string(gray, config="--psm 6")
        lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]

    # Merchant
    merchant = next((ln for ln in lines[:5] if len(ln) > 2 and ln.lower() not in ("receipt","invoice","tax","total","subtotal","balance")), None)
    merchant = merchant or (lines[0] if lines else "Unknown")

    # Date
    parsed_date = None
    for ln in lines:
        for pat in _date_patterns:
            m = pat.search(ln)
            if m:
                try:
                    cand = dparser.parse(m.group(1), fuzzy=True)
                    if cand.date() <= dt.date.today():
                        parsed_date = cand.date()
                        break
                except Exception:
                    continue
        if parsed_date:
            break
    if not parsed_date:
        try:
            cand = dparser.parse(ocr_text, fuzzy=True, default=dt.datetime.now())
            if cand.date() <= dt.date.today():
                parsed_date = cand.date()
        except Exception:
            parsed_date = None

    # Total
    total_value = None
    currency = None
    for ln in reversed(lines[-12:]):
        if re.search(r"(total|amount due|amount|balance|grand total|amt due|payment)", ln, re.IGNORECASE):
            m = _money_rx.search(ln)
            if m:
                currency = m.group(1) or ""
                total_value = _safe_float(m.group(2))
                if total_value is not None:
                    break
    if total_value is None:
        candidate = None
        for ln in lines[-20:]:
            for m in _money_rx.finditer(ln):
                val = _safe_float(m.group(2))
                if val is not None and (candidate is None or val > candidate[0]):
                    candidate = (val, m.group(1) or "", ln)
        if candidate:
            total_value, currency, _ = candidate

    # Items
    items = []
    for ln in lines:
        if not _is_likely_item_line(ln):
            continue  # skip headers/addresses

        matched = False
        for pat in _item_patterns:
            m = pat.match(ln)
            if not m:
                continue
            qty = float(m.group("qty") or 1)
            name = m.group("name").strip()
            unit_price = _safe_float(m.group("unit_price"))
            line_total = _safe_float(m.group("line_total")) or (unit_price * qty if unit_price else None)
            items.append({
                "name": re.sub(r"\s{2,}", " ", name),
                "qty": qty,
                "unit_price": unit_price,
                "line_total": line_total,
                "raw_line": ln,
                "price": line_total or 0.0
            })
            matched = True
            break
        if not matched:
            # fallback heuristic
            m = re.search(r"(.+?)\s+([0-9\.,]+\d)\s*$", ln)
            if m:
                name = m.group(1).strip()
                val = _safe_float(m.group(2)) or 0.0
                items.append({
                    "name": re.sub(r"\s{2,}", " ", name),
                    "qty": 1.0,
                    "unit_price": None,
                    "line_total": val,
                    "raw_line": ln,
                    "price": val
                })

    # Remove subtotal/tax/total lines
    items = [it for it in items if not any(k in it["name"].lower() for k in ("subtotal","tax","total","change","balance","visa","mastercard","amt"))]

    # Metadata
    meta = {
        "merchant": merchant,
        "date": parsed_date.isoformat() if parsed_date else None,
        "total": float(total_value) if total_value else 0.0,
        "currency": currency or "",
        "raw_text": ocr_text,
    }

    return meta, items

# ---------------------------
# Prepare documents for vector indexing
# ---------------------------
def make_documents_for_embedding(meta: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    receipt_id = (meta.get("date") or "unknown-date") + "-" + (meta.get("merchant") or "unknown-merchant")
    header_text = f"Merchant: {meta.get('merchant')}\nDate: {meta.get('date')}\nTotal: {meta.get('total')}\n"
    docs.append({
        "id": f"receipt::{receipt_id}::header",
        "text": header_text + "\n" + (meta.get("raw_text") or ""),
        "meta": {
            "merchant": meta.get("merchant"),
            "date": meta.get("date"),
            "total": meta.get("total"),
        }
    })
    for i, it in enumerate(items):
        price_val = it.get("line_total") or it.get("unit_price")
        text = f"{it.get('name')} — qty: {it.get('qty')}, unit_price: {it.get('unit_price')}, line_total: {it.get('line_total')}"
        docs.append({
            "id": f"receipt::{receipt_id}::item::{i}",
            "text": text,
            "meta": {
                "merchant": meta.get("merchant"),
                "date": meta.get("date"),
                "total": meta.get("total"),
                "name": it.get("name"),
                "price": float(price_val) if price_val is not None else 0.0,
                "quantity": it.get("qty"),
                "unit_price": it.get("unit_price"),
                "line_total": it.get("line_total")
            }
        })
    return docs
