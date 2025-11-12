#!/usr/bin/env python3
"""
llm_resume_parser.py

LLM-first resume parsing:
- Extract selectable text (PyMuPDF) else OCR (pdf2image + pytesseract)
- Chunk text safely and send to OpenRouter/OpenAI-compatible LLM
- JSON-only prompt, robust backoff + Retry-After handling
- Validate JSON, fall back to regex for deterministic fields
- Save audit files and output Excel/JSON

Usage:
    python llm_resume_parser.py --single path/to/resume.pdf
    python llm_resume_parser.py --batch resumes_folder
"""

import os
import re
import sys
import time
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

import fitz  # pymupdf
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import requests
import pandas as pd
from dotenv import load_dotenv
import phonenumbers
from dateutil import parser as dateutil_parser

# ---------- Config ----------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:free")

# Tesseract config
os.environ["TESSDATA_PREFIX"] = os.getenv("TESSDATA_PREFIX", "/usr/share/tessdata/")

# Operational config
OCR_DPI = 300
MAX_PROMPT_CHARS = 28000   # conservative chunk size to avoid token overflow
MAX_RETRIES = 4
BASE_BACKOFF = 5
JITTER = 2

# Output
AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(exist_ok=True)

# Fields/schema
FIELDS = [
    "Name of Candidate",
    "Birth Date",
    "Marital Status",
    "Permanent Address",
    "Contact Number",
    "Email ID",
    "Education",
    "Total Years of Experience",
    "Experience Details",
    "Current Monthly Salary",
    "District",
    "Gender",
    "Present Address",
    "PAN Card",
    "Aadhar Card",
    "State",
    "Preferred Job Location",
]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("llm_resume_parser")

# ---------- Helpers: extraction ----------
def extract_text_selectable(pdf_path: Path) -> str:
    """Try to extract text via PyMuPDF (fast & accurate when present)."""
    try:
        doc = fitz.open(str(pdf_path))
        parts = []
        for page in doc:
            txt = page.get_text("text")
            if txt and txt.strip():
                parts.append(txt)
        doc.close()
        return "\n".join(parts).strip()
    except Exception as e:
        logger.debug("Selectable text extraction failed: %s", e)
        return ""

def extract_text_ocr(pdf_path: Path, dpi: int = OCR_DPI) -> str:
    """Fallback: convert PDF->images then OCR via pytesseract."""
    logger.info("Using OCR for %s (dpi=%d)", pdf_path, dpi)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    texts = []
    for i, page in enumerate(pages):
        # simple preprocessing: convert to RGB then OCR
        img = page.convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 6")
        texts.append(text)
    return "\n".join(texts).strip()

# ---------- Helpers: chunking ----------
def chunk_text(text: str, limit: int = MAX_PROMPT_CHARS) -> List[str]:
    """Split long text into chunks <= limit, trying to split at blank lines."""
    if len(text) <= limit:
        return [text]
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= limit:
            current += (p + "\n\n")
        else:
            if current:
                chunks.append(current.strip())
            # if paragraph itself is longer than limit, hard split
            if len(p) > limit:
                for i in range(0, len(p), limit):
                    chunks.append(p[i:i+limit])
                current = ""
            else:
                current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

# ---------- Helpers: regex fallback & normalization ----------
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}", re.I)
RE_IN_PHONE = re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}\b")
RE_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
RE_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
RE_DATE = re.compile(r"\b(?:\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}|\w{3,9}\s\d{1,2},\s?\d{4})\b")

def fallback_regex_fields(text: str) -> Dict[str, str]:
    out = {f: "" for f in FIELDS}
    email = RE_EMAIL.search(text)
    if email:
        out["Email ID"] = email.group(0)
    phone = RE_IN_PHONE.search(text)
    if phone:
        out["Contact Number"] = normalize_phone(phone.group(0)) or phone.group(0)
    pan = RE_PAN.search(text)
    if pan:
        out["PAN Card"] = pan.group(0)
    aad = RE_AADHAAR.search(text)
    if aad:
        out["Aadhar Card"] = aad.group(0)
    # DOB heuristic
    for m in RE_DATE.finditer(text):
        try:
            dt = dateutil_parser.parse(m.group(0), dayfirst=True, fuzzy=True)
            year = dt.year
            if 1950 <= year <= 2005:
                out["Birth Date"] = dt.strftime("%d/%m/%Y")
                break
        except Exception:
            continue
    return out

def normalize_phone(text: str) -> Optional[str]:
    try:
        p = phonenumbers.parse(text, "IN")
        if phonenumbers.is_valid_number(p):
            return phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 10:
            return digits[-10:]
    return None

# ---------- Helpers: LLM call with robust backoff ----------
def call_llm(prompt: str, model: str = MODEL_NAME, max_retries: int = MAX_RETRIES) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set. Cannot call LLM.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise resume parser. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
    }

    attempt = 0
    while attempt <= max_retries:
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                # OpenRouter-like shape: choices[0].message.content
                try:
                    content = data["choices"][0]["message"]["content"]
                except Exception:
                    # fallback: maybe different structure
                    content = json.dumps(data)
                return content
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after)
                else:
                    wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, JITTER)
                logger.warning("Rate limited (429). Waiting %s seconds (attempt %d).", wait, attempt+1)
                time.sleep(wait)
            else:
                logger.error("LLM error %d: %s", resp.status_code, resp.text[:300])
                # backoff for 5xx
                if 500 <= resp.status_code < 600:
                    wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, JITTER)
                    time.sleep(wait)
                else:
                    # client error -> do not retry
                    break
        except requests.RequestException as e:
            wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, JITTER)
            logger.warning("Network error calling LLM: %s. Retrying in %s sec.", e, wait)
            time.sleep(wait)
        attempt += 1
    logger.error("Max retries exceeded for LLM call.")
    return None

# ---------- Helpers: prompt & JSON validation ----------
def build_prompt_for_chunk(text_chunk: str) -> str:
    # Construct a strict instruction for JSON-only response
    fields_list = "\n".join([f'- "{f}"' for f in FIELDS])
    prompt = f"""
Extract ONLY the following fields from the resume text and return a VALID JSON object with these exact keys (use empty string "" when not found). Do NOT include explanations or any extra text.

FIELDS:
{fields_list}

Resume Text:
{text_chunk}
"""
    return prompt.strip()

def extract_json_from_response(resp_text: str) -> Optional[Dict[str, Any]]:
    """Locate the first JSON object in the response and parse it."""
    if not resp_text:
        return None
    # find the first { ... } block
    m = re.search(r"\{[\s\S]*\}", resp_text)
    try:
        if m:
            parsed = json.loads(m.group(0))
            return parsed
        # if no explicit JSON found, attempt to parse entire content
        return json.loads(resp_text)
    except Exception as e:
        logger.warning("Failed to parse JSON from LLM response: %s", e)
        return None

def validate_and_fill(parsed: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """
    Ensure all required fields exist. If missing or empty, attempt regex fallback for deterministic fields.
    Return a dict with exactly FIELDS keys (values may be empty strings).
    """
    out = {f: "" for f in FIELDS}
    if parsed:
        for f in FIELDS:
            v = parsed.get(f, "") if isinstance(parsed, dict) else ""
            out[f] = v if v is not None else ""
    # fallback for deterministic items
    fallback = fallback_regex_fields(original_text)
    for f in ["Email ID", "Contact Number", "PAN Card", "Aadhar Card", "Birth Date"]:
        if not out.get(f):
            out[f] = fallback.get(f, "")
    return out

# ---------- Main per-resume flow ----------
def parse_resume_llm(pdf_path: Path, use_ocr_if_needed: bool = True) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    logger.info("Processing: %s", pdf_path)

    # 1) Try selectable text
    text = extract_text_selectable(pdf_path)
    used_ocr = False
    if not text or len(text.strip()) < 50:
        if use_ocr_if_needed:
            text = extract_text_ocr(pdf_path)
            used_ocr = True
        else:
            logger.error("No selectable text and OCR disabled.")
            text = ""

    # save raw text for audit
    audit_base = AUDIT_DIR / pdf_path.stem
    audit_base.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_base.with_suffix(".raw.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)

    if not text.strip():
        logger.error("No text extracted from %s", pdf_path)
        return {f: "" for f in FIELDS}

    # 2) Chunk text and call LLM for each chunk (we will merge results preferring first non-empty values)
    chunks = chunk_text(text, limit=MAX_PROMPT_CHARS)
    merged = {f: "" for f in FIELDS}
    for i, chunk in enumerate(chunks, start=1):
        logger.info("Calling LLM for chunk %d/%d", i, len(chunks))
        prompt = build_prompt_for_chunk(chunk)
        resp = call_llm(prompt)
        if not resp:
            logger.warning("No response from LLM for chunk %d", i)
            continue
        # save chunk response
        with open(audit_base.with_suffix(f".chunk{i}.llm.txt"), "w", encoding="utf-8") as fh:
            fh.write(resp)
        parsed = extract_json_from_response(resp)
        validated = validate_and_fill(parsed or {}, chunk)
        # merge: fill only empty fields in merged
        for f in FIELDS:
            if not merged[f] and validated.get(f):
                merged[f] = validated[f]

    # 3) Final validation: run a final pass of regex fallback on full text to fill truly missing deterministic fields
    fallback = fallback_regex_fields(text)
    for f in ["Email ID", "Contact Number", "PAN Card", "Aadhar Card", "Birth Date"]:
        if not merged.get(f):
            merged[f] = fallback.get(f, "")

    # 4) Save final JSON audit
    final = {"File Name": pdf_path.name}
    final.update(merged)
    with open(audit_base.with_suffix(".parsed.json"), "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2, ensure_ascii=False)

    logger.info("Parsed %s", pdf_path)
    return final

# ---------- Batch / CLI ----------
def process_batch(folder: str = "resumes", out_xlsx: str = "llm_resume_output.xlsx"):
    folder = Path(folder)
    if not folder.exists():
        logger.error("Folder not found: %s", folder)
        return
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s", folder)
        return
    rows = []
    for p in pdfs:
        try:
            res = parse_resume_llm(p)
            rows.append(res)
            # small pause to avoid bursts (LLM rate limits)
            time.sleep(1.0 + random.random() * 1.0)
        except Exception as e:
            logger.exception("Failed processing %s: %s", p, e)
    df = pd.DataFrame(rows)
    df.to_excel(out_xlsx, index=False)
    with open(Path(out_xlsx).with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Batch complete. Output: %s", out_xlsx)

# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LLM-first resume parser")
    p.add_argument("--single", type=str, help="Single PDF path")
    p.add_argument("--batch", type=str, help="Folder containing PDFs (default resumes)", default="resumes")
    p.add_argument("--out", type=str, help="Output Excel file (batch mode)", default="llm_resume_output.xlsx")
    args = p.parse_args()

    if args.single:
        res = parse_resume_llm(Path(args.single))
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        process_batch(args.batch, args.out)
