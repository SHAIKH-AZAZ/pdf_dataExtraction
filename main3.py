#!/usr/bin/env python3
"""
resume_parser_improved.py

A robust resume parsing pipeline:
- Hybrid text extraction (PyMuPDF for selectable text, else pdf2image + Tesseract OCR)
- Image preprocessing (OpenCV)
- Layout-aware token/box alignment via LayoutLMv3Processor (for future token-classification)
- Regex fallback extraction for high-confidence fields (email, phone, PAN, Aadhaar, dates, salary)
- OpenRouter LLM fallback with exponential jitter backoff and Retry-After handling
- Logging, audit files (raw OCR + JSON), and Excel/JSON outputs
- Field-level extraction_method metadata: "layout", "regex", "llm"
"""

import os
import re
import json
import time
import math
import random
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import requests
import pandas as pd
import phonenumbers
from dateutil import parser as dateutil_parser
from dotenv import load_dotenv

# Optional: LayoutLMv3 processor (used for token/box alignment)
from transformers import LayoutLMv3Processor

# -------------------------
# Config / Constants
# -------------------------
load_dotenv()

# Tesseract data prefix (adjust if needed)
os.environ["TESSDATA_PREFIX"] = os.getenv("TESSDATA_PREFIX", "/usr/share/tessdata/")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

# Fields we want
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

# Behavior config
OCR_DPI = 300
MAX_TEXT_LENGTH = 4000
RATE_LIMIT_DELAY = 2
MAX_RETRIES = 4
RATE_LIMIT_BACKOFF_BASE = 5

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("resume_parser")

# LayoutLMv3 Processor - used only for token-box alignment
try:
    layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
except Exception as e:
    logger.warning("Could not initialize LayoutLMv3Processor: %s. Layout features disabled.", e)
    layout_processor = None

# -------------------------
# Helpers: Image preprocessing
# -------------------------
def preprocess_image_pil(pil_img: Image.Image, dpi: int = OCR_DPI) -> Image.Image:
    """
    Convert to grayscale, resize for DPI, denoise / binarize.
    Returns a PIL.Image (mode L or RGB).
    """
    # Ensure mode
    img = pil_img.convert("RGB")
    # Resize approximating DPI (pdf2image should give decent DPI already)
    # Convert to numpy for OpenCV ops
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Optional: deskew (approx)
    coords = np.column_stack(np.where(gray > 0))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Binarize with adaptive thresholding
    try:
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    except Exception:
        _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Denoise
    bin_img = cv2.medianBlur(bin_img, 3)

    # Convert back to PIL
    pil_out = Image.fromarray(bin_img)
    return pil_out


# -------------------------
# Helpers: Text extraction
# -------------------------
def extract_text_from_pdf_selectable(pdf_path: str) -> str:
    """
    Attempt to extract selectable text using PyMuPDF (fast and accurate if present).
    """
    try:
        doc = fitz.open(pdf_path)
        text_chunks = []
        for page in doc:
            text = page.get_text("text")
            if text and len(text.strip()) > 5:
                text_chunks.append(text)
        doc.close()
        return "\n".join(text_chunks).strip()
    except Exception as e:
        logger.debug("PyMuPDF extraction error: %s", e)
        return ""


def extract_text_from_pdf_ocr(pdf_path: str, audit_folder: Optional[str] = None) -> str:
    """
    Convert to images, preprocess each page, run Tesseract OCR, return combined text.
    Also saves raw OCR per-page to audit folder if provided.
    """
    logger.info("Converting PDF to images (dpi=%s): %s", OCR_DPI, pdf_path)
    pages = convert_from_path(pdf_path, dpi=OCR_DPI)
    all_text = []
    for i, page in enumerate(pages):
        logger.debug("Preprocessing page %d", i + 1)
        pre = preprocess_image_pil(page)
        # Tesseract config: psm 3 or 6 depending on structure; pick 6 (assumes a uniform block)
        config = "--psm 6"
        page_text = pytesseract.image_to_string(pre, config=config)
        page_text = page_text.strip()
        all_text.append(page_text)
        if audit_folder:
            Path(audit_folder).mkdir(parents=True, exist_ok=True)
            with open(Path(audit_folder) / f"{Path(pdf_path).stem}_page{i+1}_ocr.txt", "w", encoding="utf-8") as fh:
                fh.write(page_text)
    combined = "\n".join(all_text)
    return combined


def extract_words_and_boxes_from_image(pil_img: Image.Image) -> Tuple[List[str], List[List[int]]]:
    """
    Use pytesseract.image_to_data to get words + bounding boxes.
    Returns (words, boxes) where boxes are [x0, y0, x1, y1]
    """
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    words = []
    boxes = []
    for i, w in enumerate(data.get("text", [])):
        txt = str(w).strip()
        if txt:
            x, y, w_box, h_box = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append(txt)
            boxes.append([x, y, x + w_box, y + h_box])
    return words, boxes


# -------------------------
# Helpers: Normalization & regex
# -------------------------
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}", re.I)
RE_IN_PHONE = re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}\b")
RE_GENERIC_PHONE = re.compile(r"(?:\+?\d[\d\-\s]{7,}\d)")
RE_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
RE_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
RE_DATE = re.compile(r"\b(?:\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}|\w{3,9}\s\d{1,2},\s?\d{4}|\d{4}-\d{2}-\d{2})\b")
RE_SALARY = re.compile(r"[\₹$€]?\s?[\d,]+(?:\.\d+)?\s*(?:per month|/month|pm|per annum|pa|per year|p\.a\.|annum|year)?", re.I)

def normalize_date(text: str) -> Optional[str]:
    try:
        dt = dateutil_parser.parse(text, dayfirst=True, fuzzy=True)
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return None

def extract_first(regex: re.Pattern, text: str) -> Optional[str]:
    m = regex.search(text)
    return m.group(0).strip() if m else None

def extract_all(regex: re.Pattern, text: str) -> List[str]:
    return [m.group(0).strip() for m in regex.finditer(text)]

def normalize_phone(text: str) -> Optional[str]:
    try:
        # Try parsing with phonenumbers (best-effort)
        parsed = phonenumbers.parse(text, "IN")
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        # fallback: digits only
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 10:
            return digits[-10:]
    return None

def parse_salary_to_monthly(value: str) -> Optional[float]:
    if not value: return None
    s = value.lower().replace(",", "")
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m: return None
    val = float(m.group(1))
    if "lakh" in s or "lac" in s:
        val = val * 100000
    if "per month" in s or "/month" in s or "pm" in s:
        return val
    if "per annum" in s or "pa" in s or "per year" in s or "annum" in s:
        return val / 12.0
    # guess: if value > 10000 assume annual? We won't guess; return raw
    return val

# -------------------------
# Layout-aware token alignment (optional)
# -------------------------
def layout_align_tokens(pil_img: Image.Image, words: List[str], boxes: List[List[int]]):
    """
    Use LayoutLMv3Processor to align tokens -> boxes.
    Returns encoding (tokens, attention_mask, bbox) that can be fed to a token-classifier.
    If layout_processor is None, returns None.
    """
    if layout_processor is None:
        return None
    try:
        encoding = layout_processor(images=pil_img, words=words, boxes=boxes, return_tensors="pt", truncation=True)
        return encoding
    except Exception as e:
        logger.debug("LayoutLMv3Processor alignment error: %s", e)
        return None

# -------------------------
# Local heuristic extraction
# -------------------------
def heuristic_extract_all(text: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Run a set of regex/heuristic extractors and return a dict of field->value and field->method.
    This is our reliable local fallback.
    """
    results = {f: "" for f in FIELDS}
    methods = {f: "" for f in FIELDS}

    # Emails
    email = extract_first(RE_EMAIL, text)
    if email:
        results["Email ID"] = email
        methods["Email ID"] = "regex"

    # Phone (prioritize IN phone)
    phone = extract_first(RE_IN_PHONE, text) or extract_first(RE_GENERIC_PHONE, text)
    if phone:
        normalized = normalize_phone(phone)
        results["Contact Number"] = normalized or phone
        methods["Contact Number"] = "regex"

    # PAN / Aadhar
    pan = extract_first(RE_PAN, text)
    if pan:
        results["PAN Card"] = pan
        methods["PAN Card"] = "regex"

    aad = extract_first(RE_AADHAAR, text)
    if aad:
        results["Aadhar Card"] = aad
        methods["Aadhar Card"] = "regex"

    # Dates (DOB guess: look for 'DOB' or 'Date of Birth' near a date)
    dob = None
    for match in RE_DATE.finditer(text):
        surrounding = text[max(0, match.start()-40):match.end()+40].lower()
        if "birth" in surrounding or "dob" in surrounding or "date of birth" in surrounding:
            dob_candidate = normalize_date(match.group(0))
            if dob_candidate:
                dob = dob_candidate
                break
    if dob:
        results["Birth Date"] = dob
        methods["Birth Date"] = "regex"
    else:
        # fallback: first date that looks like a DOB (year between 1950 and 2005)
        for match in RE_DATE.finditer(text):
            normalized = normalize_date(match.group(0))
            if normalized:
                year = int(normalized.split("/")[-1])
                if 1950 <= year <= 2005:
                    results["Birth Date"] = normalized
                    methods["Birth Date"] = "regex"
                    break

    # Salary
    sal = extract_first(RE_SALARY, text)
    if sal:
        parsed_sal = parse_salary_to_monthly(sal)
        results["Current Monthly Salary"] = parsed_sal if parsed_sal else sal
        methods["Current Monthly Salary"] = "regex"

    # Experience: look for patterns like "X years"
    m = re.search(r"(\d+)\s+years?", text, re.I)
    if m:
        results["Total Years of Experience"] = m.group(1)
        methods["Total Years of Experience"] = "regex"

    # Education: heuristics - look for degrees
    edu_keywords = ["bachelor", "b\.tech", "btech", "be ", "mtech", "m\.tech", "mba", "msc", "bsc", "phd", "degree", "diploma"]
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in edu_keywords) and len(line.strip()) > 8:
            results["Education"] = line.strip()
            methods["Education"] = "regex"
            break

    # Name heuristic: line at top, often first non-empty line (use with caution)
    top_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if top_lines:
        candidate_name = top_lines[0]
        # avoid grabbing "resume" or "curriculum vitae"
        if len(candidate_name.split()) <= 5 and not re.search(r"resume|curriculum vitae|cv", candidate_name, re.I):
            # also avoid emails or phone numbers
            if "@" not in candidate_name and not RE_IN_PHONE.search(candidate_name):
                results["Name of Candidate"] = candidate_name
                methods["Name of Candidate"] = "heuristic"

    # Addresses & present/permanent address heuristics (naive: look for 'permanent' / 'present' keywords)
    if "permanent address" in text.lower():
        m = re.search(r"(Permanent Address[:\s]*)(.+?)(?:\n\n|\n[A-Z][a-z]+:|\Z)", text, re.I | re.S)
        if m:
            addr = m.group(2).strip().split("\n")[0:4]
            results["Permanent Address"] = " ".join(addr)
            methods["Permanent Address"] = "regex"
    if "present address" in text.lower() or "current address" in text.lower():
        m = re.search(r"(Present Address|Current Address)[:\s]*([\s\S]{10,200})", text, re.I)
        if m:
            results["Present Address"] = m.group(2).split("\n")[0:4]
            results["Present Address"] = " ".join(results["Present Address"])
            methods["Present Address"] = "regex"

    # Gender (simple)
    if re.search(r"\b(male|female|other|non-binary|nonbinary)\b", text, re.I):
        g = re.search(r"\b(male|female|other|non-binary|nonbinary)\b", text, re.I).group(1)
        results["Gender"] = g.capitalize()
        methods["Gender"] = "regex"

    # District / State / Preferred Job Location - simple heuristics
    # Look for "District:" or "State:" patterns
    for key in ["District", "State", "Preferred Job Location"]:
        m = re.search(rf"{key}\s*[:\-]\s*([A-Za-z0-9 ,\-]+)", text, re.I)
        if m:
            results[key] = m.group(1).strip()
            methods[key] = "regex"

    return results, methods

# -------------------------
# OpenRouter LLM fallback with robust backoff
# -------------------------
def call_openrouter_with_backoff(prompt: str, model: str = "openai/gpt-oss-20b:free",
                                 max_retries: int = MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """
    Call OpenRouter endpoint with exponential backoff + jitter and Retry-After handling.
    Returns JSON-parsed dict or None on failure.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not configured. Skipping LLM fallback.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise resume parser. Return ONLY valid JSON with exact fields."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    attempt = 0
    while attempt <= max_retries:
        try:
            logger.info("OpenRouter attempt %d", attempt + 1)
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                # Expecting content in choices[0].message.content
                try:
                    content = data["choices"][0]["message"]["content"]
                except Exception:
                    content = json.dumps(data)
                # extract JSON object in response if surrounded by text
                m = re.search(r"\{[\s\S]*\}", content)
                if m:
                    parsed = json.loads(m.group(0))
                else:
                    parsed = json.loads(content)
                return parsed
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after)
                else:
                    # exponential backoff + jitter
                    wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 3)
                logger.warning("OpenRouter 429; sleeping %s seconds", wait)
                time.sleep(wait)
            else:
                logger.error("OpenRouter error %d: %s", resp.status_code, resp.text[:200])
                # for 5xx, wait and retry
                if 500 <= resp.status_code < 600:
                    wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 2)
                    time.sleep(wait)
                else:
                    break
        except requests.exceptions.RequestException as e:
            logger.warning("OpenRouter request exception: %s", e)
            wait = RATE_LIMIT_DELAY * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)
        attempt += 1

    logger.error("OpenRouter: max retries exceeded or unrecoverable error.")
    return None

def prompt_for_llm_from_text(text: str) -> str:
    short_text = text if len(text) < MAX_TEXT_LENGTH else text[:MAX_TEXT_LENGTH]
    prompt_fields = "\n".join([f'- "{f}":' for f in FIELDS])
    prompt = f"""Extract ONLY the following information from the resume text and return a VALID JSON object with these exact keys (use empty string "" when not found):

{prompt_fields}

Resume Text:
{short_text}
"""
    return prompt

# -------------------------
# Main per-resume parsing
# -------------------------
def parse_resume(pdf_path: str, audit_dir: str = "audit", use_llm_fallback: bool = True) -> Dict[str, Any]:
    pdf_path = str(pdf_path)
    logger.info("Processing: %s", pdf_path)
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Try text extraction from selectable PDF first
    text = extract_text_from_pdf_selectable(pdf_path)
    used_method = "selectable"

    if not text or len(text.strip()) < 30:
        logger.info("No selectable text found or too short; using OCR.")
        text = extract_text_from_pdf_ocr(pdf_path, audit_folder=str(audit_dir))
        used_method = "ocr"

    # Clean text a bit
    text_clean = re.sub(r"\s+", " ", text).strip()
    if len(text_clean) > MAX_TEXT_LENGTH:
        text_for_llm = text_clean[:MAX_TEXT_LENGTH]
    else:
        text_for_llm = text_clean

    # Run local heuristic extraction first
    heuristic_values, heuristic_methods = heuristic_extract_all(text_clean)

    # Prepare result dict with per-field metadata
    result = {}
    extraction_method = {}
    for f in FIELDS:
        val = heuristic_values.get(f, "")
        result[f] = val if val is not None else ""
        extraction_method[f] = heuristic_methods.get(f, "")

    # If many fields are empty, try OpenRouter LLM to fill gaps
    empty_fields = [f for f in FIELDS if not result.get(f)]
    logger.info("Empty fields after heuristics: %s", empty_fields)
    if use_llm_fallback and len(empty_fields) > 0:
        prompt = prompt_for_llm_from_text(text_for_llm)
        llm_json = call_openrouter_with_backoff(prompt)
        if llm_json:
            # Map fields from LLM into result if empty
            for f in FIELDS:
                candidate = llm_json.get(f, "") if isinstance(llm_json, dict) else ""
                if candidate and not result.get(f):
                    result[f] = candidate
                    extraction_method[f] = "llm"

    # Save audit: raw OCR/selectable text + final json
    base = Path(audit_dir) / Path(pdf_path).stem
    # raw text
    with open(base.with_suffix(".raw.txt"), "w", encoding="utf-8") as fh:
        fh.write(text_clean)
    # result json with methods
    audit_json = {"result": result, "extraction_method": extraction_method, "source_method": used_method}
    with open(base.with_suffix(".parsed.json"), "w", encoding="utf-8") as fh:
        json.dump(audit_json, fh, indent=2, ensure_ascii=False)

    # Add File Name
    result["File Name"] = Path(pdf_path).name
    result["_extraction_method"] = extraction_method

    return result

# -------------------------
# Batch / CLI
# -------------------------
def process_resumes_batch(pdf_folder: str = "resumes", out_excel: str = "resume_data.xlsx", audit_dir: str = "audit"):
    pdf_folder = Path(pdf_folder)
    if not pdf_folder.exists():
        logger.error("Folder not found: %s", pdf_folder)
        return
    pdf_files = sorted([p for p in pdf_folder.glob("*.pdf")])
    if not pdf_files:
        logger.error("No PDFs in %s", pdf_folder)
        return

    rows = []
    success = 0
    fail = 0
    total = len(pdf_files)
    logger.info("Starting batch: %d files", total)
    for idx, pdf in enumerate(pdf_files, start=1):
        logger.info("[%d/%d] %s", idx, total, pdf.name)
        try:
            data = parse_resume(str(pdf), audit_dir=audit_dir, use_llm_fallback=True)
            rows.append(data)
            success += 1
        except Exception as e:
            logger.exception("Failed processing %s: %s", pdf, e)
            fail += 1

        # Respect a small pause to avoid API rate spikes if LLM used
        time.sleep(RATE_LIMIT_DELAY)

    df = pd.DataFrame(rows)
    df.to_excel(out_excel, index=False)
    # Also save as JSON lines for easy ingest
    with open(Path(out_excel).with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Batch complete: total=%d success=%d fail=%d, excel=%s", total, success, fail, out_excel)
    return df

def process_single(pdf_path: str, out_excel: str = "single_resume.xlsx", audit_dir: str = "audit"):
    res = parse_resume(pdf_path, audit_dir=audit_dir, use_llm_fallback=True)
    df = pd.DataFrame([res])
    df.to_excel(out_excel, index=False)
    logger.info("Single processed => %s", out_excel)
    return res

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Resume parsing pipeline (improved)")
    p.add_argument("--single", help="Process single PDF file", type=str)
    p.add_argument("--batch", help="Process all PDFs in folder", type=str, default="resumes")
    p.add_argument("--out", help="Output Excel file", type=str, default="resume_data.xlsx")
    p.add_argument("--audit", help="Audit folder", type=str, default="audit")
    p.add_argument("--no-llm", help="Disable OpenRouter LLM fallback", action="store_true")

    args = p.parse_args()

    if args.single:
        logger.info("Processing single file: %s", args.single)
        res = parse_resume(args.single, audit_dir=args.audit, use_llm_fallback=not args.no_llm)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        logger.info("Processing batch folder: %s", args.batch)
        process_resumes_batch(pdf_folder=args.batch, out_excel=args.out, audit_dir=args.audit)
