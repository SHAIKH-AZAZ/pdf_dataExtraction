import os
import re
import json
import time
import logging
import requests
import pandas as pd
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Env / Constants
# -------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1.x models are being retired; use 2.x models now.
# You can also try "gemini-2.5-flash" if you have it enabled.
MODEL_NAME = "gemini-2.0-flash"

# Tesseract config (adjust path if needed)
os.environ["TESSDATA_PREFIX"] = "/usr/share/tessdata/"

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


class ResumeParser:
    def __init__(self, api_key=GEMINI_API_KEY, model=MODEL_NAME):
        self.api_key = api_key
        self.model = model
        self.max_retries = 3
        self.rate_limit_delay = 5  # seconds between resumes

        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY is not set! AI extraction will fail.")

    # -------------------------------------------------
    # Text extraction
    # -------------------------------------------------
    def extract_text(self, pdf_path):
        """
        Hybrid text extraction:
        1. Try native PDF text extraction (fast, accurate).
        2. If text is minimal/empty, fallback to OCR (slow, robust).
        """
        text = ""
        method = "Native"

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                text = "\n".join(pages).strip()

            if len(text) < 50:
                logger.info(
                    f"📉 Low text count ({len(text)} chars). Falling back to OCR..."
                )
                method = "OCR"
                text = self.extract_text_ocr(pdf_path)

        except Exception as e:
            logger.warning(
                f"⚠️ Native extraction failed: {e}. Falling back to OCR..."
            )
            method = "OCR"
            text = self.extract_text_ocr(pdf_path)

        return text, method

    def extract_text_ocr(self, pdf_path):
        """Fallback OCR extraction using pdf2image + pytesseract"""
        try:
            images = convert_from_path(pdf_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            return ""

    def clean_text(self, text):
        """Basic text cleaning"""
        text = re.sub(r"\s+", " ", text)
        # Limit length to avoid context/token limits
        return text[:30000].strip()

    # -------------------------------------------------
    # Gemini API call
    # -------------------------------------------------
    def extract_data_with_ai(self, text):
        """Extract structured data using Google Gemini API (JSON mode)."""
        if not self.api_key:
            return {field: "" for field in FIELDS}

        prompt = f"""
        You are a precise resume parser. Extract the following details from the resume text below.
        Return ONLY a valid JSON object. Do not add any markdown formatting.

        Required Fields:
        {json.dumps(FIELDS, indent=2)}

        Resume Text:
        {text}

        If a field is not found, return an empty string "".
        """

        # v1beta generateContent endpoint with model in the path
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )

        headers = {
            "Content-Type": "application/json",
        }

        # Use JSON Mode so the model directly returns JSON text
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                # IMPORTANT: snake_case, not camelCase
                "response_mime_type": "application/json",
            },
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=60
                )

                if response.status_code == 429:
                    wait_time = (attempt + 1) * 10
                    logger.warning(
                        f"⏳ Rate limit hit ({response.status_code}). "
                        f"Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                # Extract the model output
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    # Most JSON-mode responses put JSON in the first text part
                    content = ""
                    for p in parts:
                        if "text" in p:
                            content += p["text"]
                    content = content.strip()
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(
                        f"⚠️ Unexpected response format from Gemini: {e} | Raw: {data}"
                    )
                    continue

                # Try to parse JSON
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"⚠️ Failed to parse JSON from model output: {e}\n"
                        f"Raw content: {content[:500]}..."
                    )
                    continue

                # Ensure all expected fields exist
                result = {field: parsed.get(field, "") for field in FIELDS}
                return result

            except Exception as e:
                logger.error(
                    f"⚠️ AI Extraction error (Attempt {attempt+1}/{self.max_retries}): {e}"
                )
                time.sleep(2)

        logger.error("❌ Max retries reached for AI extraction.")
        return {field: "" for field in FIELDS}

    # -------------------------------------------------
    # Orchestration
    # -------------------------------------------------
    def process_resume(self, pdf_path):
        """Process a single resume"""
        logger.info(f"📄 Processing: {os.path.basename(pdf_path)}")

        # 1. Extract Text
        raw_text, method = self.extract_text(pdf_path)
        logger.info(f"   ↳ Extracted {len(raw_text)} chars using {method}")

        if not raw_text:
            logger.error("   ❌ No text found.")
            data = {field: "" for field in FIELDS}
        else:
            # 2. Clean Text
            cleaned_text = self.clean_text(raw_text)

            # 3. AI Extraction
            data = self.extract_data_with_ai(cleaned_text)

        # Add metadata
        data["File Name"] = os.path.basename(pdf_path)
        data["Extraction Method"] = method

        return data

    def process_batch(self, folder_path, output_file="gemini_resume_data.xlsx"):
        """Process all PDFs in a folder"""
        if not os.path.exists(folder_path):
            logger.error(f"❌ Folder not found: {folder_path}")
            return

        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        logger.info(f"🚀 Starting batch processing for {len(pdf_files)} resumes...")

        results = []
        for i, file in enumerate(pdf_files, 1):
            full_path = os.path.join(folder_path, file)

            data = self.process_resume(full_path)
            results.append(data)

            if i < len(pdf_files):
                logger.info(f"⏳ Waiting {self.rate_limit_delay}s...")
                time.sleep(self.rate_limit_delay)

        # Save to Excel
        df = pd.DataFrame(results)

        # Ensure all fields are present as columns
        for field in FIELDS:
            if field not in df.columns:
                df[field] = ""

        cols = ["File Name", "Extraction Method"] + FIELDS
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        df.to_excel(output_file, index=False)
        logger.info(f"✅ Batch processing complete. Saved to {output_file}")


# -------------------------------------------------
# CLI entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    import sys

    parser = ResumeParser()

    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target):
            result = parser.process_resume(target)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif os.path.isdir(target):
            parser.process_batch(target)
        else:
            logger.error(f"❌ Path not found: {target}")
    else:
        # Default: process "resumes" folder in CWD
        parser.process_batch("resumes")
