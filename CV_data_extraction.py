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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b:free"

# Tesseract configuration (if needed)
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
    def __init__(self, api_key=OPENROUTER_API_KEY, model=MODEL_NAME):
        self.api_key = api_key
        self.model = model
        self.max_retries = 3
        self.rate_limit_delay = 5  # Seconds between requests

        if not self.api_key:
            logger.warning("⚠️ OPENROUTER_API_KEY is not set! AI extraction will fail.")

    def extract_text(self, pdf_path):
        """
        Hybrid text extraction:
        1. Try native PDF text extraction (fast, accurate).
        2. If text is minimal/empty, fallback to OCR (slow, robust).
        """
        text = ""
        method = "Native"
        
        try:
            # Method 1: Native Extraction with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                text = "\n".join(pages).strip()

            # Check if native extraction yielded enough text
            if len(text) < 50:
                logger.info(f"📉 Low text count ({len(text)} chars). Falling back to OCR...")
                method = "OCR"
                text = self.extract_text_ocr(pdf_path)
            
        except Exception as e:
            logger.warning(f"⚠️ Native extraction failed: {e}. Falling back to OCR...")
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
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Limit length to avoid token limits (approx 4000 words ~ 5-6k tokens)
        return text[:30000].strip()

    def extract_data_with_ai(self, text):
        """Extract structured data using OpenRouter API"""
        if not self.api_key:
            return {field: "" for field in FIELDS}

        prompt = f"""
        You are a precise resume parser. Extract the following details from the resume text below.
        Return ONLY a valid JSON object. Do not add any markdown formatting (like ```json).
        
        Required Fields:
        {json.dumps(FIELDS, indent=2)}

        Resume Text:
        {text}

        If a field is not found, return an empty string "".
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts data from resumes as JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
                
                # Check for rate limiting
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 15
                    logger.warning(f"⏳ Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # Log any errors
                if response.status_code != 200:
                    logger.error(f"⚠️ API Error {response.status_code}: {response.text}")
                    time.sleep(2)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                # Extract content from OpenRouter response
                content = data["choices"][0]["message"]["content"]
                
                # Clean up potential markdown code blocks
                content = content.replace("```json", "").replace("```", "").strip()
                
                return json.loads(content)

            except Exception as e:
                logger.error(f"⚠️ AI Extraction error (Attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(3)

        logger.error("❌ Max retries reached for AI extraction.")
        return {field: "" for field in FIELDS}

    def process_resume(self, pdf_path):
        """Process a single resume"""
        logger.info(f"📄 Processing: {os.path.basename(pdf_path)}")
        
        # 1. Extract Text
        raw_text, method = self.extract_text(pdf_path)
        logger.info(f"   ↳ Extracted {len(raw_text)} chars using {method}")

        if not raw_text:
            logger.error("   ❌ No text found.")
            return {field: "" for field in FIELDS}

        # 2. Clean Text
        cleaned_text = self.clean_text(raw_text)

        # 3. AI Extraction
        data = self.extract_data_with_ai(cleaned_text)
        
        # Add metadata
        data["File Name"] = os.path.basename(pdf_path)
        data["Extraction Method"] = method
        
        return data

    def process_batch(self, folder_path, output_file="openrouter_resume_data.xlsx"):
        """Process all PDFs in a folder"""
        if not os.path.exists(folder_path):
            logger.error(f"❌ Folder not found: {folder_path}")
            return

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        logger.info(f"🚀 Starting batch processing for {len(pdf_files)} resumes...")

        results = []
        for i, file in enumerate(pdf_files, 1):
            full_path = os.path.join(folder_path, file)
            
            data = self.process_resume(full_path)
            results.append(data)
            
            # Rate limiting
            if i < len(pdf_files):
                logger.info(f"⏳ Waiting {self.rate_limit_delay}s...")
                time.sleep(self.rate_limit_delay)

        # Save to Excel
        df = pd.DataFrame(results)
        
        # Ensure all fields are present
        for field in FIELDS:
            if field not in df.columns:
                df[field] = ""
                
        # Reorder columns
        cols = ["File Name", "Extraction Method"] + FIELDS
        # Filter cols that exist
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        df.to_excel(output_file, index=False)
        logger.info(f"✅ Batch processing complete. Saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    parser = ResumeParser()
    
    if len(sys.argv) > 1:
        # Single file mode
        target = sys.argv[1]
        if os.path.isfile(target):
            result = parser.process_resume(target)
            print(json.dumps(result, indent=2))
        elif os.path.isdir(target):
            parser.process_batch(target)
    else:
        # Default batch mode
        parser.process_batch("resumes")
