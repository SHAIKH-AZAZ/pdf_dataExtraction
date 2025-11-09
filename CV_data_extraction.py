import os
import re
import requests
import json
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from dotenv import load_dotenv
import time



#  OPENROUTER_API_KEY=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


os.environ["TESSDATA_PREFIX"] = "/usr/share/tessdata/"

load_dotenv()

# Load LayoutLMv3
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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

# Configuration for batch processing
MAX_TEXT_LENGTH = 4000  # Limit text length to avoid overwhelming the model
RATE_LIMIT_DELAY = 5  # Seconds between API calls (increased for free tier)
MAX_RETRIES = 3  # Number of retries for failed API calls
RATE_LIMIT_BACKOFF = 30  # Seconds to wait when hitting rate limit (429 error)


# -------------------- OCR + LayoutLMv3 --------------------


def extract_text_layout(image):
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i, word in enumerate(ocr_data["text"]):
        if word.strip():
            x, y, w, h = (
                ocr_data["left"][i],
                ocr_data["top"][i],
                ocr_data["width"][i],
                ocr_data["height"][i],
            )
            boxes.append([x, y, x + w, y + h])
            words.append(word)
    return words, boxes


def clean_text(text):
    """Clean and normalize extracted text to reduce noise"""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters that might confuse the model
    text = re.sub(r"[^\w\s@.,\-/():]", "", text)
    # Limit length to avoid overwhelming the model
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    return text.strip()


def extract_text_from_resume(pdf_path):
    """Extract text from PDF with proper cleaning"""
    pages = convert_from_path(pdf_path)
    all_text = ""
    for page in pages:
        text = pytesseract.image_to_string(page)
        all_text += "\n" + text

    # Clean the extracted text
    cleaned_text = clean_text(all_text)
    return cleaned_text


# -------------------- OpenRouter Integration --------------------


def extract_with_openrouter(text, retry_count=0):
    """Extract structured data from resume text using OpenRouter API with retry logic"""

    # Create a more structured prompt to avoid confusion
    prompt = f"""Extract ONLY the following information from this single resume and return a valid JSON object.

IMPORTANT: This is ONE resume. Extract information for ONE candidate only.

Required fields (use exact keys):
- "Name of Candidate": Full name
- "Birth Date": Date of birth (format: DD/MM/YYYY or DD-MM-YYYY)
- "Marital Status": Married/Single/etc
- "Permanent Address": Full permanent address
- "Contact Number": Phone number
- "Email ID": Email address
- "Education": Educational qualifications
- "Total Years of Experience": Number (e.g., "5" or "5 years")
- "Experience Details": Work experience summary
- "Current Monthly Salary": Salary amount
- "District": District name
- "Gender": Male/Female/Other
- "Present Address": Current address
- "PAN Card": PAN number
- "Aadhar Card": Aadhar number
- "State": State name
- "Preferred Job Location": Preferred location

Resume Text:
{text}

Return ONLY valid JSON with the exact field names above. If a field is not found, use empty string "".
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [
            {
                "role": "system",
                "content": "You are a precise resume parser. Extract information from ONE resume at a time and return valid JSON only. Never mix data from multiple resumes.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,  # Lower temperature for more consistent output
        "max_tokens": 1000,
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=30
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]

        # Try to extract JSON from the response (in case there's extra text)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)

        json_data = json.loads(content)

        # Validate that we have the expected fields
        result = {field: json_data.get(field, "") for field in FIELDS}
        return result

    except requests.exceptions.HTTPError as e:
        # Special handling for rate limit errors (429)
        if e.response.status_code == 429:
            print(f"⚠️ Rate Limit Hit (429 - Too Many Requests)")
            if retry_count < MAX_RETRIES:
                wait_time = RATE_LIMIT_BACKOFF * (retry_count + 1)  # Exponential backoff
                print(f"⏳ Waiting {wait_time}s before retry (Attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                return extract_with_openrouter(text, retry_count + 1)
            else:
                print("❌ Max retries reached. Returning empty data.")
                return {field: "" for field in FIELDS}
        else:
            print(f"⚠️ API Request Error: {e}")
            if retry_count < MAX_RETRIES:
                print(f"🔄 Retrying... (Attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(RATE_LIMIT_DELAY * 2)
                return extract_with_openrouter(text, retry_count + 1)
            else:
                print("❌ Max retries reached. Returning empty data.")
                return {field: "" for field in FIELDS}

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network Error: {e}")
        if retry_count < MAX_RETRIES:
            print(f"🔄 Retrying... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RATE_LIMIT_DELAY * 2)
            return extract_with_openrouter(text, retry_count + 1)
        else:
            print("❌ Max retries reached. Returning empty data.")
            return {field: "" for field in FIELDS}

    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ Error parsing response: {e}")
        if retry_count < MAX_RETRIES:
            print(f"🔄 Retrying... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RATE_LIMIT_DELAY)
            return extract_with_openrouter(text, retry_count + 1)
        else:
            print("❌ Max retries reached. Returning empty data.")
            return {field: "" for field in FIELDS}


# -------------------- Main Resume Parsing --------------------


def parse_resume(pdf_path):
    """Parse a single resume with proper error handling"""
    print(f"📄 Processing: {pdf_path}")

    try:
        # Extract text from PDF
        text = extract_text_from_resume(pdf_path)

        if not text or len(text.strip()) < 50:
            print(f"⚠️ Warning: Very little text extracted from {pdf_path}")
            return {field: "" for field in FIELDS}

        # Extract structured data using OpenRouter
        structured_data = extract_with_openrouter(text)

        print(f"✅ Successfully processed: {pdf_path}")
        return structured_data

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return {field: "" for field in FIELDS}


# -------------------- Batch Processing --------------------


def process_resumes_batch(pdf_folder="resumes"):
    """Process multiple resumes ONE AT A TIME with proper separation and rate limiting"""

    if not os.path.exists(pdf_folder):
        print(f"❌ Folder '{pdf_folder}' not found!")
        return

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"❌ No PDF files found in '{pdf_folder}'")
        return

    print(f"\n🚀 Starting ONE-AT-A-TIME processing of {len(pdf_files)} resumes...")
    print(f"⏱️  Rate limit: {RATE_LIMIT_DELAY}s between each resume")
    print(f"📋 Processing order: Sequential (one complete before next starts)\n")

    rows = []
    successful = 0
    failed = 0

    for idx, file in enumerate(pdf_files, 1):
        print(f"\n{'='*60}")
        print(f"RESUME {idx}/{len(pdf_files)}: {file}")
        print(f"{'='*60}")

        full_path = os.path.join(pdf_folder, file)

        # CRITICAL: Process ONE resume completely before moving to next
        # This ensures the model doesn't mix data between resumes
        data = parse_resume(full_path)
        data["File Name"] = file

        # Check if processing was successful
        if any(data.get(field, "") for field in FIELDS):
            successful += 1
        else:
            failed += 1

        rows.append(data)

        # IMPORTANT: Wait between resumes to ensure complete separation
        if idx < len(pdf_files):
            print(f"\n⏳ Waiting {RATE_LIMIT_DELAY}s before processing next resume...")
            print(f"   (This ensures model processes each resume separately)")
            time.sleep(RATE_LIMIT_DELAY)

    # Save results
    df = pd.DataFrame(rows)
    output_file = "openrouter_resume_data.xlsx"
    df.to_excel(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"✅ ONE-AT-A-TIME BATCH PROCESSING COMPLETE!")
    print(f"📊 Total resumes: {len(rows)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"💾 Data saved to: {output_file}")
    print("=" * 60)


# -------------------- Single Resume Processing --------------------


def process_single_resume(pdf_path):
    """Process just ONE resume - useful for testing"""

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    print(f"\n🎯 Processing SINGLE resume: {pdf_path}")
    print("=" * 60)

    # Process the single resume
    data = parse_resume(pdf_path)
    data["File Name"] = os.path.basename(pdf_path)

    # Save to Excel
    df = pd.DataFrame([data])
    output_file = "single_resume_data.xlsx"
    df.to_excel(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"✅ Single resume processed!")
    print(f"💾 Data saved to: {output_file}")
    print("=" * 60)

    # Also print the extracted data
    print("\n📋 Extracted Data:")
    for field, value in data.items():
        if value:
            print(f"  {field}: {value}")


# -------------------- Run Processing --------------------

if __name__ == "__main__":
    import sys

    # Check if user wants to process a single file
    if len(sys.argv) > 1:
        # Process single resume: python CV_data_extraction.py path/to/resume.pdf
        process_single_resume(sys.argv[1])
    else:
        # Process all resumes in folder (ONE AT A TIME)
        process_resumes_batch("resumes")
