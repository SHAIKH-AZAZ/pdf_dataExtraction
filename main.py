from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import torch
import re
import pandas as pd

# Load pretrained model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

def extract_text_layout(image):
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = ocr_data["text"]
    boxes = []
    for i in range(len(words)):
        if words[i].strip():
            (x, y, w, h) = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
            boxes.append([x, y, x + w, y + h])
    return words, boxes

def parse_resume(pdf_path):
    pages = convert_from_path(pdf_path)
    all_text = ""
    for page in pages:
        words, boxes = extract_text_layout(page)
        encoded = processor(page, words, boxes=boxes, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**encoded)
        text = " ".join(words)
        all_text += " " + text
    return all_text

def extract_fields(text):
    name = re.search(r"Name[:\- ]*(.*)", text, re.IGNORECASE)
    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    phone = re.search(r"(\+?\d[\d\s\-\(\)]{8,}\d)", text)
    skills = re.findall(r"(Python|FastAPI|React|SQL|C\+\+|Java)", text, re.IGNORECASE)

    return {
        "Name": name.group(1).strip() if name else "",
        "Email": email.group(0) if email else "",
        "Phone": phone.group(0) if phone else "",
        "Skills": ", ".join(set(skills)),
    }

pdf_folder = "resumes"
rows = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        text = parse_resume(os.path.join(pdf_folder, file))
        info = extract_fields(text)
        info["File"] = file
        rows.append(info)

df = pd.DataFrame(rows)
df.to_excel("layoutlmv3_resume_data.xlsx", index=False)
print("✅ AI Resume data saved to layoutlmv3_resume_data.xlsx")
