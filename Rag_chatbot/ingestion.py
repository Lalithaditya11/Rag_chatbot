import os
import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langdetect import detect
from tqdm import tqdm

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "data" / "pdfs"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

# Windows users only
# Uncomment and change if required
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# ---------------------------------------------------
# LANGUAGE DETECTION
# ---------------------------------------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# ---------------------------------------------------
# OCR PAGE
# ---------------------------------------------------

def perform_ocr(page):
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )
    text = pytesseract.image_to_string(img)
    return text

# ---------------------------------------------------
# PROCESS SINGLE PDF
# ---------------------------------------------------
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    filename = Path(pdf_path).name
    pdf_id = Path(pdf_path).stem
    pages = []
    print(f"\nProcessing : {filename}")
    for page_number in tqdm(range(len(doc))):
        page = doc.load_page(page_number)
        text = page.get_text("text")
        if len(text.strip()) < 50:
            text = perform_ocr(page)
        text = clean_text(text)
        language = detect_language(text)
        page_data = {
            "pdf_id": pdf_id,
            "filename": filename,
            "page_number": page_number + 1,
            "language": language,
            "text": text
        }
        pages.append(page_data)
    return pages

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDFs\n")

    if not pdf_files:
        print("No PDF files found. Place PDFs in the data/pdfs folder and run again.")
        return

    for pdf in pdf_files:
        data = process_pdf(str(pdf))
        output_file = OUTPUT_FOLDER / f"{pdf.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )
        print(f"Saved -> {output_file}")
    print("\nAll PDFs processed successfully.")

if __name__ == "__main__":
    main()