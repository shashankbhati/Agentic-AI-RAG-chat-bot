import pdfplumber
import json
from pathlib import Path

raw_folder = Path("../../../data/raw")       # relative to this script
processed_folder = Path("../../../data/processed")
processed_folder.mkdir(exist_ok=True)

for pdf_file in raw_folder.glob("*.pdf"):
    text_data = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_data += page_text + "\n"
    
    # save cleaned text as JSON
    output = {
        "filename": pdf_file.name,
        "text": text_data.strip()
    }
    out_file = processed_folder / (pdf_file.stem + ".json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

print("✅ All PDFs processed to JSON")
