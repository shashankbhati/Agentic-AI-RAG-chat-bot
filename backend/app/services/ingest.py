from pathlib import Path
import pdfplumber
import json
from datetime import datetime
import logging

# -----------------------------
# Folder setup
# -----------------------------
RAW_FOLDER = Path("../../../data/raw")
PROCESSED_FOLDER = Path("../../../data/processed")
PROCESSED_FOLDER.mkdir(exist_ok=True)

# -----------------------------
# Logger setup
# -----------------------------
LOG_FILE = Path("../../../data/ingestion.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_file(file_name: str, status: str, message: str = ""):
    logging.info(f"{file_name} - {status} - {message}")

# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(file_path: Path):
    """
    Extracts text, headings, page numbers, title from a PDF file
    """
    text_data = ""
    headings = []
    page_numbers = []
    title = file_path.stem  # fallback: filename as title

    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_data += page_text + "\n"
                    page_numbers.append(i)

                    # simple heading extraction: all-caps lines > 3 chars
                    for line in page_text.split("\n"):
                        if line.isupper() and len(line) > 3:
                            headings.append({"page": i, "heading": line.strip()})
    except Exception as e:
        log_file(file_path.name, "FAILURE", str(e))
        return None

    return {
        "text": text_data.strip(),
        "title": title,
        "headings": headings,
        "pages": page_numbers
    }

# -----------------------------
# PDF processing
# -----------------------------
def process_pdf(file_path: Path):
    pdf_data = extract_pdf_text(file_path)
    if pdf_data is None:
        return None

    metadata = {
        "filename": file_path.name,
        "length": len(pdf_data["text"]),
        "processed_at": datetime.utcnow().isoformat(),
        "version": "v1.0"
    }
    pdf_data["metadata"] = metadata
    return pdf_data

# -----------------------------
# Process all PDFs
# -----------------------------
def process_all_pdfs():
    pdf_files = list(RAW_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDFs found in", RAW_FOLDER)
        return

    print(f"Starting ingestion at {datetime.utcnow().isoformat()}")
    for pdf_file in pdf_files:
        try:
            pdf_data = process_pdf(pdf_file)
            if pdf_data is None:
                continue

            output_path = PROCESSED_FOLDER / (pdf_file.stem + ".json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pdf_data, f, ensure_ascii=False, indent=2)

            log_file(pdf_file.name, "SUCCESS")
            print(f"✅ Processed {pdf_file.name} → {output_path.name}")

        except Exception as e:
            log_file(pdf_file.name, "FAILURE", str(e))
            print(f"❌ Failed {pdf_file.name}: {e}")

    print(f"Ingestion finished at {datetime.utcnow().isoformat()}")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    process_all_pdfs()
