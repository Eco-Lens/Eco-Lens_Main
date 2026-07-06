from pathlib import Path
import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = PROJECT_ROOT / "Dataset"


def count_pdf_pages():

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files\n")

    total_pages = 0

    for pdf_file in pdf_files:

        try:

            doc = fitz.open(pdf_file)

            page_count = doc.page_count

            total_pages += page_count

            print(
                f"{pdf_file.name}: "
                f"{page_count} pages"
            )

            doc.close()

        except Exception as e:

            print(
                f"[ERROR] {pdf_file.name}"
            )

            print(e)

    print("\n====================")
    print(f"Total PDFs : {len(pdf_files)}")
    print(f"Total Pages: {total_pages}")


if __name__ == "__main__":
    count_pdf_pages()