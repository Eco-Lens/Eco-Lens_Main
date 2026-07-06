from pathlib import Path
import re

import fitz  # PyMuPDF


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = PROJECT_ROOT / "Dataset"
IMG_DIR = PROJECT_ROOT / "DatasetIMG"

ZOOM = 4


def extract_year(filename: str) -> str:
    match = re.search(r"(20\d{2})", filename)

    if match:
        return match.group(1)

    return "unknown"


def create_company_name(report_name: str, year: str) -> str:
    """
    Chuyển tên PDF thành tên công ty ngắn gọn:
    Asia Pacific Securities JSC - 2024 - Annual Report
    ->
    asia_pacific_securities_jsc
    """

    company_name = report_name

    if year != "unknown":
        company_name = company_name.split(year)[0]

    company_name = re.sub(r"[^a-zA-Z0-9]+", "_", company_name)
    company_name = company_name.strip("_").lower()

    return company_name


def process_pdf(pdf_path: Path):

    report_name = pdf_path.stem

    year = extract_year(report_name)

    company_name = create_company_name(
        report_name,
        year
    )

    output_dir = IMG_DIR / report_name

    if not output_dir.exists():
        print(f"[WARNING] Không tìm thấy folder: {output_dir}")
        return

    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(ZOOM, ZOOM)

    try:
        total_pages = doc.page_count

        for page_index in range(total_pages):

            page = doc.load_page(page_index)

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_name = (
                f"{company_name}"
                f"_{year}"
                f"_p{page_index + 1:03d}.jpg"
            )

            output_path = output_dir / image_name

            pix.save(output_path)

        print(
            f"[DONE] {report_name} "
            f"({total_pages} pages)"
        )

    finally:
        doc.close()


def process_all_pdfs():

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:

        try:
            process_pdf(pdf_file)

        except Exception as e:

            print(f"[ERROR] {pdf_file.name}")
            print(e)


if __name__ == "__main__":
    process_all_pdfs()