from pathlib import Path
import re

import fitz


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

    company_name = report_name

    if year != "unknown":
        company_name = company_name.split(year)[0]

    company_name = re.sub(r"[^a-zA-Z0-9]+", "_", company_name)
    company_name = company_name.strip("_").lower()

    return company_name


def build_pdf_lookup():

    pdf_lookup = {}

    for pdf_file in PDF_DIR.glob("*.pdf"):

        normalized_name = pdf_file.stem.replace(
            " (SustainabilityReports.com)",
            ""
        ).strip()

        pdf_lookup[normalized_name] = pdf_file

    return pdf_lookup


def process_pdf(pdf_path: Path, output_dir: Path):

    report_name = pdf_path.stem

    normalized_report_name = report_name.replace(
        " (SustainabilityReports.com)",
        ""
    )

    year = extract_year(normalized_report_name)

    company_name = create_company_name(
        normalized_report_name,
        year
    )

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
            f"[DONE] {normalized_report_name} "
            f"({total_pages} pages)"
        )

    finally:
        doc.close()


def process_all_pdfs():

    pdf_lookup = build_pdf_lookup()

    folders = [
        f for f in IMG_DIR.iterdir()
        if f.is_dir()
    ]

    print(f"Found {len(folders)} folders")

    for index, folder in enumerate(folders, start=1):

        try:

            if folder.name not in pdf_lookup:

                print(
                    f"[NOT FOUND] {folder.name}"
                )

                continue

            pdf_path = pdf_lookup[folder.name]

            print(
                f"\n[{index}/{len(folders)}]"
            )

            print(
                f"{folder.name}"
            )

            process_pdf(
                pdf_path,
                folder
            )

        except Exception as e:

            print(
                f"[ERROR] {folder.name}"
            )

            print(e)

    print("\nALL DONE")


if __name__ == "__main__":
    process_all_pdfs()


folders = [f for f in IMG_DIR.iterdir() if f.is_dir()]
print(len(folders))