from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = PROJECT_ROOT / "Dataset"
IMG_DIR = PROJECT_ROOT / "FolderSep"


def clean_name(name: str) -> str:
    return name.replace("(SustainabilityReports.com)", "").strip()


def create_pdf_folders():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    print(f"Tìm thấy {len(pdf_files)} file PDF")

    for pdf_file in pdf_files:
        folder_name = clean_name(pdf_file.stem)
        target_folder = IMG_DIR / folder_name

        target_folder.mkdir(parents=True, exist_ok=True)

        print(f"Đã tạo: {target_folder}")

    print("Hoàn tất.")


if __name__ == "__main__":
    create_pdf_folders()