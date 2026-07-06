from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HAO_DIR = PROJECT_ROOT / "DatasetIMG_Reduce"


def check_folders():

    folders = sorted(
        [f for f in HAO_DIR.iterdir() if f.is_dir()]
    )

    empty_folders = []

    total_images = 0

    for folder in folders:

        image_count = len(
            list(folder.glob("*.jpg"))
        )

        total_images += image_count

        if image_count == 0:
            empty_folders.append(folder.name)

        print(
            f"{folder.name}: "
            f"{image_count} images"
        )

    print("\n====================")
    print(f"Folders: {len(folders)}")
    print(f"Total images: {total_images}")
    print(f"Empty folders: {len(empty_folders)}")

    if empty_folders:

        print("\nFolders with 0 images:")

        for folder_name in empty_folders:
            print(folder_name)


if __name__ == "__main__":
    check_folders()