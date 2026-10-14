import os
import shutil


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


def prepare_image_folder(
    source_folder="/content/drive/MyDrive/مساجد ومدارس العصر المملوكي البحري",
    destination_folder="/content/mamluk_bahri_images",
):
    """Copy architectural images from Google Drive to Colab."""

    if not os.path.isdir("/content/drive/MyDrive"):
        try:
            from google.colab import drive
        except ImportError as exc:
            raise RuntimeError(
                "This function must be run in Google Colab "
                "to access Google Drive."
            ) from exc

        drive.mount("/content/drive")

    if not os.path.isdir(source_folder):
        raise FileNotFoundError(
            f"Image source folder was not found: {source_folder}"
        )

    os.makedirs(destination_folder, exist_ok=True)

    copied_count = 0

    for filename in sorted(os.listdir(source_folder)):
        source_path = os.path.join(source_folder, filename)

        if (
            os.path.isfile(source_path)
            and filename.lower().endswith(SUPPORTED_EXTENSIONS)
        ):
            destination_path = os.path.join(
                destination_folder, filename
            )

            shutil.copy2(source_path, destination_path)
            copied_count += 1

    print(f"Copied {copied_count} images to Colab.")

    return destination_folder
