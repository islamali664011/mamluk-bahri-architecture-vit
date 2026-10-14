import os

from src.data_loader import prepare_image_folder
from src.vit_features import (
    load_vit_model,
    extract_image_features,
    save_vit_model,
)
from src.similarity_search import find_similar_image


# ==========================================
# Project configuration
# ==========================================

SOURCE_FOLDER = (
    "/content/drive/MyDrive/"
    "مساجد ومدارس العصر المملوكي البحري"
)

IMAGE_FOLDER = "/content/mamluk_bahri_images"

QUERY_IMAGE_PATH = "/content/images.jpeg"

MODEL_OUTPUT_PATH = "/content/vit_mamluk_bahri.pt"

TOP_K = 5


# ==========================================
# Main execution
# ==========================================

def main():

    # 1. Load images from Google Drive into Colab.
    image_folder = prepare_image_folder(
        source_folder=SOURCE_FOLDER,
        destination_folder=IMAGE_FOLDER,
    )

    # 2. Load the pretrained Vision Transformer model.
    model, preprocess, device = load_vit_model(
        model_name="vit_base_patch16_224"
    )

    # 3. Extract visual features from the image dataset.
    features_dict = extract_image_features(
        image_folder=image_folder,
        model=model,
        preprocess=preprocess,
        device=device,
    )

    if not features_dict:
        raise RuntimeError(
            "No image features were extracted. "
            "Please check the dataset folder."
        )

    # 4. Save the model state dictionary.
    save_vit_model(
        model=model,
        output_path=MODEL_OUTPUT_PATH,
    )

    # 5. Find and display the most similar images.
    if os.path.isfile(QUERY_IMAGE_PATH):

        find_similar_image(
            query_image_path=QUERY_IMAGE_PATH,
            features_dict=features_dict,
            model=model,
            preprocess=preprocess,
            device=device,
            image_folder=image_folder,
            top_k=TOP_K,
        )

    else:
        print(
            f"\nQuery image not found: {QUERY_IMAGE_PATH}"
        )
        print(
            "Upload your query image to Colab and update "
            "QUERY_IMAGE_PATH in main.py."
        )

    print("\nProject execution completed.")


if __name__ == "__main__":
    main()
