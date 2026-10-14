import os

import timm
import torch
from PIL import Image
from torchvision import transforms


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_vit_model(model_name="vit_base_patch16_224"):
    """Load a pretrained Vision Transformer model for feature extraction."""

    model = timm.create_model(model_name, pretrained=True)

    # Remove the classification head to extract visual features.
    model.head = torch.nn.Identity()
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=model.default_cfg["mean"],
            std=model.default_cfg["std"],
        ),
    ])

    print(f"ViT model loaded successfully: {model_name}")
    print(f"Using device: {device}")

    return model, preprocess, device


def extract_image_features(
    image_folder,
    model,
    preprocess,
    device,
):
    """Extract and store visual features for all supported images."""

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            f"Image folder was not found: {image_folder}"
        )

    features_dict = {}

    filenames = sorted(os.listdir(image_folder))

    for filename in filenames:

        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        image_path = os.path.join(image_folder, filename)

        if not os.path.isfile(image_path):
            continue

        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")

            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.inference_mode():
                feature = model(image_tensor)

            # Preserve each feature vector under its original filename.
            features_dict[filename] = feature.cpu().numpy()

        except Exception as error:
            print(f"Could not process {filename}: {error}")

    print(
        f"Feature extraction completed for "
        f"{len(features_dict)} images."
    )

    return features_dict


def save_vit_model(
    model,
    output_path="/content/vit_mamluk_bahri.pt",
):
    """Save the trained or pretrained model state dictionary."""

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    torch.save(model.state_dict(), output_path)

    print(f"Model saved to: {output_path}")

    return output_path
