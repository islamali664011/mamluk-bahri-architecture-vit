import os
import textwrap

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import torch

from bidi.algorithm import get_display
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


def arabic_display(text):
    """Format Arabic text for correct visual display."""
    return get_display(arabic_reshaper.reshape(str(text)))


def find_similar_image(
    query_image_path,
    features_dict,
    model,
    preprocess,
    device,
    image_folder,
    top_k=5,
):
    """
    Find and display the most visually similar architectural images.

    Args:
        query_image_path: Path to the query image.
        features_dict: Dictionary containing extracted image features.
        model: Loaded ViT feature extraction model.
        preprocess: Image preprocessing pipeline.
        device: CPU or CUDA device.
        image_folder: Folder containing the reference images.
        top_k: Number of similar images to display.
    """

    if not os.path.isfile(query_image_path):
        raise FileNotFoundError(
            f"Query image was not found: {query_image_path}"
        )

    if not features_dict:
        raise ValueError("No image features are available for comparison.")

    # Load and preprocess the query image.
    with Image.open(query_image_path) as source_image:
        img = source_image.convert("RGB")

    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Extract the query image features using ViT.
    model.eval()

    with torch.inference_mode():
        query_feature = model(img_tensor).cpu().numpy()

    similarity_scores = []

    # Compare the query image against every reference image.
    for filename, feature in features_dict.items():

        score = cosine_similarity(
            query_feature,
            feature
        )[0][0]

        similarity_scores.append((filename, float(score)))

    # Sort from highest to lowest cosine similarity.
    similarity_scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    results = similarity_scores[:max(1, top_k)]

    print(f"\nTop {len(results)} visually similar images:\n")

    for rank, (filename, score) in enumerate(results, start=1):
        print(
            f"{rank}. {filename} | "
            f"Cosine similarity: {score * 100:.2f}%"
        )

    # Create the comparison figure.
    fig, axes = plt.subplots(
        1,
        len(results) + 1,
        figsize=(5 * (len(results) + 1), 7)
    )

    axes = np.atleast_1d(axes)

    # Display the query image.
    axes[0].imshow(img)
    axes[0].set_title("Query Image", fontsize=12)
    axes[0].axis("off")

    query_filename = os.path.basename(query_image_path)

    axes[0].text(
        0.5,
        -0.10,
        arabic_display(textwrap.fill(query_filename, width=25)),
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=10,
        clip_on=False,
    )

    # Display the most similar reference images.
    for index, (filename, score) in enumerate(results, start=1):

        image_path = os.path.join(image_folder, filename)

        with Image.open(image_path) as source_image:
            similar_image = source_image.convert("RGB")

        axes[index].imshow(similar_image)

        axes[index].set_title(
            f"Similarity: {score * 100:.2f}%",
            fontsize=11
        )

        axes[index].axis("off")

        # Display the original filename below each image.
        axes[index].text(
            0.5,
            -0.10,
            arabic_display(textwrap.fill(filename, width=25)),
            transform=axes[index].transAxes,
            ha="center",
            va="top",
            fontsize=10,
            clip_on=False,
        )

    plt.subplots_adjust(
        bottom=0.30,
        wspace=0.20
    )

    plt.show()

    return results
