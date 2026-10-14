# mamluk-bahri-architecture-vit
# ViT-Based Visual Similarity in Bahri Mamluk Religious Architecture

## Project Overview

This project explores visual similarity in Bahri Mamluk religious architecture using a Vision Transformer (ViT) model and deep visual feature extraction.

The dataset consists of 866 images representing mosques, madrasas, khanqahs, and other religious architectural structures and their architectural elements.

The project aims to explore visual relationships between architectural images based on their extracted visual features, including similarities in architectural appearance, patterns, colors, and proportions.

## Objectives

- Explore visual similarities between images of Bahri Mamluk religious architecture.
- Retrieve the five most visually similar images for a given query image.
- Compare architectural images representing different monuments and architectural elements.
- Support exploratory research into Islamic architectural heritage using computer vision.

## Methodology

1. Load a pretrained Vision Transformer (ViT-Base/16) model.
2. Preprocess images by resizing them to 224 × 224 pixels and applying the model's normalization settings.
3. Extract deep visual features from each image.
4. Calculate cosine similarity between a query image and the images in the dataset.
5. Display the five most similar images alongside their similarity scores and original filenames.

## Technologies

- Python
- PyTorch
- Torchvision
- timm
- Vision Transformer (ViT)
- Scikit-learn
- NumPy
- Matplotlib
- Pillow
- Google Colab
- Google Drive

## Dataset

The dataset contains 866 images of Bahri Mamluk religious architecture.

The images are stored in Google Drive and are not included in this repository.

## Output

For each query image, the workflow displays the five most similar images, their cosine similarity scores, and their original filenames.

## Limitations

This project uses a pretrained model for visual feature extraction. It does not independently identify monuments, date buildings, or verify architectural styles. Similarity scores indicate relationships between extracted visual features and should be interpreted as exploratory results rather than definitive historical or architectural conclusions.

## Research Context

This project explores the application of computer vision and artificial intelligence to the comparative study of Islamic architectural heritage, with an initial focus on Bahri Mamluk religious architecture.
