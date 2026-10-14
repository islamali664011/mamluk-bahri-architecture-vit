# mamluk-bahri-architecture-vit
# ViT-Based Visual Similarity in Bahri Mamluk Religious Architecture

# ViT-Based Visual Similarity in Bahri Mamluk Religious Architecture

## Project Overview

This project explores visual similarity in Bahri Mamluk religious architecture using a pretrained Vision Transformer (ViT) model and deep visual feature extraction.

The dataset consists of 866 images representing mosques, madrasas, khanqahs, and other religious buildings and architectural elements associated with Bahri Mamluk architecture.

The project aims to support comparative research into Islamic architectural heritage by retrieving visually similar images based on their extracted visual features.

## Research Objectives

- Explore visual similarities among images of Bahri Mamluk religious architecture.
- Retrieve the five most visually similar images for a given query image.
- Examine visual relationships among different monuments and architectural elements.
- Investigate the potential of computer vision for comparative architectural research.
- Support exploratory analysis of architectural appearance, decorative patterns, and visual composition.

## Methodology

The project follows these steps:

1. Access the image dataset stored in Google Drive.
2. Load a pretrained Vision Transformer model, ViT-Base/16.
3. Resize images to 224 × 224 pixels and apply the model's normalization settings.
4. Extract deep visual features from each image.
5. Calculate cosine similarity between a query image and the reference image features.
6. Rank images by similarity score.
7. Display the five highest-ranking images with their scores and original filenames.
8. Save the model state dictionary for subsequent use.

## Project Structure

```text
mamluk-bahri-architecture-vit/
│
├── README.md
├── requirements.txt
├── main.py
├── .gitignore
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── vit_features.py
    └── similarity_search.py
```

### Main Components

- `main.py`: Coordinates the complete workflow.
- `src/data_loader.py`: Connects to Google Drive and copies the reference images into the Colab environment.
- `src/vit_features.py`: Loads the pretrained ViT model, extracts image features, and saves the model weights.
- `src/similarity_search.py`: Calculates cosine similarity and displays the most similar images with their original filenames.
- `requirements.txt`: Lists the Python dependencies.
- `.gitignore`: Excludes specified image files, model weights, and temporary files from Git tracking.

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
- arabic-reshaper
- python-bidi
- Google Colab
- Google Drive

## Dataset

The dataset contains 866 images of Bahri Mamluk religious architecture.

The images are stored in a Google Drive folder named:

`مساجد ومدارس العصر المملوكي البحري`

The dataset is not included in this repository. Users must have access to the images in Google Drive and update the source path in `main.py` if their folder location differs.

## Installation and Execution

The project is designed to run in Google Colab.

### 1. Clone the Repository

Open a new Google Colab notebook and execute:

```python
!git clone https://github.com/islamali664011/mamluk-bahri-architecture-vit.git
%cd mamluk-bahri-architecture-vit
```

### 2. Install Dependencies

```python
!pip install -r requirements.txt
```

### 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ensure that the dataset is available at the source path configured in `main.py`.

### 4. Configure the Query Image

In `main.py`, update `QUERY_IMAGE_PATH` to the full path of the image you want to compare.

Example:

```python
QUERY_IMAGE_PATH = "/content/images.jpeg"
```

Upload the query image to the specified location in the Colab environment before running the project.

### 5. Run the Project

```python
%run main.py
```

The workflow will copy the reference dataset, load the pretrained model, extract image features, calculate similarity scores, and display the results.

## Output

For each query image, the project displays:

- The original query image.
- The five highest-ranking reference images.
- A cosine similarity score for each result.
- The original filename beneath each displayed image.

Arabic filenames are formatted for right-to-left display.

The model state dictionary is saved to:

`/content/vit_mamluk_bahri.pt`

This file is saved in the Colab environment and is not automatically uploaded to GitHub.

## Limitations and Interpretation

This project uses a pretrained Vision Transformer for visual feature extraction. It does not independently identify historical monuments, establish construction dates, or verify architectural styles.

The displayed similarity percentage is a rescaled cosine similarity score, not a calibrated probability that two images represent the same monument. Results depend on the model's learned visual representations and should be interpreted as exploratory comparisons rather than definitive architectural or historical conclusions.

The workflow has not been independently validated as a specialist model for identifying Islamic architectural styles.

## Research Context

This project investigates the application of computer vision and artificial intelligence to the comparative study of Islamic architectural heritage, with an initial focus on Bahri Mamluk religious architecture.

It provides an exploratory computational workflow that may support further research into visual relationships among religious buildings, architectural elements, and related structures across different Islamic architectural contexts.
