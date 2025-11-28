# FinCognition 9: Deep Learning for Cetacean Classification

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-ee4c2c.svg)

> **A Deep Learning Project for Species Classification using Dorsal Fins**
>
> *Deep Learning Course, Fall 2022*
>
> **Authors:**
> *   Lorenzo Clementi
> *   Elena Muià

## 📖 Introduction

The classification of cetacean species poses a considerable challenge due to their vast morphological and ecological diversity. **FinCognition** aims to categorize **30 species** of whales and dolphins by examining the dorsal fin as it protrudes out of the water.

Akin to the human fingerprint, the dorsal fin of cetaceans is unique and can be used to identify individual cetacean species. This project leverages Deep Learning techniques to automate this identification process, aiding in marine biology research and conservation efforts.

## 📊 Dataset

We utilized the **HappyWhale dataset** from Kaggle to train our models. The dataset provides:
*   Images of dorsal fins in the sea.
*   Cetacean species labels.
*   Individual IDs.

### Challenges
The main hurdle was the dataset’s **highly imbalanced nature**:
*   Most frequent class: **7,593** samples
*   Least frequent class: **14** samples

## 🧠 Methodology

We implemented and compared multiple models to monitor progress. Our best performing approach involves a **Convolutional Neural Network (CNN)** developed entirely from scratch.

### Key Features
*   **Preprocessing:** Image resizing, normalization, and handling of grayscale/RGB formats.
*   **Architecture:** Custom CNN designed for feature extraction from dorsal fin patterns.
*   **Training:** Optimized using PyTorch with techniques to handle class imbalance (e.g., WeightedRandomSampler).

## 🏆 Results

Despite the significant class imbalance, our custom algorithm achieved satisfactory results, confirming the hypothesis that individual cetacean species can be discerned via dorsal fin analysis.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **90%** |
| **Test Loss** | **0.375** |
| **F1 Score (Micro)** | **90%** |
| **F1 Score (Macro)** | **80%** |

## 📂 Project Organization

```
├── data
│   ├── dataset_refs   <- Data from third party sources.
│   └── test_data      <- Test data.
├── models             <- Trained model checkpoints.
├── notebooks          <- Jupyter notebooks (main.ipynb contains the full analysis).
├── references         <- Research Papers, manuals, and explanatory materials.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures.
├── src                <- Source code for use in this project.
│   ├── data           <- Scripts to download and generate data.
│   ├── models         <- Scripts to train models and inference.
│   ├── preprocessing  <- Scripts to turn raw data into features.
│   └── visualization  <- Scripts to create visualizations.
└── requirements.txt   <- The requirements file for reproducing the analysis environment.
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Pandas, NumPy, Matplotlib, Seaborn

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lorecleme/finCogNitioN.git
    cd finCogNitioN
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the package in editable mode:**
    ```bash
    pip install -e .
    ```

## 💻 Usage

The core analysis and model training steps are documented in `notebooks/main.ipynb`.

To run the training scripts directly (after configuring paths):
```bash
python src/models/train_model.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
