# Facial Recognition Pipeline Comparison on LFW

## Overview

This repository contains a comparative experimental study of five facial-recognition classification pipelines using the Labeled Faces in the Wild (LFW) dataset.

The study evaluates closed-set identity classification over seven individuals using five shared stratified train/test partitions.

The five pipelines are:

| Pipeline | Feature extraction | Classifier |
|---|---|---|
| SKLearn | StandardScaler + PCA | RBF SVM |
| FaceNet | InceptionResnetV1, pretrained on VGGFace2 | Linear SVM |
| VGG-Face | DeepFace VGG-Face embeddings | Linear SVM |
| ResNet101 | ImageNet-pretrained ResNet101 features | Linear SVM |
| Dlib | Face detection, alignment, pretrained face descriptors | KNN |

This study compares the implemented pipelines as complete systems, rather than isolating individual model architectures.

## Dataset and Evaluation

The LFW dataset is loaded through `sklearn.datasets.fetch_lfw_people` with `min_faces_per_person=70` and `resize=0.4`.

The resulting dataset contains 1,288 grayscale images belonging to seven identities.

Five stratified 80/20 train/test partitions are generated using random seeds 42, 123, 2024, 2025, and 2026. Each partition contains 1,030 training images and 258 test images.

Evaluation metrics include classification accuracy, macro F1, per-class precision, recall and F1, and confusion matrices.

For Dlib, detection coverage and end-to-end accuracy are also reported because some images fail face detection.

## Requirements

The recorded experimental environment uses Python 3.12.10 on Windows 11.

Python dependencies are listed in `requirements.txt`. Some packages, particularly Dlib and the deep-learning frameworks, may require platform-specific installation prerequisites.

Pretrained weights are required for FaceNet, VGG-Face, ResNet101, and Dlib. Some weights are downloaded automatically by their respective libraries.

## Installation

Clone the repository:

```bash
git clone https://github.com/Zanardi1/Articol-Cristina-3.git
cd Articol-Cristina-3
```

Create and activate a virtual environment on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The dependency file records the packages used in the original environment. Installation on a different machine is not guaranteed to succeed without adjustments.

## Running the Experiments

Run all five pipelines across the five shared partitions:

```bash
python main.py
```

This runs the complete experimental evaluation and generates the associated CSV reports and confusion matrices.

The experiment may take considerable time and requires sufficient memory for pretrained models and extracted feature matrices.

The LFW dataset and pretrained model weights may need to be downloaded during the initial execution. Network access may therefore be required.

## Regenerating the Publication Table

If `Results/multi_split_results.csv` already exists, the publication comparison table can be regenerated without running the models:

```bash
python report_results.py
```

This produces `Results/paper_comparison.csv`.

## Recording the Environment

Run:

```bash
python environment_info.py
```

This records the current Python environment and selected package versions in `Results/environment_info.json`.

This command does not execute the facial-recognition experiment.

## Results

The `Results` directory contains:

| Output | Description |
|---|---|
| `multi_split_results.csv` | Results for each pipeline and random seed |
| `multi_split_summary.csv` | Aggregated metrics |
| `paper_comparison.csv` | Publication-oriented results |
| `class_metrics.csv` | Per-class classification metrics |
| `predictions.csv` | Prediction-level records and detection status |
| `confusion_matrices/` | Confusion-matrix images |
| `environment_info.json` | Environment snapshot |

The reported results are descriptive statistics across five overlapping stratified holdout partitions.

Dlib's recognition accuracy is conditional on successful face detection. Its end-to-end accuracy includes detection failures.

## Documentation

See [methodology.md](methodology.md) for the detailed experimental methodology.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for software environment details, pretrained model dependencies, and reproducibility limitations.

## Scope and Limitations

The experiment evaluates closed-set classification over seven LFW identities. It does not establish performance for open-set recognition, unknown identities, or deployment scenarios.

The pretrained models may have been trained on datasets containing identities or images overlapping with the evaluation dataset. Such overlap has not been independently excluded.

The five pipelines differ in preprocessing, representation learning, and classifier configuration. Their results should therefore be interpreted as pipeline-level comparisons.