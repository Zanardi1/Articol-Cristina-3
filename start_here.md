# START HERE — Facial Recognition Research Project

**Project:** Comparative evaluation of five facial-recognition pipelines on LFW  
**Last documentation review:** November 2026  
**Status:** Experimental evaluation completed; manuscript preparation is the next phase.

## 1. A message to future me

This repository contains the Python implementations, experimental results, and methodological audits for a scientific article comparing five facial-recognition pipelines.

The project was developed and validated during October–November 2026.

If you are returning after several months or years, **do not begin by rerunning every script**. The completed results are already stored in `Results/`.

First, read this document, inspect the existing results, and decide whether you actually need to execute anything.

The project is a research experiment, not a production facial-recognition application.

## 2. What the project does

The experiment compares five pipelines:

| Pipeline | Representation | Classifier |
|---|---|---|
| SKLearn | StandardScaler + PCA | RBF SVM |
| FaceNet | Pretrained InceptionResnetV1 | Linear SVM |
| VGG-Face | Pretrained DeepFace VGG-Face | Linear SVM |
| ResNet101 | ImageNet-pretrained ResNet101 | Linear SVM |
| Dlib | Face detector and pretrained descriptors | KNN |

Dataset: Labeled Faces in the Wild (LFW).

Configuration:

- 1,288 images
- Seven identities
- 1,030 training and 258 test images per split
- Five stratified holdout seeds: 42, 123, 2024, 2025, 2026
- 25 algorithm/seed evaluations

All pipelines use the same original dataset partitions within each seed.

The main evaluation metrics are accuracy and macro F1. Dlib additionally reports detection coverage and end-to-end accuracy.

## 3. How the program works

The entry point is:

`main.py`

This file imports and calls `main()` from `multi_split.py`.

The actual experiment orchestration is performed by `multi_split.py`.

Execution order:

1. SKLearn
2. FaceNet
3. VGG-Face
4. ResNet101
5. Dlib

For each algorithm, the program prepares the necessary image representation or embeddings, trains classifiers on the shared training partitions, and evaluates them on the corresponding test partitions.

`multi_split.py` collects the numerical results and writes the main result files.

`evaluation_reporting.py` produces prediction-level records, class-level metrics, and confusion matrices.

**Important:** Feature extraction for the pretrained pipelines is performed before the five train/test evaluations. The pretrained feature extractors are not fine-tuned on the LFW partitions.

## 4. Python script inventory

### A. Main execution and configuration

| File | Purpose |
|---|---|
| `main.py` | Main entry point; calls `multi_split.main()` |
| `multi_split.py` | Orchestrates the five pipelines, shared partitions, evaluation, and result saving |
| `experiment_config.py` | Defines dataset parameters, random seeds, test size, and expected dataset dimensions |
| `base.py` | Shared dataset-loading, splitting, training, and reporting functionality |

### B. Algorithm implementations

| File | Purpose |
|---|---|
| `sk.py` | PCA + SVM baseline, including hyperparameter search |
| `facenet_torch.py` | FaceNet image preprocessing and embedding extraction |
| `vgg_deepface.py` | VGG-Face preprocessing and DeepFace embedding extraction |
| `resnet.py` | ResNet101 preprocessing, pretrained model, and feature extraction |
| `dlib_alg.py` | Dlib face detection, landmarks, descriptor extraction, and detection-failure tracking |

These modules contain the algorithm implementations. They are not the recommended entry points for reproducing the complete five-algorithm experiment.

### C. Reporting and environment utilities

| File | Purpose |
|---|---|
| `evaluation_reporting.py` | Records individual predictions, per-class metrics, and confusion matrices |
| `report_results.py` | Regenerates the publication comparison table from existing results |
| `environment_info.py` | Records Python, operating system, and package-version information |

### D. Scientific audit scripts

| File | Purpose |
|---|---|
| `audit_deepface.py` | Inspects the installed DeepFace API and relevant preprocessing source code |
| `audit_facenet_normalization.py` | Compares FaceNet input normalization variants and their effect on embeddings and predictions |
| `audit_vgg_detection.py` | Tests VGG-Face's configured face detector and records detection failures |
| `audit_vgg_detection_debug.py` | Diagnoses DeepFace detection behavior on selected images |
| `audit_pretrained_models.py` | Locates pretrained model files and records sizes and SHA-256 hashes |
| `audit_evaluation_consistency.py` | Independently verifies saved prediction indices, labels, partitions, and metrics |

These scripts were created during scientific validation.

**They are not part of the normal experiment execution sequence.** Run an audit only when investigating its corresponding methodological question.

### E. Historical repair utility

| File | Purpose |
|---|---|
| `repair_predictions.py` | Historical one-time repair of SKLearn prediction indices in `Results/predictions.csv` |

**Do not run `repair_predictions.py` as part of the normal workflow.**

It modifies an existing results file in place. The repaired predictions subsequently passed the independent evaluation-consistency audit.

Retain the script as a record of the earlier reporting correction.

## 5. Important files that are not Python scripts

| File | Purpose |
|---|---|
| `readme.md` | Project overview, installation, and basic execution |
| `methodology.md` | Detailed scientific methodology |
| `reproducibility.md` | Environment and reproducibility record |
| `requirements.txt` | Python dependencies |
| `dlib_face_recognition_resnet_model_v1.dat` | Dlib pretrained face-recognition model |
| `shape_predictor_68_face_landmarks.dat` | Dlib facial landmark model |
| `deepface_audit.txt` | Historical DeepFace inspection output |
| `Article discussion.txt` | Draft discussion text for the scientific article |
| `Article conclusion.txt` | Draft conclusion text for the scientific article |
| `config.rc` | Existing repository configuration file; not part of the primary experiment entry point |

The repository also contains `.idea/` configuration files associated with the development environment.

## 6. Results directory

The `Results/` directory contains the saved experimental outputs.

| File | Meaning |
|---|---|
| `multi_split_results.csv` | Primary numerical results for all 25 evaluations |
| `multi_split_summary.csv` | Aggregated results across the five seeds |
| `paper_comparison.csv` | Formatted comparison table for publication |
| `predictions.csv` | Individual test predictions, original image indices, and detection status |
| `class_metrics.csv` | Per-class precision, recall, F1, and support |
| `confusion_matrices/` | Confusion matrices for individual evaluations |
| `environment_info.json` | Recorded software environment |
| `facenet_normalization_audit.json` | FaceNet preprocessing audit |
| `vgg_detection_audit.json` | VGG-Face detection audit |
| `pretrained_model_audit.json` | Pretrained checkpoint hashes and file metadata |

The saved results represent the completed experimental work.

Do not overwrite them casually.

## 7. Common operations

### A. Inspect the existing results

Open:

`Results/paper_comparison.csv`

For detailed results, inspect:

`Results/multi_split_results.csv`

No Python execution is required.

### B. Regenerate the publication comparison table

```bash
python report_results.py
```

Requires the existing `Results/multi_split_results.csv`.

Rewrites `Results/paper_comparison.csv`.

Does not retrain any models.

### C. Verify the saved evaluation results

```bash
python audit_evaluation_consistency.py
```

Checks prediction records against the original LFW labels, expected test partitions, and reported metrics.

Requires the saved results and access to the LFW dataset.

Does not retrain the recognition models.

The final audit in November 2026 passed all 25 algorithm/seed evaluations.

### D. Record the current software environment

```bash
python environment_info.py
```

Rewrites `Results/environment_info.json` using the current environment.

This is not necessarily the same environment as the original experiment.

### E. Rerun the complete experiment

```bash
python main.py
```

**Warning: This is the expensive operation.**

It runs all five pipelines across five stratified partitions, performs feature extraction, trains classifiers, and regenerates the primary results.

It may require downloading pretrained weights and the LFW dataset.

It may take considerable time and consume substantial memory.

It can overwrite saved experimental outputs, including numerical results, prediction records, and confusion matrices.

Before running it, preserve the existing results or work on a separate branch or copy.

## 8. Scientific decisions worth remembering

### Dlib accuracy is conditional

Dlib achieved 100% classification accuracy on successfully detected faces in the recorded experiments.

However, some images failed detection.

Mean detection coverage and end-to-end accuracy were approximately 92.48%.

Never present Dlib's 100% conditional accuracy as its overall end-to-end performance.

### VGG-Face uses detection fallback

The VGG-Face implementation uses DeepFace with `enforce_detection=False`.

A dedicated audit found that the configured OpenCV detector did not detect faces in the evaluated preprocessed images.

Fallback processing allowed feature extraction to continue.

Consequently, the reported VGG-Face accuracy must not be interpreted as evidence of successful face detection.

### FaceNet normalization was audited

The implemented FaceNet normalization was compared with a reference normalization convention.

The differences were small, and no test predictions changed in the audited seed-42 experiment.

### Train/test consistency was audited

All 25 saved evaluations passed the final consistency audit.

The audit verified shared test partitions, original image-to-label alignment, recalculated accuracy, recalculated macro F1, and Dlib coverage metrics.

### Pretraining overlap remains unresolved

An audit of pretrained checkpoint files recorded model-file hashes.

However, the investigation into possible identity or image overlap between external pretraining datasets and LFW was deferred.

Do not claim that such overlap has been ruled out.

### This is a closed-set classification experiment

The dataset contains seven known identities.

The experiment does not establish performance on unseen identities, open-set recognition, face verification, or real-world deployment conditions.

The five approaches are complete pipelines with different preprocessing and training histories, not an isolated comparison of architectures.

## 9. If something breaks in the future

First determine which part is failing:

- Dataset loading: inspect `base.py` and `experiment_config.py`.
- Shared partitioning or evaluation: inspect `multi_split.py`.
- SKLearn training: inspect `sk.py`.
- FaceNet processing: inspect `facenet_torch.py` and its normalization audit.
- VGG-Face processing: inspect `vgg_deepface.py` and its detection audits.
- ResNet101 processing: inspect `resnet.py`.
- Dlib detection: inspect `dlib_alg.py`.
- Incorrect saved metrics: inspect `evaluation_reporting.py`, `report_results.py`, and `audit_evaluation_consistency.py`.
- Missing model files: inspect `audit_pretrained_models.py` and `reproducibility.md`.

Check the dependency versions before assuming that an algorithm has become incorrect. External library APIs and model-download locations can change.

## 10. Recommended reading order

When returning to the repository:

1. `START_HERE.md` — Recover the workflow.
2. `readme.md` — Review setup and execution.
3. `methodology.md` — Recover the scientific reasoning.
4. `reproducibility.md` — Review environment and known limitations.
5. `multi_split.py` — Understand orchestration.
6. Individual algorithm scripts — Only when necessary.
7. Audit scripts — Only when investigating specific issues.

## 11. Project preservation policy

The experiment was considered complete in November 2026.

Future changes should be purposeful and documented.

Before modifying the experimental implementation:

- Preserve the existing results.
- Record the reason for the change.
- Avoid silently changing the published experimental configuration.
- Rerun the relevant validation checks after substantive changes.
- Keep historical results distinguishable from newly generated results.

**Final reminder:** You do not need to understand every script to use this repository. Start with the scientific question, identify the relevant part of the workflow, and only then inspect the implementation.
