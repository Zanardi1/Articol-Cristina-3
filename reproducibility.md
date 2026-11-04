# Reproducibility Record

## 1. Purpose

This document records the software environment, pretrained-model dependencies, experimental controls, and remaining limitations associated with the LFW facial-recognition comparison.

The goal is to distinguish the parts of the experiment that can be reproduced directly from those that require additional environment or pretrained-weight verification.

## 2. Recorded Execution Environment

The environment snapshot was generated on November 3, 2026.

| Component | Recorded value |
|---|---|
| Python | 3.12.10 |
| Operating system | Windows 11 |
| Processor identifier | Intel64 Family 6 Model 142 Stepping 10 |
| Logical CPUs | 8 |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |
| pandas | 3.0.1 |
| scikit-learn | 1.8.0 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| PyTorch | 2.10.0 |
| torchvision | 0.25.0 |
| facenet-pytorch | 2.6.0 |
| DeepFace | 0.0.99 |
| Dlib | 20.0.0 |
| OpenCV | 4.13.0.92 |
| Matplotlib | 3.10.8 |

The complete dependency list is provided in `requirements.txt`.

The environment snapshot was captured separately from the model evaluation. It is not an immutable record of the environment at the original execution time.

GPU configuration, physical memory, exact processor model, and low-level backend details have not been fully recorded.

## 3. Dataset Configuration

Dataset loader: `sklearn.datasets.fetch_lfw_people`

Configuration:

- `min_faces_per_person=70`
- `resize=0.4`
- 1,288 selected images
- Seven identities
- Grayscale images, 50 × 37 pixels

The dataset is downloaded and cached by scikit-learn when necessary.

An exact dataset archive checksum has not been recorded.

## 4. Randomness and Data Partitioning

The shared train/test seeds are:

`42, 123, 2024, 2025, 2026`

Each partition uses:

- `test_size=0.2`
- Stratification by identity label
- 1,030 training images
- 258 test images

All algorithms use the same original dataset indices within a given seed.

The Scikit-learn PCA pipeline uses random state 42 for randomized PCA and hyperparameter search.

The fixed-feature pipelines fit a separate classifier for each partition.

Not all framework-level random-number generators, hardware execution settings, or library backends are explicitly controlled. Exact bit-for-bit reproduction is therefore not guaranteed.

## 5. Pretrained Model Dependencies

### FaceNet

Library: `facenet-pytorch`

Architecture: `InceptionResnetV1`

Pretrained configuration: `vggface2`

Input size: 160 × 160

Feature extraction is performed on the CPU.

The exact downloaded checkpoint checksum has not been recorded.

### VGG-Face

Library: `deepface`

Model identifier: `VGG-Face`

Normalization argument: `VGGFace`

Detection enforcement: `False`

Input size: 224 × 224

The exact checkpoint checksum has not been recorded.

DeepFace's internal representation and preprocessing behavior depend on the installed library version.

### ResNet101

Library: Keras

Architecture: `ResNet101`

Weights: `imagenet`

Classification head: excluded using `include_top=False`

Input size: 224 × 224

The model produces convolutional feature maps that are flattened before SVM classification.

The exact pretrained checkpoint checksum has not been recorded.

### Dlib

The pipeline uses:

- `dlib.get_frontal_face_detector()`
- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

The model files are obtained from the Dlib model distribution when not already available.

The implementation uses detection upsampling 0, 68-point facial landmarks, aligned face chips, and pretrained face descriptors.

SHA-256 hashes and file sizes for the FaceNet, VGG-Face, ResNet101, and Dlib pretrained model files were recorded during the November 2026 checkpoint audit. See Results/pretrained_model_audit.json. These hashes identify the locally inspected files; they do not independently establish the contents of the original pretraining datasets.

## 6. Classifier Configurations

| Pipeline | Classifier |
|---|---|
| SKLearn | RBF SVM, balanced class weights, randomized hyperparameter search |
| FaceNet | Linear SVM, probability estimation enabled |
| VGG-Face | Linear SVM, probability estimation enabled |
| ResNet101 | Linear SVM, probability estimation enabled |
| Dlib | KNN with scikit-learn default parameters |

The SKLearn baseline performs 10 randomized parameter trials with five-fold cross-validation inside the training partition.

The other classifiers use the configurations defined in `multi_split.py`.

## 7. Evaluation and Reporting

Each algorithm is evaluated on five shared stratified holdout partitions.

The primary metrics are accuracy and macro F1.

Additional outputs include class-specific precision, recall, F1, support, and confusion matrices.

For Dlib, recognition accuracy is conditional on successful face detection.

Detection coverage and end-to-end accuracy are reported separately.

The summary table uses arithmetic means and sample standard deviations across the five partitions.

The five holdouts are not statistically independent.

## 8. Reproduction Procedures

### Full Experimental Reproduction

Install dependencies and execute:

```bash
python main.py
```

This reruns feature extraction, classifier training, and evaluation.

The execution may download the dataset and pretrained weights.

The generated results may differ due to environment differences, model-file versions, or uncontrolled sources of randomness.

### Reporting Reproduction

Using an existing `Results/multi_split_results.csv`, execute:

```bash
python report_results.py
```

This regenerates the publication comparison table without rerunning the models.

### Environment Snapshot

Execute:

```bash
python environment_info.py
```

This records the currently installed package versions and system information.

## 9. Completed Audits and Outstanding Reproducibility Questions

### Completed audits

1. **FaceNet normalization:** Compared the implemented input normalization with a reference convention. Small input and embedding differences were observed, but the audited seed-42 classification predictions were unchanged. See `Results/facenet_normalization_audit.json`.
2. **VGG-Face detection:** Audited the configured OpenCV detector and documented fallback behavior. See `Results/vgg_detection_audit.json`.
3. **Pretrained checkpoint inventory:** Recorded local model-file sizes and SHA-256 hashes. See `Results/pretrained_model_audit.json`.
4. **Evaluation consistency:** Verified the saved predictions, shared test partitions, label alignment, and reported metrics for all 25 algorithm/seed evaluations. See `audit_evaluation_consistency.py`.

### Outstanding questions

1. Determine whether the face-specific pretrained models' training datasets overlap with the evaluated LFW identities or images. This investigation was deliberately deferred.
2. Record an exact checksum of the LFW dataset archive or processed dataset.
3. Establish the original download provenance of the pretrained model files beyond their locally recorded hashes.
4. Record additional hardware and execution-backend details if computational benchmarking becomes part of the study.
5. Verify portability of the dependency environment through a clean installation on another machine.

These limitations should be acknowledged when making strong claims about reproducibility or generalization.

## 10. Interpretation

The recorded experimental results describe five implemented facial-recognition classification pipelines evaluated on a restricted LFW subset.

They are not evidence that one underlying neural-network architecture is universally superior to another.

The comparison is influenced by pretrained representations, preprocessing, classifier choices, and the handling of face-detection failures.

The repository preserves the experimental code, numerical results, and methodological documentation to support independent review and further replication.