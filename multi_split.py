import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dlib_alg import DlibClassification
from facenet_torch import FaceNetTorchClassification
from resnet import TensorFlowClassification
from sk import SKLearnClassification
from vgg_deepface import VGGClassification

SEEDS = [42, 123, 2024, 2025, 2026]


def evaluate_splits(name, features, labels, original_indices=None):
    """Evaluate fixed embeddings using five stratified holdout splits."""

    labels = np.asarray(labels)
    features = np.asarray(features)

    if original_indices is None:
        original_indices = np.arange(len(labels))
    else:
        original_indices = np.asarray(original_indices)

    if len(features) != len(original_indices):
        raise ValueError("Features and indices have different lengths.")

    results = []

    for seed in SEEDS:
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=0.2,
            random_state=seed,
            stratify=labels
        )

        train_mask = np.isin(original_indices, train_idx)
        test_mask = np.isin(original_indices, test_idx)

        X_train = features[train_mask]
        X_test = features[test_mask]
        y_train = labels[original_indices[train_mask]]
        y_test = labels[original_indices[test_mask]]

        if name == "Dlib":
            classifier = KNeighborsClassifier()
        else:
            classifier = SVC(kernel="linear", probability=True)

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        row = {
            "algorithm": name,
            "seed": seed,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

        if name == "Dlib":
            coverage = len(y_test) / len(test_idx)
            end_to_end = accuracy * coverage

            row["coverage"] = coverage
            row["end_to_end_accuracy"] = end_to_end

            print(
                f"{name} | seed={seed} | "
                f"accuracy={accuracy:.4f} | "
                f"macro F1={macro_f1:.4f} | "
                f"coverage={coverage:.4f} | "
                f"end-to-end={end_to_end:.4f}"
            )
        else:
            print(
                f"{name} | seed={seed} | "
                f"accuracy={accuracy:.4f} | "
                f"macro F1={macro_f1:.4f}"
            )

        results.append(row)

    return results


def run_sklearn():
    print('\n---Scikit-learn PCA+SVM---')

    model = SKLearnClassification()
    results = model.evaluate_multiple_splits()

    for row in results:
        row['algorithm'] = 'SKLearn'

    del model
    gc.collect()

    return results


def run_facenet():
    print("\n=== FaceNet ===")

    model = FaceNetTorchClassification()
    model.load_data()
    model.process_images()
    model.load_model()
    model.get_embeddings()

    results = evaluate_splits(
        "FaceNet",
        model.X_embeddings,
        model.y
    )

    del model
    gc.collect()
    return results


def run_vgg():
    print("\n=== VGG-Face ===")

    model = VGGClassification()
    model.load_data()
    model.preprocess_images()
    model.encode_labels()

    # Uses the current preprocessing implementation in
    # vgg_deepface.py, including its chosen normalization.
    embeddings = model.extract_embeddings(model.X_rgb)

    results = evaluate_splits(
        "VGG-Face",
        embeddings,
        model.y_encoded
    )

    del embeddings, model
    gc.collect()
    return results


def run_resnet():
    print("\n=== ResNet101 ===")

    model = TensorFlowClassification()
    model.load_data()
    model.resize_images()
    model.define_model()

    # Extract features in batches to control memory usage.
    features = model.model.predict(
        model.X_resized,
        batch_size=16,
        verbose=1
    )

    features = features.reshape(len(features), -1)

    results = evaluate_splits(
        "ResNet101",
        features,
        model.y
    )

    del features, model
    gc.collect()
    return results


def run_dlib():
    print("\n=== Dlib ===")

    model = DlibClassification()
    model.load_data()
    model.create_shared_split(random_state=42)
    model.initialize_detectors()

    # Detect faces and extract embeddings only once.
    model.fill_face_embeddings_and_labels()

    results = evaluate_splits(
        "Dlib",
        model.face_embeddings,
        model.y,
        original_indices=model.embedding_indices
    )

    del model
    gc.collect()
    return results


def main():
    all_results = []

    # Run one algorithm at a time.
    all_results.extend(run_sklearn())
    all_results.extend(run_facenet())
    all_results.extend(run_vgg())
    all_results.extend(run_resnet())
    all_results.extend(run_dlib())

    df = pd.DataFrame(all_results)

    expected_algorithms = {
        'SKLearn',
        'FaceNet',
        'VGG-Face',
        'ResNet101',
        'Dlib'
    }

    assert set(df['algorithm']) == expected_algorithms
    assert len(df) == 25

    print("\n=== ALL INDIVIDUAL RESULTS ===")
    print(df.to_string(index=False))

    metrics = [
        "accuracy",
        "macro_f1",
        'coverage',
        'end_to_end_accuracy'
    ]

    summary = df.groupby("algorithm")[metrics].agg(
        ["mean", "std"]
    )

    print("\n=== FIVE-SPLIT SUMMARY ===")
    print(summary.to_string())

    # Preserve exact numerical results for the paper.

    output_dir = Path("Results")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / "multi_split_results.csv",
        index=False
    )

    summary.to_csv(
        output_dir / "multi_split_summary.csv"
    )

    print("\nResults saved in the Results directory.")


if __name__ == "__main__":
    main()
