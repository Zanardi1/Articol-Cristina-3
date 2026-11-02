import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dlib_alg import DlibClassification
from experiment_config import SEEDS, EXPECTED_CLASSES, EXPECTED_SAMPLES
from experiment_config import TEST_SIZE
from facenet_torch import FaceNetTorchClassification
from resnet import TensorFlowClassification
from sk import SKLearnClassification
from vgg_deepface import VGGClassification


def evaluate_splits(name, features, labels, original_indices=None):
    """Evaluate fixed embeddings using five stratified holdout splits."""

    labels = np.asarray(labels)
    features = np.asarray(features)

    if len(labels) != EXPECTED_SAMPLES:
        raise ValueError(
            f'{name}: unexpected dataset size'
        )

    if len(np.unique(labels)) != EXPECTED_CLASSES:
        raise ValueError(
            f'{name}: unexpected number of classes'
        )

    if original_indices is None:
        original_indices = np.arange(len(labels))
    else:
        original_indices = np.asarray(original_indices)

    if features.ndim != 2:
        raise ValueError(
            f'{name}: expected a feature 2D matrix'
        )
    if len(features) != len(original_indices):
        raise ValueError(
            f'{name}: features and indices have different lengths'
        )
    if len(np.unique(original_indices)) != len(original_indices):
        raise ValueError(
            f'{name}: duplicate original indices'
        )
    if np.any(original_indices < 0) or np.any(original_indices >= len(labels)):
        raise ValueError(
            f'{name}: original indices outside dataset bounds'
        )
    if name != 'Dlib' and len(original_indices) != len(labels):
        raise ValueError(
            f'{name}: unexpected missing samples'
        )

    results = []

    for seed in SEEDS:
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=labels
        )

        train_mask = np.isin(original_indices, train_idx)
        test_mask = np.isin(original_indices, test_idx)

        selected_train = original_indices[train_mask]
        selected_test = original_indices[test_mask]

        if np.intersect1d(selected_train, selected_test).size:
            raise ValueError(
                f'{name}: train/test overlap for seed {seed}'
            )
        if name != 'Dlib':
            if not np.array_equal(
                    np.sort(selected_train), np.sort(train_idx)
            ):
                raise ValueError(
                    f'{name}: training partition mismatch'
                )
            if not np.array_equal(
                    np.sort(selected_test), np.sort(test_idx)
            ):
                raise ValueError(
                    f'{name}: test partition mismatch'
                )
        else:
            if len(selected_train) + len(selected_test) != len(original_indices):
                raise ValueError(
                    f'{name}: detected samples missing from split'
                )

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
            assert 0 <= coverage <= 1
            assert 0 <= end_to_end <= coverage
            assert np.isclose(end_to_end, accuracy * coverage)

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

    if not np.array_equal(model.labels,model.y[model.embedding_indices]):
        raise ValueError(
            'Dlib embeddings and original labels are misaligned'
        )

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
        output_dir / "multi_split_results original.csv",
        index=False
    )

    summary.to_csv(
        output_dir / "multi_split_summary original.csv"
    )

    print("\nResults saved in the Results directory.")


if __name__ == "__main__":
    main()
