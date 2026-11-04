"""Audit FaceNet input normalization and its effect on classification.

Compares the implemented normalization with a reference convention,
including input differences, embedding differences, and predictions.

Uses the seed-42 train/test partition.
Output: Results/facenet_normalization_audit.json

Run: python audit_facenet_normalization.py
Loads the pretrained FaceNet model and performs feature extraction.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from experiment_config import TEST_SIZE
from facenet_torch import FaceNetTorchClassification

SEED = 42
BATCH_SIZE = 16


def extract_embeddings(model, images):
    embeddings = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE]
            output = model(batch)
            embeddings.append(output.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def evaluate(embeddings, labels, train_idx, test_idx):
    classifier = SVC(kernel="linear", probability=True)

    classifier.fit(
        embeddings[train_idx],
        labels[train_idx]
    )

    predictions = classifier.predict(embeddings[test_idx])

    return {
        "accuracy": float(
            accuracy_score(labels[test_idx], predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels[test_idx],
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "predictions": predictions
    }


def main():
    torch.set_num_threads(1)

    model = FaceNetTorchClassification()
    model.load_data()
    model.process_images()
    model.load_model()

    network = model.model.to(torch.device("cpu")).eval()

    # Existing preprocessing: (x - 0.5) / 0.5
    current_images = model.X_processed

    # Reference preprocessing: (x_255 - 127.5) / 128
    # The current tensor is already normalized:
    # x_255 = (current_images + 1) * 127.5
    reference_images = (
                               (current_images + 1.0) * 127.5 - 127.5
                       ) / 128.0

    input_difference = (
            current_images - reference_images
    ).abs()

    print("\n=== INPUT COMPARISON ===")
    print(
        "Maximum absolute difference:",
        input_difference.max().item()
    )
    print(
        "Mean absolute difference:",
        input_difference.mean().item()
    )

    print("\nExtracting current embeddings...")
    current_embeddings = extract_embeddings(
        network, current_images
    )

    print("Extracting reference embeddings...")
    reference_embeddings = extract_embeddings(
        network, reference_images
    )

    embedding_difference = np.abs(
        current_embeddings - reference_embeddings
    )

    print("\n=== EMBEDDING COMPARISON ===")
    print(
        "Maximum absolute difference:",
        embedding_difference.max()
    )
    print(
        "Mean absolute difference:",
        embedding_difference.mean()
    )

    labels = np.asarray(model.y)

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels
    )

    current_result = evaluate(
        current_embeddings, labels, train_idx, test_idx
    )

    reference_result = evaluate(
        reference_embeddings, labels, train_idx, test_idx
    )

    changed_predictions = int(np.sum(
        current_result["predictions"]
        != reference_result["predictions"]
    ))

    print("\n=== CLASSIFICATION COMPARISON ===")
    print("Seed:", SEED)
    print(
        "Current accuracy:",
        current_result["accuracy"]
    )
    print(
        "Reference accuracy:",
        reference_result["accuracy"]
    )
    print(
        "Current macro F1:",
        current_result["macro_f1"]
    )
    print(
        "Reference macro F1:",
        reference_result["macro_f1"]
    )
    print(
        "Changed predictions:",
        changed_predictions
    )

    results = {
        "seed": SEED,
        "input_max_abs_difference": float(
            input_difference.max().item()
        ),
        "input_mean_abs_difference": float(
            input_difference.mean().item()
        ),
        "embedding_max_abs_difference": float(
            embedding_difference.max()
        ),
        "embedding_mean_abs_difference": float(
            embedding_difference.mean()
        ),
        "current_accuracy": current_result["accuracy"],
        "reference_accuracy": reference_result["accuracy"],
        "current_macro_f1": current_result["macro_f1"],
        "reference_macro_f1": reference_result["macro_f1"],
        "changed_predictions": changed_predictions
    }

    output_dir = Path("Results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
            output_dir / "facenet_normalization_audit.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
