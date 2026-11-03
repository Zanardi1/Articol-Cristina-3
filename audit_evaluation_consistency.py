
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from experiment_config import (
    SEEDS, TEST_SIZE, MIN_FACES_PER_PERSON,
    IMAGE_RESIZE, EXPECTED_SAMPLES
)

RESULTS = Path("Results")
ALGORITHMS = {
    "SKLearn", "FaceNet", "VGG-Face",
    "ResNet101", "Dlib"
}


def main():
    data = fetch_lfw_people(
        data_home=str(Path.cwd()),
        min_faces_per_person=MIN_FACES_PER_PERSON,
        resize=IMAGE_RESIZE
    )
    labels = data.target

    predictions = pd.read_csv(
        RESULTS / "predictions.csv"
    )
    results = pd.read_csv(
        RESULTS / "multi_split_results.csv"
    )

    assert len(labels) == EXPECTED_SAMPLES
    assert len(results) == len(SEEDS) * len(ALGORITHMS)

    checks = 0

    for seed in SEEDS:
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=labels
        )

        assert not np.intersect1d(
            train_idx, test_idx
        ).size

        for algorithm in sorted(ALGORITHMS):
            rows = predictions[
                (predictions["seed"] == seed) &
                (predictions["algorithm"] == algorithm)
            ]

            assert len(rows) == len(test_idx), (
                algorithm, seed, "row count"
            )

            indices = rows["original_index"].to_numpy()

            assert len(np.unique(indices)) == len(indices)
            assert np.array_equal(
                np.sort(indices), np.sort(test_idx)
            ), (algorithm, seed, "test indices")

            detected = rows[
                rows["status"] == "detected"
            ]

            failed = rows[
                rows["status"] == "detection failed"
            ]

            if algorithm != "Dlib":
                assert len(failed) == 0

            actual_indices = detected[
                "original_index"
            ].to_numpy(dtype=int)

            y_true = detected[
                "true_label"
            ].to_numpy(dtype=int)

            y_pred = detected[
                "predicted_label"
            ].to_numpy(dtype=int)

            assert np.array_equal(
                y_true, labels[actual_indices]
            ), (algorithm, seed, "label alignment")

            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(
                y_true, y_pred,
                labels=np.arange(len(data.target_names)),
                average="macro",
                zero_division=0
            )

            summary = results[
                (results["seed"] == seed) &
                (results["algorithm"] == algorithm)
            ]

            assert len(summary) == 1

            assert np.isclose(
                accuracy, summary.iloc[0]["accuracy"]
            ), (algorithm, seed, "accuracy")

            assert np.isclose(
                macro_f1, summary.iloc[0]["macro_f1"]
            ), (algorithm, seed, "macro F1")

            if algorithm == "Dlib":
                coverage = len(detected) / len(test_idx)
                end_to_end = accuracy * coverage

                assert np.isclose(
                    coverage,
                    summary.iloc[0]["coverage"]
                )

                assert np.isclose(
                    end_to_end,
                    summary.iloc[0]["end_to_end_accuracy"]
                )

            checks += 1
            print(f"PASS: {algorithm}, seed={seed}")

    print(
        f"\nAUDIT PASSED: {checks} "
        "algorithm/seed evaluations verified."
    )


if __name__ == "__main__":
    main()
