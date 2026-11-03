from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

RESULTS_DIR = Path('Results')
MATRIX_DIR = RESULTS_DIR / 'confusion_matrices'

class_rows = []
prediction_rows = []


def record_evaluation(
        algorithm,
        seed,
        y_true,
        y_pred,
        target_names,
        test_indices,
        detected_indices=None
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    test_indices = np.asarray(test_indices)

    labels = np.arange(len(target_names))

    if detected_indices is None:
        detected_indices = test_indices
    else:
        detected_indices = np.asarray(detected_indices)

    if len(np.unique(detected_indices)) != len(detected_indices):
        raise ValueError("Duplicate detected indices")

    if not np.isin(detected_indices, test_indices).all():
        raise ValueError("Detected indices outside test partition")

    if not np.isin(y_true, labels).all():
        raise ValueError("Invalid true labels")

    if not np.isin(y_pred, labels).all():
        raise ValueError("Invalid predicted labels")

    report = classification_report(y_true, y_pred, labels=labels, target_names=list(target_names), output_dict=True,
                                   zero_division=0)

    for class_id, class_name in enumerate(target_names):
        metrics = report[str(class_name)]
        class_rows.append({
            'algorithm': algorithm,
            'seed': seed,
            'class_id': class_id,
            'class_name': str(class_name),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1-score'],
            'support': int(metrics['support']),
        })

    for index, true_label, predicted_label in zip(detected_indices, y_true, y_pred):
        prediction_rows.append({
            'algorithm': algorithm,
            'seed': seed,
            'original_index': int(index),
            'true_label': int(true_label),
            'predicted_label': int(predicted_label),
            'status': 'detected'
        })

    failed_indices = np.setdiff1d(test_indices, detected_indices)

    if algorithm != 'Dlib' and len(failed_indices):
        raise ValueError(f"{algorithm}: unexpected missing test predictions")

    for index in failed_indices:
        prediction_rows.append({
            'algorithm': algorithm,
            'seed': seed,
            'original_index': int(index),
            'true_label': np.nan,
            'predicted_label': np.nan,
            'status': 'detection failed'
        })

    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=target_names,
        xticks_rotation=90,
        cmap='Blues',
        colorbar=False
    )

    plt.title(f'{algorithm} - {seed}')
    plt.tight_layout()
    plt.savefig(
        MATRIX_DIR / f'{algorithm}_{seed}.png',
        dpi=200,
        bbox_inches='tight'
    )
    plt.close()


def save_reports():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(class_rows).to_csv(
        RESULTS_DIR / 'class_metrics.csv',
        index=False,
    )

    pd.DataFrame(prediction_rows).to_csv(
        RESULTS_DIR / 'predictions.csv',
        index=False,
    )

    print('Class-level reports saved')
