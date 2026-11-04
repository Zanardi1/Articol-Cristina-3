"""Regenerate the publication comparison table from saved metrics.

Input: Results/multi_split_results.csv
Output: Results/paper_comparison.csv

Validates the expected algorithms, seeds, and metric ranges.
Computes mean and sample standard deviation across five holdouts.

Run: python report_results.py
Does not load or train facial-recognition models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_config import SEEDS

RESULTS_DIR = Path('Results')
INPUT_FILE = RESULTS_DIR / 'multi_split_results.csv'
OUTPUT_FILE = RESULTS_DIR / 'paper_comparison.csv'

ALGORITHMS = [
    'SKLearn',
    'FaceNet',
    'VGG-Face',
    'ResNet101',
    'Dlib'
]


def format_metric(mean, std):
    if pd.isna(mean) or pd.isna(std):
        return 'N/A'

    return f'{mean * 100:.2f} +/- {std * 100:.2f}'


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {
        'algorithm',
        'seed',
        'accuracy',
        'macro_f1',
        'coverage',
        'end_to_end_accuracy',
    }

    if not required.issubset(df.columns):
        raise ValueError('Missing required result columns')

    if len(df) != len(ALGORITHMS) * len(SEEDS):
        raise ValueError('Unexpected number of result rows')

    if set(df['algorithm']) != set(ALGORITHMS):
        raise ValueError('Unexpected algorithms')

    for algorithm in ALGORITHMS:
        subset = df[df['algorithm'] == algorithm]

        if len(subset) != len(SEEDS):
            raise ValueError(f'{algorithm}: Unexpected number of splits')

        if set(subset['seed']) != set(SEEDS):
            raise ValueError(f'{algorithm}: Missing or duplicate seeds')

    for metric in ('accuracy', 'macro_f1'):
        if df[metric].isna().any():
            raise ValueError(f'Missing {metric} values')

    for metric in ('accuracy', 'macro_f1', 'coverage', 'end_to_end_accuracy'):
        values = df[metric].dropna()
        if not values.between(0, 1).all():
            raise ValueError(f'Invalid {metric} values')

    dlib = df[df['algorithm'] == 'Dlib']

    if dlib[['coverage', 'end_to_end_accuracy']].isna().any().any():
        raise ValueError('Missing Dlib coverage metrics')

    if not np.allclose(
            dlib['end_to_end_accuracy'],
            dlib['accuracy'] * dlib['coverage'],
    ):
        raise ValueError('Inconsistent Dlib end-to-end accuracy')

    metrics = [
        'accuracy',
        'macro_f1',
        'coverage',
        'end_to_end_accuracy',
    ]

    summary = df.groupby('algorithm')[metrics].agg(['mean', 'std'])
    rows = []

    for algorithm in ALGORITHMS:
        row = {'Algorithm': algorithm}

        for metric, heading in [
            ('accuracy', 'Accuracy (%)'),
            ('macro_f1', 'Macro F1 (%)'),
            ('coverage', 'Detection coverage (%)'),
            ('end_to_end_accuracy', 'End-to-end accuracy (%)'),
        ]:
            row[heading] = format_metric(
                summary.loc[algorithm, (metric, 'mean')],
                summary.loc[algorithm, (metric, 'std')]
            )

        rows.append(row)

    paper_table = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paper_table.to_csv(OUTPUT_FILE, index=False)

    print('\n---Publication Comparison Table---')
    print(paper_table.to_string(index=False))
    print(f'\nSaved to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
