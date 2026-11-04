"""Historical one-time repair of SKLearn prediction indices.

Reassigns SKLearn original image indices in the saved predictions CSV
and verifies label alignment against the FaceNet records.

Run only when intentionally reproducing the historical repair.
WARNING: Modifies Results/predictions.csv in place.

Not part of normal execution or routine validation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

path = Path('Results/predictions.csv')
df = pd.read_csv(path)

seeds = (42, 123, 2024, 2025, 2026)

for seed in seeds:
    mask = ((df['algorithm'] == 'SKLearn') & (df['seed'] == seed))
    indices = np.sort(df.loc[mask, 'original_index'].to_numpy())
    assert len(indices) == 258
    assert len(np.unique(indices)) == 258
    df.loc[mask, 'original_index'] = indices

reference = df.loc[df['algorithm'] == 'FaceNet', ['seed', 'original_index', 'true_label']].rename(
    columns={'true_label': 'reference_label'})

sklearn = df.loc[df['algorithm'] == 'SKLearn'].merge(reference, on=['seed', 'original_index'], validate='one_to_one')

assert len(sklearn) == 1290
assert np.array_equal(sklearn['true_label'].to_numpy(), sklearn['reference_label'].to_numpy())

df.to_csv(path, index=False)
print('SKLearn prediction indices repaired and verified.')
