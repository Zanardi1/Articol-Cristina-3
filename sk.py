import numpy as np
from experiment_config import SEEDS
from scipy.stats import loguniform
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from base import Base


class SKLearnClassification(Base):

    def __init__(self):
        """Initializes the SKLearnClassification object with default values."""
        super().__init__()
        self.n_components = 150
        self.param_grid = {
            'C': loguniform(1e3, 1e5),
            'gamma': loguniform(1e-4, 1e-1)
        }

    def execute(self):
        super().execute()
        self.split_data(X=self.X, y=self.y)
        self.train_model(self.build_search_pipeline())
        self.evaluate_model()
        print(f'Best parameters: {self.model.best_params_}')
        self.print_results()

    def evaluate_multiple_splits(self):
        results = []

        self.load_data()
        self.print_load_results()

        for seed in SEEDS:
            print(f'\nEvaluating split with random_state={seed}')

            self.create_shared_split(random_state=seed)
            self.split_data(X=self.X, y=self.y)

            self.train_model(self.build_search_pipeline())
            self.evaluate_model()

            macro_f1 = f1_score(
                self.y_test,
                self.y_pred,
                average='macro',
                zero_division=0
            )

            results.append({
                'seed': seed,
                'accuracy': self.accuracy,
                'macro_f1': macro_f1
            })

            print(f'Seed: {seed}\n')
            print(f'Accuracy: {self.accuracy:.4f}\n')
            print(f'Macro F1: {macro_f1:.4f}\n')
            print(f'Best parameters: {self.model.best_params_}')

        accuracies = np.array([result['accuracy'] for result in results])
        f1_scores = np.array([result['macro_f1'] for result in results])

        print('\n=== FIVE-SPLIT SUMMARY ===')
        print(f'Mean accuracy: {accuracies.mean():.4f}')
        print(f'Std accuracy: {accuracies.std(ddof=1):.4f}')
        print(f'Mean macro F1: {f1_scores.mean():.4f}')
        print(f'Std macro F1: {f1_scores.std(ddof=1):.4f}')

        return results

    def build_search_pipeline(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(
                n_components=self.n_components,
                svd_solver='randomized',
                whiten=True,
                random_state=42
            )),
            ('svc', SVC(
                kernel='rbf',
                class_weight='balanced',
            ))
        ])

        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                'svc__C': self.param_grid['C'],
                'svc__gamma': self.param_grid['gamma'],
            },
            n_iter=10,
            cv=5,
            random_state=42,
            n_jobs=-1
        )
