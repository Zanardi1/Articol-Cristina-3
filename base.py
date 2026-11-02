import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from experiment_config import TEST_SIZE,MIN_FACES_PER_PERSON,IMAGE_RESIZE,EXPECTED_SAMPLES,EXPECTED_CLASSES


class Base:

    def __init__(self):
        self.n_samples = 0
        self.h = 0
        self.w = 0
        self.X = None
        self.n_features = 0
        self.y = None
        self.target_names = None
        self.n_classes = 0
        self.y_test = []
        self.y_pred = []
        self.accuracy = 0
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.model = None
        self.train_indices = None
        self.test_indices = None

    def load_data(self,image_format='flat'):
        lfw_people = fetch_lfw_people(
            data_home=os.getcwd(),
            min_faces_per_person=MIN_FACES_PER_PERSON,
            resize=IMAGE_RESIZE
        )
        self.n_samples, self.h, self.w = lfw_people.images.shape

        if image_format=='images':
            self.X=lfw_people.images
        elif image_format=='flat':
            self.X=lfw_people.data
        else:
            raise ValueError(
                f'Unsupported image format: {image_format}'
            )
        self.n_features = self.X.shape[1]

        self.y = lfw_people.target
        self.target_names = lfw_people.target_names
        self.n_classes = len(self.target_names)
        if self.n_samples!=EXPECTED_SAMPLES:
            raise ValueError(
                f'Expected {EXPECTED_SAMPLES} samples, '
                f'found {self.n_samples}'
            )
        if self.n_classes!=EXPECTED_CLASSES:
            raise ValueError(
                f'Expected {EXPECTED_CLASSES} classes, '
                f'found {self.n_classes}'
            )

    def print_load_results(self):
        print('Total dataset size: ')
        print(f'n_samples: {self.n_samples}')
        print(f'n_features: {self.n_features}')
        print(f'n_classes: {self.n_classes}')

    def create_shared_split(self,random_state=42):
        indices = np.arange(self.n_samples)
        self.train_indices, self.test_indices = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=random_state,
            stratify=self.y
        )
        print(f'Shared training samples:{len(self.train_indices)}')
        print(f'Shared test samples:{len(self.test_indices)}')

    def execute(self):
        self.load_data()
        self.print_load_results()
        self.create_shared_split()

    def print_accuracy(self):
        print(f'Face recognition accuracy: {self.accuracy:.2f}')

    def print_classification_report(self):
        print(classification_report(self.y_test, self.y_pred, target_names=self.target_names))

    def print_confusion_matrix(self):
        path = Path('Results')
        path.mkdir(parents=True, exist_ok=True)

        ConfusionMatrixDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            display_labels=self.target_names,
            xticks_rotation='vertical',
        )
        plt.tight_layout()
        plt.savefig(path / (str(self.__class__.__name__) + '.png'), format='png')
        plt.close()

    def print_results(self):
        self.print_accuracy()
        self.print_classification_report()
        self.print_confusion_matrix()
        self.print_results_to_file()

    def print_results_to_file(self):
        path = Path('Results')
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'Results.txt', 'a',encoding='utf-8') as f:
            print(f'Rezultate pentru {self.__class__.__name__}: \n', file=f)
            print('Total dataset size: \n', file=f)
            print(f'Samples: {self.n_samples}', file=f)
            print(f'Features: {self.n_features}', file=f)
            print(f'Classes: {self.n_classes}', file=f)
            print(f'Accuracy: {self.accuracy:.2f}\n', file=f)
            print(classification_report(self.y_test, self.y_pred, target_names=self.target_names), file=f)
            print('-----\n', file=f)

    def split_data(self, X, y, stratify=None, original_indices=None):
        X = np.array(X)
        y = np.array(y)

        if original_indices is None:
            if len(X) != self.n_samples or len(y) != self.n_samples:
                raise ValueError('Filtered data must provide original indices')
            original_indices = np.arange(self.n_samples)
        else:
            original_indices = np.asarray(original_indices)

        if len(X) != len(y) or len(X) != len(original_indices):
            raise ValueError('X, y and original_indices must have same length.')

        train_mask = np.isin(original_indices, self.train_indices)
        test_mask = np.isin(original_indices, self.test_indices)

        self.X_train = X[train_mask]
        self.X_test = X[test_mask]
        self.y_train = y[train_mask]
        self.y_test = y[test_mask]

    def train_model(self, model):
        self.model = model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
