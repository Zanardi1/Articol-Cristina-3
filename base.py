import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


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

    def load_data(self):
        from facenet_torch import FaceNetTorchClassification
        from dlib_alg import DlibClassification

        lfw_people = fetch_lfw_people(data_home=os.getcwd(), min_faces_per_person=70, resize=0.4)
        self.n_samples, self.h, self.w = lfw_people.images.shape
        if isinstance(self, FaceNetTorchClassification) or isinstance(self, DlibClassification):
            self.X = lfw_people.images
        else:
            self.X = lfw_people.data
        self.n_features = self.X.shape[1]

        self.y = lfw_people.target
        self.target_names = lfw_people.target_names
        self.n_classes = self.target_names.shape[0]

    def print_load_results(self):
        print('Total dataset size: ')
        print(f'n_samples: {self.n_samples}')
        print(f'n_features: {self.n_features}')
        print(f'n_classes: {self.n_classes}')

    def create_shared_split(self):
        indices = np.arange(self.n_samples)
        self.train_indices, self.test_indices = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
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
        from sk import SKLearnClassification

        if isinstance(self, SKLearnClassification):
            X = self.X_test_pca
        else:
            X = self.X_test

        ConfusionMatrixDisplay.from_estimator(self.model, X, self.y_test,
                                              display_labels=self.target_names,
                                              xticks_rotation='vertical')
        plt.tight_layout()
        # plt.show()
        path = Path('Results')
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / (str(self.__class__.__name__) + '.png'), format='png')

    def print_results(self):
        self.print_accuracy()
        self.print_classification_report()
        self.print_confusion_matrix()
        self.print_results_to_file()

    def print_results_to_file(self):
        path = Path('Results')
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'Results.txt', 'a') as f:
            print(f'Rezultate pentru {self.__class__.__name__}: \n', file=f)
            print(f'Face recognition accuracy: {self.accuracy:.2f}\n', file=f)
            print(classification_report(self.y_test, self.y_pred, target_names=self.target_names) + '\n', file=f)
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
        from sk import SKLearnClassification
        from resnet import TensorFlowClassification
        from vgg_deepface import VGGClassification

        if isinstance(self, SKLearnClassification):
            self.X_train = self.X_train_pca

        if isinstance(self, TensorFlowClassification):
            self.X_train = self.X_train_features

        if isinstance(self, VGGClassification):
            self.X_train = self.X_train_embedded

        self.model = model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        from resnet import TensorFlowClassification
        from vgg_deepface import VGGClassification

        if isinstance(self, TensorFlowClassification):
            self.X_test = self.X_test_features
        elif isinstance(self, VGGClassification):
            self.X_test = self.X_test_embedded

        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
