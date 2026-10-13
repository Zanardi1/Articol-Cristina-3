# pylint: disable=no-member, invalid-name

from pathlib import Path
from time import time

import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from base import Base


class SKLearnClassification(Base):

    def __init__(self):
        """Initializes the SKLearnClassification object with default values."""
        super().__init__()
        self.y_pred = None
        self.pca = None
        self.eigenfaces = None
        self.X_train_pca = None
        self.X_test_pca = None
        self.n_components = 150
        self.clf = None
        self.param_grid = {
            'C': loguniform(1e3, 1e5),
            'gamma': loguniform(1e-4, 1e-1)
        }

    def plot_gallery(self, images, titles, h, w, n_row=3, n_col=4):
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.9, hspace=0.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks()
            plt.yticks()
        # plt.show()
        plt.savefig(f'Results/Gallery {self.__class__.__name__}.png', format='png')

    def title(self, y_pred, y_test, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        return f"Predicted: {pred_name}\ntrue: {true_name}"

    def scale_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        return self.X_train, self.X_test

    def extract_eigenfaces(self):
        self.n_components = 150
        print(f'Extracting top {self.n_components} eigenfaces from {self.X_train.shape[0]} faces')
        t0 = time()
        self.pca = PCA(n_components=self.n_components, svd_solver='randomized', whiten=True).fit(self.X_train)
        print(f'Done in {time() - t0:.3f} s')
        return self.pca

    def reshape_components(self):
        self.eigenfaces = self.pca.components_.reshape((self.n_components, self.h, self.w))
        return self.eigenfaces

    def transform_data(self):
        print('Projecting the input data on the eigenfaces orthonormal basis')
        t0 = time()
        self.X_train_pca = self.pca.transform(self.X_train)
        self.X_test_pca = self.pca.transform(self.X_test)
        print(f'Done in {time() - t0:.3f} s')
        return self.X_train_pca, self.X_test_pca

    def prediction(self):
        print("Predicting people''s names on the test set: ")
        t0 = time()
        self.y_pred = self.model.predict(self.X_test_pca)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f'Done in {time() - t0:.3f} s')
        return self.y_pred

    def print_classification_report(self):
        print(classification_report(self.y_test, self.y_pred, target_names=self.target_names))

    def print_confusion_matrix(self):
        path = Path('Results')
        path.mkdir(parents=True, exist_ok=True)
        ConfusionMatrixDisplay.from_estimator(self.model, self.X_test_pca, self.y_test,
                                              display_labels=self.target_names,
                                              xticks_rotation='vertical')
        plt.tight_layout()
        # plt.show()
        plt.savefig(path / 'Confusion Matrix SKLearn.png', format='png')

    def print_plot_gallery(self):
        prediction_titles = [
            self.title(y_pred=self.y_pred, y_test=self.y_test, target_names=self.target_names, i=i) for i in
            range(self.y_pred.shape[0])
        ]
        self.plot_gallery(images=self.X_test, titles=prediction_titles, h=self.h, w=self.w)

    def print_results(self):
        super().print_results()
        self.print_plot_gallery()

    def execute(self):
        super().execute()
        self.split_data(X=self.X, y=self.y)
        self.scale_data()
        self.extract_eigenfaces()
        self.reshape_components()
        self.transform_data()
        self.train_model(
            model=RandomizedSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_distributions=self.param_grid,
                                     n_iter=10))
        self.prediction()
        self.print_results()
