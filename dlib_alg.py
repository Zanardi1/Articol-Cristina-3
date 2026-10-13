import bz2
import os
import urllib.request

import cv2
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from base import Base


class DlibClassification(Base):
    def __init__(self):
        super().__init__()
        self.y_encoded = None
        self.detector = None
        self.predictor = None
        self.recognizer = None
        self.face_embeddings = []
        self.labels = []
        self.encoder = None
        self.knn = None
        self.image = None
        self.embedding = None
        self.embedding_indices = []
        self.failed_indices = []

    def download_and_extract(self, filename, url):
        """Download and extract a .bz2 file."""
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename + ".bz2")

            print(f"Extracting {filename}...")
            with bz2.BZ2File(filename + ".bz2", "rb") as bz2_file:
                with open(filename, "wb") as out_file:
                    out_file.write(bz2_file.read())

            os.remove(filename + ".bz2")  # Remove compressed file
            print(f"Downloaded and extracted {filename}.")

    def get_face_embedding(self, image):
        # Convert float32 to uint8
        image = (image * 255).astype(np.uint8)

        # Resize the image to a larger size (e.g., 200x200)
        image_resized = cv2.resize(image, (200, 200))

        # Convert to grayscale
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        for face in faces:
            shape = self.predictor(gray, face)
            aligned_face = dlib.get_face_chip(image_resized, shape)
            self.embedding = self.recognizer.compute_face_descriptor(aligned_face)
            return np.array(self.embedding)

        return None

    def load_data(self):
        super().load_data()

        self.download_and_extract("shape_predictor_68_face_landmarks.dat",
                                  "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        self.download_and_extract("dlib_face_recognition_resnet_model_v1.dat",
                                  "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")

    def initialize_detectors(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    def fill_face_embeddings_and_labels(self):
        self.face_embeddings = []
        self.labels = []
        self.embedding_indices = []
        self.failed_indices = []

        for i in range(self.n_samples):
            image = cv2.cvtColor(self.X[i], cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR (dlib expects color)
            embedding = self.get_face_embedding(image)

            if embedding is not None:
                self.face_embeddings.append(embedding)
                self.labels.append(self.y[i])
                self.embedding_indices.append(i)
            else:
                self.failed_indices.append(i)

        self.face_embeddings = np.asarray(self.face_embeddings)
        self.labels = np.array(self.labels)
        self.embedding_indices = np.asarray(self.embedding_indices)

        train_failures = np.isin(
            self.failed_indices, self.train_indices
        ).sum()

        test_failures = np.isin(
            self.failed_indices, self.test_indices
        ).sum()

        print(f'Dlib face detection failures: {len(self.failed_indices)}')
        print(f'Training set failures: {train_failures}')
        print(f'Test set failures: {test_failures}')
        print(f'Test set detection coverage: {(len(self.test_indices) - test_failures) / len(self.test_indices):.2%}')

    def encode_targets(self):
        self.encoder = LabelEncoder()
        self.y_encoded = self.encoder.fit_transform(self.labels)

    def execute(self):
        super().execute()
        self.load_data()
        self.initialize_detectors()
        self.fill_face_embeddings_and_labels()
        self.split_data(X=self.face_embeddings, y=self.labels, original_indices=self.embedding_indices)
        self.train_model(model=KNeighborsClassifier())
        self.evaluate_model()
        self.print_results()
