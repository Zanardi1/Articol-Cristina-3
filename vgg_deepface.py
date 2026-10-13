import cv2
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from base import Base


class VGGClassification(Base):
    def __init__(self):
        super().__init__()
        self.X_rgb = None
        self.y_encoded = None
        self.clf = None
        self.X_train_embedded = []
        self.X_test_embedded = []

    def preprocess_images(self):
        self.X_rgb = np.array([
            cv2.resize(cv2.cvtColor(img.reshape(self.h, self.w), cv2.COLOR_GRAY2RGB), (224, 224))
            for img in self.X
        ])

    def encode_labels(self):
        encoder = LabelEncoder()
        self.y_encoded = encoder.fit_transform(self.y)

    def extract_embeddings(self, images):
        return np.array([
            DeepFace.represent(img_path=img, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
            for img in images
        ])

    def get_embeddings(self):
        self.X_train_embedded = self.extract_embeddings(self.X_train)
        self.X_test_embedded = self.extract_embeddings(self.X_test)

    def execute(self):
        super().execute()
        self.preprocess_images()
        self.encode_labels()
        self.split_data(X=self.X_rgb, y=self.y_encoded, stratify=self.y_encoded)
        self.get_embeddings()
        self.train_model(model=SVC(kernel='linear', probability=True))
        self.evaluate_model()
        self.print_results()