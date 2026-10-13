# pylint: disable=import-error, invalid-name, no-member

import numpy as np
from keras.applications import ResNet101
from keras.applications.resnet import preprocess_input
from keras.models import Model
from keras.utils import array_to_img, img_to_array
from sklearn.svm import SVC

from base import Base


class TensorFlowClassification(Base):

    def __init__(self):
        """Initializes the TensorFlowClassification class with default attributes."""
        super().__init__()
        self.X_resized = []
        self.model = None
        self.X_train_features = []
        self.X_test_features = []
        self.clf = []

    def preprocess_images(self, target_size=(224, 224)):
        images_resized = []
        for img in self.X:
            img = img.reshape(self.h, self.w, 1)  # Add a single channel (50, 37) → (50, 37, 1)
            img = np.repeat(img, 3, axis=-1)  # Convert grayscale to RGB (50, 37, 1) → (50, 37, 3)
            img = array_to_img(img)  # Convert to PIL Image
            img = img.resize(target_size)  # Resize to (224, 224)
            img = img_to_array(img)  # Convert back to NumPy array
            images_resized.append(img)

        images_resized = np.array(images_resized)
        images_resized = preprocess_input(images_resized)  # Normalize for ResNet101
        return images_resized

    def resize_images(self):
        self.X_resized = self.preprocess_images()

    def define_model(self):
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.model = Model(inputs=base_model.input, outputs=base_model.output)
        return self.model

    def get_features(self):
        self.X_train_features = self.model.predict(self.X_train)
        self.X_test_features = self.model.predict(self.X_test)

    def reshape_features(self):
        self.X_train_features = self.X_train_features.reshape(self.X_train_features.shape[0], -1)
        self.X_test_features = self.X_test_features.reshape(self.X_test_features.shape[0], -1)

    def execute(self):
        super().execute()
        self.resize_images()
        self.split_data(X=self.X_resized, y=self.y)
        self.define_model()
        self.get_features()
        self.reshape_features()
        self.train_model(model=SVC(kernel='linear', probability=True))
        self.evaluate_model()
        self.print_results()
