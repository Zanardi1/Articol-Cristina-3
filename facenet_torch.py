"""FaceNet feature extraction using pretrained InceptionResnetV1.

Loads VGGFace2-pretrained weights through facenet-pytorch.
Converts LFW grayscale images into three-channel 160x160 tensors,
normalizes them, and extracts embeddings on the CPU.

The main experiment uses these fixed embeddings with a linear SVM.
No LFW fine-tuning of the pretrained network is performed.

See audit_facenet_normalization.py for the preprocessing audit.
"""

import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from torchvision import transforms

from base import Base


class FaceNetTorchClassification(Base):
    def __init__(self):
        super().__init__()
        self.processed_images = []
        self.X_processed = []
        self.X_embeddings = None

    def preprocess_images(self):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)),
            transforms.Resize((160, 160)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        for img in self.X:
            img_tensor = transform(img)
            self.processed_images.append(img_tensor)
        return torch.stack(self.processed_images)

    def process_images(self):
        self.X_processed = self.preprocess_images()

        print('FaceNet preprocessing diagnostics: ')
        print(f'Shape: {self.X_processed.shape}')
        print(f'Data type: {self.X_processed.dtype}')
        print(f'Minimum: {self.X_processed.min().item():.4f}')
        print(f'Maximum: {self.X_processed.max().item():.4f}')

    def load_model(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def extract_embeddings(self, images):
        images = images.to(torch.device('cpu'))
        self.model = self.model.to(torch.device('cpu'))
        with torch.no_grad():
            return self.model(images).numpy()

    def get_embeddings(self):
        self.X_embeddings = self.extract_embeddings(self.X_processed)

    def load_data(self):
        super().load_data(image_format='images')

    def execute(self):
        super().execute()
        self.process_images()
        self.load_model()
        self.get_embeddings()
        self.split_data(X=self.X_embeddings, y=self.y)
        self.train_model(model=SVC(kernel='linear', probability=True))
        self.evaluate_model()
        self.print_results()
