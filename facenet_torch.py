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
        self.model = None
        self.X_embeddings = None
        self.classifier = None

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

    def load_model(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def extract_embeddings(self, images):
        images = images.to(torch.device('cpu'))
        self.model = self.model.to(torch.device('cpu'))
        with torch.no_grad():
            return self.model(images).numpy()

    def get_embeddings(self):
        self.X_embeddings = self.extract_embeddings(self.X_processed)

    def execute(self):
        super().execute()
        self.process_images()
        self.load_model()
        self.get_embeddings()
        self.split_data(X=self.X_embeddings, y=self.y)
        self.train_model(model=SVC(kernel='linear', probability=True))
        self.evaluate_model()
        self.print_results()
