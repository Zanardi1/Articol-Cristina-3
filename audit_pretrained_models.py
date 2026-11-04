"""Inventory pretrained model checkpoints used by the experiment.

Loads the FaceNet, VGG-Face, and ResNet101 pretrained models, searches
their checkpoint locations, and records file sizes and SHA-256 hashes.
Also inventories the Dlib model files in the working directory.

Output: Results/pretrained_model_audit.json
Run: python audit_pretrained_models.py

May download model weights if they are not already cached.
"""

import hashlib
import json
from pathlib import Path

from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1
from keras.applications import ResNet101


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
                lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def file_record(path):
    path = Path(path)

    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path)
    }


def main():
    # Ensure pretrained checkpoints are available.
    print("Loading FaceNet...")
    InceptionResnetV1(pretrained="vggface2").eval()

    print("Loading VGG-Face...")
    DeepFace.build_model("VGG-Face")

    print("Loading ResNet101...")
    ResNet101(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    home = Path.home()

    locations = {
        "FaceNet": [
            home / ".cache" / "torch" / "checkpoints"
        ],
        "VGG-Face": [
            home / ".deepface" / "weights"
        ],
        "ResNet101": [
            home / ".keras" / "models"
        ],
        "Dlib": [
            Path.cwd()
        ]
    }

    patterns = {
        "FaceNet": ["*vggface2*"],
        "VGG-Face": ["*vgg_face*"],
        "ResNet101": ["*resnet101*"],
        "Dlib": [
            "dlib_face_recognition_resnet_model_v1.dat",
            "shape_predictor_68_face_landmarks.dat"
        ]
    }

    results = {}

    for name, directories in locations.items():
        records = []

        for directory in directories:
            if not directory.exists():
                continue

            for pattern in patterns[name]:
                for path in directory.rglob(pattern):
                    if path.is_file():
                        records.append(file_record(path))

        results[name] = records

    output_dir = Path("Results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / "pretrained_model_audit.json"

    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n=== CHECKPOINT AUDIT ===")

    for name, records in results.items():
        print(f"\n{name}: {len(records)} matching files")

        for record in records:
            print("Filename:", record["filename"])
            print("Size:", record["size_bytes"])
            print("SHA256:", record["sha256"])

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
