
import json
from pathlib import Path

import numpy as np
from deepface import DeepFace

from vgg_deepface import VGGClassification


def main():
    model = VGGClassification()
    model.load_data()
    model.preprocess_images()

    detected = 0
    fallback = 0
    multiple_faces = 0
    failures = []
    fallback_indices = []

    for index, img in enumerate(model.X_rgb):
        try:
            # Test actual detection without fallback.
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend="opencv",
                enforce_detection=True,
                align=True
            )

            detected += 1

            if len(faces) > 1:
                multiple_faces += 1

        except ValueError:
            # Detection failures may trigger ValueError.
            fallback += 1
            fallback_indices.append(index)

        except Exception as exc:
            failures.append({
                "index": index,
                "error": str(exc)
            })

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(model.X_rgb)}")

    total = len(model.X_rgb)

    results = {
        "total_images": total,
        "detected_images": detected,
        "fallback_images": fallback,
        "multiple_face_images": multiple_faces,
        "unexpected_errors": failures,
        "fallback_indices": fallback_indices,
        "detection_rate": detected / total
    }

    output_dir = Path("Results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "vgg_detection_audit.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n=== VGG-FACE DETECTION AUDIT ===")
    print(f"Total images: {total}")
    print(f"Detected: {detected}")
    print(f"Fallback: {fallback}")
    print(f"Multiple faces: {multiple_faces}")
    print(f"Unexpected errors: {len(failures)}")
    print(f"Detection rate: {detected / total:.2%}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
