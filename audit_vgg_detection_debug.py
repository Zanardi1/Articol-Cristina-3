"""Diagnose DeepFace detection behavior on selected LFW images.

Tests enforcement-enabled and fallback-enabled detection on a small
selection of preprocessed images and prints diagnostic information.

Run: python audit_vgg_detection_debug.py
Does not write the main experimental results.
"""

from deepface import DeepFace

from vgg_deepface import VGGClassification

model = VGGClassification()
model.load_data()
model.preprocess_images()

for index in [0, 1, 2, 100, 500]:
    img = model.X_rgb[index]

    print(f"\n=== Image {index} ===")
    print("Shape:", img.shape)
    print("Dtype:", img.dtype)
    print("Range:", float(img.min()), float(img.max()))

    for enforce in [True, False]:
        try:
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend="opencv",
                enforce_detection=enforce,
                align=True
            )
            print(
                f"enforce_detection={enforce}:",
                len(faces),
                "faces",
                "confidences:",
                [f.get("confidence") for f in faces]
            )
        except Exception as exc:
            print(
                f"enforce_detection={enforce}:",
                type(exc).__name__,
                str(exc)
            )
