import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "keras",
    "torch",
    "torchvision",
    "facenet-pytorch",
    "deepface",
    "dlib",
    "opencv-python",
    "matplotlib",
]


def get_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main():
    info = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': sys.version,
        'operating_system': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'packages': {
            package: get_version(package) for package in PACKAGES
        },

    }

    output_dir = Path('Results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'environment_info.json'

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(info, f, indent=4)

    print(json.dumps(info, indent=4))
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
