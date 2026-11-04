
"""Command-line launcher for the LFW facial-recognition research project.

Provides a single entry point for experiment execution, reporting,
scientific audits, and project navigation.

Run: python project.py help

This launcher delegates to existing scripts without changing their logic.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results"

AUDITS = {
    "evaluation": (
        "audit_evaluation_consistency.py",
        "Verify saved predictions, partitions, and metrics",
    ),
    "facenet": (
        "audit_facenet_normalization.py",
        "Compare FaceNet normalization conventions",
    ),
    "vgg": (
        "audit_vgg_detection.py",
        "Test VGG-Face detection and fallback behavior",
    ),
    "vgg-debug": (
        "audit_vgg_detection_debug.py",
        "Inspect detection on selected images",
    ),
    "pretrained": (
        "audit_pretrained_models.py",
        "Inventory pretrained model files and hashes",
    ),
    "deepface": (
        "audit_deepface.py",
        "Inspect installed DeepFace implementation",
    ),
}

RESULT_FILES = {
    "multi_split_results.csv": "Individual evaluation metrics",
    "multi_split_summary.csv": "Five-split aggregate metrics",
    "paper_comparison.csv": "Publication comparison table",
    "predictions.csv": "Individual test predictions",
    "class_metrics.csv": "Per-class metrics",
    "environment_info.json": "Recorded environment",
    "facenet_normalization_audit.json": "FaceNet normalization audit",
    "vgg_detection_audit.json": "VGG-Face detection audit",
    "pretrained_model_audit.json": "Pretrained checkpoint inventory",
}


def run_script(filename):
    """Execute an existing repository script in a separate Python process."""
    script = ROOT / filename

    if not script.is_file():
        print(f"Missing script: {script}", file=sys.stderr)
        return 1

    print(f"\nExecuting: {filename}", flush=True)

    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=False,
        )
        return completed.returncode
    except OSError as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        return 1


def show_info():
    """Display a short overview of the repository."""
    print("\nFACIAL RECOGNITION RESEARCH PROJECT")
    print("=" * 45)
    print("Dataset: LFW")
    print("Images: 1,288")
    print("Identities: 7")
    print("Pipelines: SKLearn, FaceNet, VGG-Face, ResNet101, Dlib")
    print("Evaluation: 5 shared stratified holdout splits")
    print("Total evaluations: 25")

    print("\nWORKFLOW")
    print("main.py -> multi_split.py -> algorithm implementations")
    print("        -> evaluation_reporting.py -> Results/")

    print("\nDOCUMENTATION")
    for filename in (
        "start_here.md",
        "readme.md",
        "methodology.md",
        "reproducibility.md",
    ):
        status = "available" if (ROOT / filename).is_file() else "missing"
        print(f"  {filename}: {status}")

    print("\nUse 'python project.py help' for commands.")


def show_results():
    """List saved result files without modifying them."""
    print("\nSAVED RESULTS")
    print("=" * 45)

    if not RESULTS.is_dir():
        print("Results directory does not exist.")
        return

    for filename, description in RESULT_FILES.items():
        path = RESULTS / filename
        status = "FOUND" if path.is_file() else "MISSING"
        print(f"[{status}] {filename}")
        print(f"        {description}")

    matrices = RESULTS / "confusion_matrices"
    count = len(list(matrices.glob("*.png"))) if matrices.is_dir() else 0
    print(f"\nConfusion matrices: {count} PNG files")
    print(f"Results directory: {RESULTS}")


def list_audits():
    """Display the available scientific audits."""
    print("\nAVAILABLE AUDITS")
    print("=" * 45)

    for name, (script, description) in AUDITS.items():
        print(f"\n{name}")
        print(f"  {description}")
        print(f"  Script: {script}")

    print("\nSome audits load pretrained models or write audit outputs.")


def confirm_experiment():
    """Require confirmation before rerunning the full experiment."""
    print("\nWARNING: FULL EXPERIMENT")
    print("=" * 45)
    print("This operation runs all five pipelines over five splits.")
    print("It may take considerable time and download model weights.")
    print("Existing experimental results may be overwritten.")
    print("\nPreserve the Results directory before proceeding.")

    try:
        answer = input("\nType RUN to continue: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return False

    if answer != "RUN":
        print("Cancelled. No experiment launched.")
        return False

    return True


def build_parser():
    """Configure the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Launcher for the LFW facial-recognition project."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="command"
    )

    subparsers.add_parser("help", help="Show available commands")
    subparsers.add_parser("info", help="Show project information")
    subparsers.add_parser("results", help="List saved results")
    subparsers.add_parser("report", help="Regenerate publication table")
    subparsers.add_parser("run", help="Run the full experiment")

    audit_parser = subparsers.add_parser(
        "audit",
        help="Run or list scientific audits"
    )

    audit_parser.add_argument(
        "name",
        nargs="?",
        choices=sorted(AUDITS),
        help="Name of the audit to run"
    )

    audit_parser.add_argument(
        "--list",
        action="store_true",
        help="List available audits"
    )

    return parser


def main():
    """Dispatch the requested command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "help"):
        parser.print_help()
        return 0

    if args.command == "info":
        show_info()
        return 0

    if args.command == "results":
        show_results()
        return 0

    if args.command == "report":
        return run_script("report_results.py")

    if args.command == "audit":
        if args.list or args.name is None:
            list_audits()
            return 0

        filename, _ = AUDITS[args.name]
        return run_script(filename)

    if args.command == "run":
        if not confirm_experiment():
            return 0
        return run_script("main.py")

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
