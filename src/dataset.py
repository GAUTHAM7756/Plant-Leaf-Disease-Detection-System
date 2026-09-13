"""Dataset discovery, validation, and lightweight image-path handling."""

from pathlib import Path
from typing import List, Tuple

from .config import DATASET_DIR, SUPPORTED_EXTENSIONS


def discover_classes(dataset_dir: Path = DATASET_DIR) -> List[str]:
    """Return sorted class-folder names from the color dataset."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}\n"
            "Create dataset/color and place one folder per class inside it."
        )

    classes = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class folders found in: {dataset_dir}")
    return classes


def discover_images(
    dataset_dir: Path = DATASET_DIR,
) -> List[Tuple[Path, str]]:
    """Return image paths paired with their class names without loading images."""
    classes = discover_classes(dataset_dir)
    samples: List[Tuple[Path, str]] = []

    for class_name in classes:
        class_dir = dataset_dir / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((path, class_name))
    return samples


def validate_dataset(
    dataset_dir: Path = DATASET_DIR,
    expected_classes: int | None = 38,
) -> dict:
    """Validate the dataset structure and return useful statistics."""
    classes = discover_classes(dataset_dir)
    samples = discover_images(dataset_dir)

    empty_classes = []
    counts = {}
    for class_name in classes:
        count = sum(1 for _, label in samples if label == class_name)
        counts[class_name] = count
        if count == 0:
            empty_classes.append(class_name)

    if empty_classes:
        raise ValueError(f"Empty class folders: {empty_classes}")

    if expected_classes is not None and len(classes) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} classes, but found {len(classes)}: {classes}"
        )

    return {
        "dataset_dir": str(dataset_dir),
        "class_count": len(classes),
        "image_count": len(samples),
        "classes": classes,
        "images_per_class": counts,
    }


def print_dataset_info(dataset_dir: Path = DATASET_DIR) -> None:
    """Print dataset statistics in a readable form."""
    info = validate_dataset(dataset_dir, expected_classes=None)
    print(f"Dataset: {info['dataset_dir']}")
    print(f"Classes: {info['class_count']}")
    print(f"Images:  {info['image_count']}")
    for name, count in info["images_per_class"].items():
        print(f"  {name}: {count}")
