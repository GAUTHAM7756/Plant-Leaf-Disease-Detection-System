"""Generate the 12-dimensional feature CSV from every color-dataset image."""

import argparse
import csv
import logging
from pathlib import Path

from .config import FEATURE_COLUMNS, FEATURE_CSV
from .dataset import discover_images, validate_dataset
from .features import extract_features
from .preprocessing import preprocess_image
from .segmentation import segment_leaf

LOGGER = logging.getLogger("feature_extraction")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def create_feature_dataset(
    dataset_dir: Path,
    output_csv: Path = FEATURE_CSV,
    max_images: int | None = None,
    checkpoint_every: int = 500,
    resume: bool = True,
) -> dict:
    """
    Process images sequentially and write features incrementally.

    When resume=True, already completed image paths in an existing CSV are
    skipped, making interruption recovery safe.
    """
    samples = discover_images(dataset_dir)
    if max_images is not None:
        samples = samples[:max_images]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if resume and output_csv.exists():
        with output_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("image"):
                    completed.add(row["image"])

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    mode = "a" if output_csv.exists() and not write_header else "w"

    fields = ["image", "class", *FEATURE_COLUMNS]
    success = 0
    skipped = 0
    failed = 0

    with output_csv.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for index, (image_path, class_name) in enumerate(samples, start=1):
            image_key = str(image_path.resolve())
            if image_key in completed:
                skipped += 1
                continue

            try:
                processed = preprocess_image(image_path)
                segmentation = segment_leaf(processed["lab"])
                vector = extract_features(
                    processed["lab"],
                    processed["resized_rgb"],
                    segmentation["leaf_mask"],
                )

                row = {
                    "image": image_key,
                    "class": class_name,
                }
                row.update({
                    name: f"{value:.12g}"
                    for name, value in zip(FEATURE_COLUMNS, vector)
                })
                writer.writerow(row)
                f.flush()
                success += 1

            except Exception as exc:
                failed += 1
                LOGGER.error("Failed: %s | %s", image_path, exc)

            processed_count = success + failed + skipped
            if processed_count % checkpoint_every == 0:
                LOGGER.info(
                    "Progress %d/%d | success=%d | skipped=%d | failed=%d",
                    processed_count, len(samples), success, skipped, failed,
                )

    stats = {
        "discovered": len(samples),
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "output": str(output_csv),
    }
    LOGGER.info("Feature extraction complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=FEATURE_CSV)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Useful for small tests; omit for the full dataset.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    _configure_logging()

    dataset_dir = args.dataset
    if dataset_dir is None:
        from .config import DATASET_DIR
        dataset_dir = DATASET_DIR

    validate_dataset(dataset_dir, expected_classes=None)
    create_feature_dataset(
        dataset_dir,
        args.output,
        max_images=args.max_images,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
