"""Command-line entry point for the complete project."""

import argparse
from pathlib import Path

from src.config import DATASET_DIR, FEATURE_CSV
from src.create_features import create_feature_dataset
from src.dataset import print_dataset_info, validate_dataset
from src.predict import predict_image
from src.train import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plant Leaf Disease Classification project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("dataset-info")
    info.add_argument("--dataset", type=Path, default=DATASET_DIR)

    features = subparsers.add_parser("features")
    features.add_argument("--dataset", type=Path, default=DATASET_DIR)
    features.add_argument("--output", type=Path, default=FEATURE_CSV)
    features.add_argument("--max-images", type=int, default=None)
    features.add_argument("--no-resume", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--features", type=Path, default=FEATURE_CSV)

    predict = subparsers.add_parser("predict")
    predict.add_argument("image", type=Path)

    args = parser.parse_args()

    if args.command == "dataset-info":
        print_dataset_info(args.dataset)

    elif args.command == "features":
        validate_dataset(args.dataset, expected_classes=None)
        create_feature_dataset(
            args.dataset,
            args.output,
            max_images=args.max_images,
            resume=not args.no_resume,
        )

    elif args.command == "train":
        train_and_evaluate(args.features)

    elif args.command == "predict":
        result = predict_image(args.image)
        print(f"Predicted Class: {result['class_name']}")
        print(f"Plant: {result['plant']}")
        print(f"Condition: {result['condition']}")
        print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
