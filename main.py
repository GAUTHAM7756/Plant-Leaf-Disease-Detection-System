import argparse
from pathlib import Path

from src.config import FEATURE_CSV
from src.predict import predict_image


def predict_command(image):
    result = predict_image(image)

    print()
    print("Prediction Result")
    print("=" * 40)

    print(
        f"Predicted Class: "
        f"{result['class_name']}"
    )

    print(
        f"Plant: "
        f"{result['plant']}"
    )

    print(
        f"Condition: "
        f"{result['condition']}"
    )

    print(
        f"Status: "
        f"{result['status']}"
    )

    if result["confidence"] is not None:
        print(
            f"Confidence: "
            f"{result['confidence']:.2%}"
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plant Leaf Disease Classification"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    subparsers.add_parser(
        "features",
        help="Show feature CSV information",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict a leaf image",
    )

    predict_parser.add_argument(
        "image",
        type=Path,
    )

    args = parser.parse_args()

    if args.command == "features":

        if not FEATURE_CSV.exists():
            print(
                f"Feature CSV not found:\n"
                f"{FEATURE_CSV}"
            )
            return

        import pandas as pd

        df = pd.read_csv(
            FEATURE_CSV
        )

        print(
            f"Feature CSV: {FEATURE_CSV}"
        )

        print(
            f"Samples: {len(df):,}"
        )

        print(
            f"Columns: {len(df.columns)}"
        )

        print(
            f"Classes: {df['class'].nunique()}"
        )

    elif args.command == "predict":

        if not args.image.exists():
            raise FileNotFoundError(
                f"Image not found:\n"
                f"{args.image}"
            )

        predict_command(
            args.image
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()