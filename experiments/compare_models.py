"""
Compare classical machine learning models for plant leaf disease classification.

Models:
    KNN
    SVM
    Random Forest

KNN remains the primary classifier.
SVM and Random Forest are used for comparison.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, FEATURE_CSV


RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
REPORT_FILE = RESULTS_DIR / "model_comparison_report.txt"


# Settings

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_SPLITS = 5
KNN_K = 5


def print_header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def load_data():
    """Load the extracted feature dataset."""

    print_header("DATASET")

    print(f"Feature CSV: {FEATURE_CSV}")

    if not FEATURE_CSV.exists():
        raise FileNotFoundError(
            f"\nFeature CSV not found:\n{FEATURE_CSV}\n\n"
            "Check FEATURE_CSV in src/config.py."
        )

    df = pd.read_csv(FEATURE_CSV)

    required_columns = ["image", "class"] + FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    X = df[FEATURE_COLUMNS].copy()
    y_text = df["class"].astype(str)

    for column in FEATURE_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    valid_rows = ~X.isna().any(axis=1)

    if not valid_rows.all():
        removed = (~valid_rows).sum()
        print(f"Removing {removed} rows with missing values.")
        X = X.loc[valid_rows].copy()
        y_text = y_text.loc[valid_rows].copy()

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Feature dataset contains infinite values."
        )

    print(f"Samples: {len(X):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Classes: {y_text.nunique()}")

    if y_text.nunique() != 38:
        print(
            f"Warning: expected 38 classes, "
            f"found {y_text.nunique()}."
        )

    return df, X, y_text


def encode_labels(y_text):
    """Encode class names into integer labels."""

    print_header("LABEL ENCODING")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    print(f"Classes encoded: {len(encoder.classes_)}")

    return y, encoder


def split_data(X, y):
    """Create the stratified train-test split."""

    print_header("TRAIN / TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")

    return X_train, X_test, y_train, y_test


def build_models():
    """Create the classical ML models."""

    models = {}

    models["KNN"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=KNN_K,
                    metric="euclidean",
                    weights="uniform",
                ),
            ),
        ]
    )

    models["SVM"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                ),
            ),
        ]
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    return models


def cross_validate_model(
    model_name,
    model,
    X_train,
    y_train,
):
    """Perform cross-validation on the training set."""

    print_header(f"{model_name} CROSS-VALIDATION")

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    for index, score in enumerate(scores, start=1):
        print(f"Fold {index}: {score:.4f}")

    print(f"Mean CV accuracy: {scores.mean():.4f}")
    print(f"CV std: {scores.std():.4f}")

    return scores


def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder,
):
    """Train and evaluate a model."""

    print_header(f"TRAINING {model_name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
    )

    print()
    print(f"Accuracy:           {accuracy:.4f} ({accuracy:.2%})")
    print(f"Weighted Precision: {precision:.4f} ({precision:.2%})")
    print(f"Weighted Recall:    {recall:.4f} ({recall:.2%})")
    print(f"Weighted F1-score:  {f1:.4f} ({f1:.2%})")
    print(f"Confusion matrix:   {cm.shape}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "trained_model": model,
    }


def save_results(results):
    """Save model comparison results."""

    rows = []

    for result in results:
        rows.append(
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision_weighted": result[
                    "precision_weighted"
                ],
                "recall_weighted": result[
                    "recall_weighted"
                ],
                "f1_weighted": result[
                    "f1_weighted"
                ],
            }
        )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        COMPARISON_CSV,
        index=False,
    )

    print(f"\nSaved: {COMPARISON_CSV}")

    return results_df


def save_report(
    results,
    label_encoder,
    cv_results,
):
    """Save detailed comparison information."""

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "CLASSICAL MACHINE LEARNING MODEL COMPARISON\n"
        )
        file.write("=" * 60 + "\n\n")

        file.write(
            "Plant Leaf Disease Classification Project\n\n"
        )

        file.write(
            f"Feature CSV: {FEATURE_CSV}\n"
        )

        file.write(
            f"Number of classes: "
            f"{len(label_encoder.classes_)}\n"
        )

        file.write(
            f"Number of features: "
            f"{len(FEATURE_COLUMNS)}\n"
        )

        file.write(
            f"Test size: {TEST_SIZE}\n"
        )

        file.write(
            f"Random state: {RANDOM_STATE}\n\n"
        )

        file.write("FEATURES\n")
        file.write("-" * 60 + "\n")

        for feature in FEATURE_COLUMNS:
            file.write(f"{feature}\n")

        file.write("\n")

        file.write("MODEL PERFORMANCE\n")
        file.write("-" * 60 + "\n\n")

        for result in results:

            file.write(
                f"Model: {result['model']}\n"
            )

            file.write(
                f"Accuracy: "
                f"{result['accuracy']:.4f}\n"
            )

            file.write(
                f"Weighted Precision: "
                f"{result['precision_weighted']:.4f}\n"
            )

            file.write(
                f"Weighted Recall: "
                f"{result['recall_weighted']:.4f}\n"
            )

            file.write(
                f"Weighted F1-score: "
                f"{result['f1_weighted']:.4f}\n\n"
            )

        file.write("CROSS-VALIDATION\n")
        file.write("-" * 60 + "\n\n")

        for model_name, scores in cv_results.items():

            file.write(
                f"{model_name}\n"
            )

            file.write(
                f"Mean accuracy: "
                f"{scores.mean():.4f}\n"
            )

            file.write(
                f"Standard deviation: "
                f"{scores.std():.4f}\n"
            )

            file.write(
                "Fold scores: "
                + ", ".join(
                    f"{score:.4f}"
                    for score in scores
                )
                + "\n\n"
            )

        file.write("CLASSIFICATION REPORTS\n")
        file.write("-" * 60 + "\n")

        for result in results:

            file.write(
                f"\n{result['model']}\n"
            )
            file.write("=" * 60 + "\n")

            file.write(
                result["classification_report"]
            )

            file.write("\n")

    print(f"Saved: {REPORT_FILE}")


def save_models(results):
    """Save the trained comparison models."""

    print_header("SAVING MODELS")

    filenames = {
        "KNN": "comparison_knn.pkl",
        "SVM": "comparison_svm.pkl",
        "Random Forest": "comparison_random_forest.pkl",
    }

    for result in results:

        model_name = result["model"]
        filename = filenames[model_name]

        output_path = MODELS_DIR / filename

        joblib.dump(
            result["trained_model"],
            output_path,
        )

        print(f"{model_name}: {output_path}")


def main():

    print_header("CLASSICAL MODEL COMPARISON")

    # Load data
    _, X, y_text = load_data()

    # Encode labels
    y, label_encoder = encode_labels(y_text)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    # Build models
    models = build_models()

    cv_results = {}
    results = []

    # Cross-validation
    for model_name, model in models.items():

        scores = cross_validate_model(
            model_name,
            model,
            X_train,
            y_train,
        )

        cv_results[model_name] = scores

    # Model evaluation
    for model_name, model in models.items():

        result = evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            label_encoder,
        )

        results.append(result)

    # Save results
    results_df = save_results(results)

    save_report(
        results,
        label_encoder,
        cv_results,
    )

    save_models(results)

    # Display comparison
    print_header("FINAL MODEL COMPARISON")

    display_df = results_df.copy()

    for column in [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.2%}"
        )

    print(
        display_df.to_string(index=False)
    )

    print()
    print("Comparison completed successfully.")
    print(f"Results: {COMPARISON_CSV}")
    print(f"Report:  {REPORT_FILE}")


if __name__ == "__main__":
    main()