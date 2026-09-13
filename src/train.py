"""Train, tune, evaluate, and persist the KNN classifier."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import (
    CLASSIFICATION_REPORT,
    CONFUSION_MATRIX,
    FEATURE_COLUMNS,
    FEATURE_CSV,
    KNN_MODEL,
    K_COMPARISON_CSV,
    K_VALUES,
    LABEL_ENCODER,
    METRICS_JSON,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SCALER,
    TEST_SIZE,
)


def load_feature_data(csv_path: Path = FEATURE_CSV) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and validate the 12-feature CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {csv_path}. Run feature extraction first."
        )

    df = pd.read_csv(csv_path)
    required = ["image", "class", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {missing}")

    df = df.dropna(subset=["class", *FEATURE_COLUMNS]).copy()
    if df.empty:
        raise ValueError("No usable rows remain after removing missing values.")

    X = df[FEATURE_COLUMNS].astype(np.float64).to_numpy()
    y_text = df["class"].astype(str).to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Feature CSV contains NaN or infinite values.")
    return df, X, y_text


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    """Create a readable 38-class confusion-matrix figure."""
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True class",
        xlabel="Predicted class",
        title="KNN Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(
    feature_csv: Path = FEATURE_CSV,
) -> dict:
    """Run the complete leakage-safe training and final evaluation pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, X, y_text = load_feature_data(feature_csv)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # The scaler is fitted strictly after the split and only on X_train.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select K using CV on training data only. The final test set remains untouched.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows = []

    for k in K_VALUES:
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=cv, scoring="accuracy"
        )
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        comparison_rows.append({
            "k": k,
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(K_COMPARISON_CSV, index=False)

    # K is chosen using training-only CV, not the held-out test score.
    best_row = comparison_df.loc[comparison_df["cv_accuracy_mean"].idxmax()]
    best_k = int(best_row["k"])

    final_model = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean",
    )
    final_model.fit(X_train_scaled, y_train)
    y_pred = final_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    class_names = label_encoder.classes_.tolist()
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    CLASSIFICATION_REPORT.write_text(
        "Plant Leaf Disease Classification\n"
        "=================================\n\n"
        f"Best K: {best_k}\n"
        f"Test samples: {len(y_test)}\n"
        "Metric averaging: weighted\n\n"
        f"Accuracy:  {accuracy:.6f}\n"
        f"Precision: {precision:.6f}\n"
        f"Recall:    {recall:.6f}\n"
        f"F1-score:  {f1:.6f}\n\n"
        "Classification report:\n"
        f"{report}\n",
        encoding="utf-8",
    )

    _plot_confusion_matrix(cm, class_names, CONFUSION_MATRIX)

    metrics = {
        "best_k": best_k,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "class_count": int(len(class_names)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "metric_average": "weighted",
        "classes": class_names,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(final_model, KNN_MODEL)
    joblib.dump(scaler, SCALER)
    joblib.dump(label_encoder, LABEL_ENCODER)

    print("\nTraining complete.")
    print(f"Best K (training CV): {best_k}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Saved model: {KNN_MODEL}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURE_CSV)
    args = parser.parse_args()
    train_and_evaluate(args.features)


if __name__ == "__main__":
    main()
