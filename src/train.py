"""
Plant Leaf Disease Classification
----------------------------------
Train and compare:

1. KNN
2. Random Forest
3. Linear SVM
4. RBF SVM

SVM probability estimates are generated using
CalibratedClassifierCV.

SVC(probability=True) is NOT used.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .config import (
    FEATURE_COLUMNS,
    FEATURE_CSV,
    MODELS_DIR,
    RESULTS_DIR,

    KNN_MODEL,
    KNN_SCALER,

    RANDOM_FOREST_MODEL,

    LINEAR_SVM_MODEL,
    LINEAR_SVM_SCALER,

    RBF_SVM_MODEL,
    RBF_SVM_SCALER,

    LABEL_ENCODER,

    MODEL_COMPARISON_JSON,
    MODEL_COMPARISON_REPORT,

    KNN_REPORT,
    RANDOM_FOREST_REPORT,
    LINEAR_SVM_REPORT,
    RBF_SVM_REPORT,

    KNN_CONFUSION_MATRIX,
    RANDOM_FOREST_CONFUSION_MATRIX,
    LINEAR_SVM_CONFUSION_MATRIX,
    RBF_SVM_CONFUSION_MATRIX,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load and validate the feature dataset."""

    if not FEATURE_CSV.exists():
        raise FileNotFoundError(
            f"Feature CSV not found:\n{FEATURE_CSV}"
        )

    df = pd.read_csv(FEATURE_CSV)

    required_columns = [
        "image",
        "class",
    ] + FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    X = df[FEATURE_COLUMNS].copy()

    y = df["class"].astype(str)

    # Convert features to numeric
    for column in FEATURE_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Remove invalid rows
    valid_rows = ~X.isna().any(axis=1)

    if not valid_rows.all():

        removed = int(
            (~valid_rows).sum()
        )

        print(
            f"Warning: removing "
            f"{removed} invalid rows."
        )

        X = X.loc[
            valid_rows
        ].copy()

        y = y.loc[
            valid_rows
        ].copy()

    # Check infinity
    if not np.isfinite(
        X.to_numpy()
    ).all():

        raise ValueError(
            "Feature dataset contains "
            "infinite values."
        )

    return X, y


# ============================================================
# BUILD MODELS
# ============================================================

def build_models():
    """
    Create the four classifiers.

    Important:
    SVC(probability=True) is NOT used.

    CalibratedClassifierCV provides
    predict_proba() for the SVM models.
    """

    models = {

        # ----------------------------------------------------
        # KNN
        # ----------------------------------------------------

        "KNN": {
            "model": KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1,
            ),
            "scaler": True,
        },

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scaler": False,
        },

        # ----------------------------------------------------
        # LINEAR SVM
        # ----------------------------------------------------

        "Linear SVM": {
            "model": CalibratedClassifierCV(
                estimator=SVC(
                    kernel="linear",
                    C=10,
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
                cv=3,
                ensemble=False,
            ),
            "scaler": True,
        },

        # ----------------------------------------------------
        # RBF SVM
        # ----------------------------------------------------

        "RBF SVM": {
            "model": CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
                cv=3,
                ensemble=False,
            ),
            "scaler": True,
        },
    }

    return models


# ============================================================
# GET OUTPUT PATHS
# ============================================================

def get_output_paths(model_name):
    """Return model, scaler, report and confusion paths."""

    if model_name == "KNN":

        return (
            KNN_MODEL,
            KNN_SCALER,
            KNN_REPORT,
            KNN_CONFUSION_MATRIX,
        )

    if model_name == "Random Forest":

        return (
            RANDOM_FOREST_MODEL,
            None,
            RANDOM_FOREST_REPORT,
            RANDOM_FOREST_CONFUSION_MATRIX,
        )

    if model_name == "Linear SVM":

        return (
            LINEAR_SVM_MODEL,
            LINEAR_SVM_SCALER,
            LINEAR_SVM_REPORT,
            LINEAR_SVM_CONFUSION_MATRIX,
        )

    if model_name == "RBF SVM":

        return (
            RBF_SVM_MODEL,
            RBF_SVM_SCALER,
            RBF_SVM_REPORT,
            RBF_SVM_CONFUSION_MATRIX,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model_name,
    model,
    use_scaler,
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder,
):
    """Train and evaluate one model."""

    print()
    print("-" * 60)
    print(
        f"Training {model_name}..."
    )
    print("-" * 60)

    # ========================================================
    # CROSS-VALIDATION PIPELINE
    # ========================================================
    #
    # For SVMs, evaluate the underlying SVC during CV.
    # Probability calibration is only required for the final
    # saved SVM model so that the application can use
    # predict_proba(). This avoids expensive nested calibration
    # inside every CV fold.
    #

    if model_name == "Linear SVM":

        cv_model = SVC(
            kernel="linear",
            C=10,
            probability=False,
            random_state=RANDOM_STATE,
        )

    elif model_name == "RBF SVM":

        cv_model = SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=False,
            random_state=RANDOM_STATE,
        )

    else:

        cv_model = clone(model)

    if use_scaler:

        cv_pipeline = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    cv_model,
                ),
            ]
        )

    else:

        cv_pipeline = Pipeline(
            [
                (
                    "model",
                    cv_model,
                ),
            ]
        )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        f"Running {CV_FOLDS}-fold "
        "cross-validation..."
    )

    cv_scores = cross_val_score(
        cv_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    cv_mean = float(
        cv_scores.mean()
    )

    cv_std = float(
        cv_scores.std()
    )

    print(
        f"CV Accuracy : "
        f"{cv_mean:.2%}"
        f" (± {cv_std:.2%})"
    )

    # ========================================================
    # FINAL SCALER
    # ========================================================

    if use_scaler:

        scaler = StandardScaler()

        X_train_final = scaler.fit_transform(
            X_train
        )

        X_test_final = scaler.transform(
            X_test
        )

    else:

        scaler = None

        X_train_final = X_train
        X_test_final = X_test

    # ========================================================
    # FINAL MODEL
    # ========================================================

    print(
        f"Training final "
        f"{model_name} model..."
    )

    final_model = clone(model)

    final_model.fit(
        X_train_final,
        y_train,
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = final_model.predict(
        X_test_final
    )

    # ========================================================
    # METRICS
    # ========================================================

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

    print(
        f"Accuracy    : "
        f"{accuracy:.2%}"
    )

    print(
        f"Precision   : "
        f"{precision:.2%}"
    )

    print(
        f"Recall      : "
        f"{recall:.2%}"
    )

    print(
        f"F1-score    : "
        f"{f1:.2%}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(
            len(
                label_encoder.classes_
            )
        ),
        target_names=(
            label_encoder.classes_
        ),
        zero_division=0,
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    (
        model_path,
        scaler_path,
        report_path,
        confusion_path,
    ) = get_output_paths(
        model_name
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        model_path,
    )

    if scaler is not None:

        joblib.dump(
            scaler,
            scaler_path,
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    labels = np.arange(
        len(
            label_encoder.classes_
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(
        figsize=(18, 16)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=(
            label_encoder.classes_
        ),
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=False,
    )

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    fig.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "model": model_name,

        "cv_accuracy_mean": cv_mean,

        "cv_accuracy_std": cv_std,

        "accuracy": float(
            accuracy
        ),

        "precision_weighted": float(
            precision
        ),

        "recall_weighted": float(
            recall
        ),

        "f1_weighted": float(
            f1
        ),

        "train_samples": int(
            len(X_train)
        ),

        "test_samples": int(
            len(X_test)
        ),

        "classes": int(
            len(
                label_encoder.classes_
            )
        ),

        "features": int(
            len(FEATURE_COLUMNS)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "PLANT LEAF DISEASE MODEL TRAINING"
    )
    print("=" * 60)

    # ========================================================
    # LOAD DATA
    # ========================================================

    X, y_text = load_data()

    print(
        f"Total samples : "
        f"{len(X):,}"
    )

    print(
        f"Features      : "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Classes       : "
        f"{y_text.nunique()}"
    )

    # ========================================================
    # LABEL ENCODING
    # ========================================================

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_text
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER,
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training      : "
        f"{len(X_train):,}"
    )

    print(
        f"Testing       : "
        f"{len(X_test):,}"
    )

    # ========================================================
    # BUILD MODELS
    # ========================================================

    models = build_models()

    all_results = {}

    # ========================================================
    # TRAIN ALL MODELS
    # ========================================================

    for model_name, settings in models.items():

        result = train_model(
            model_name=model_name,
            model=settings["model"],
            use_scaler=settings["scaler"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder,
        )

        all_results[
            model_name
        ] = result

    # ========================================================
    # FIND BEST MODEL
    # ========================================================

    best_model_name = max(
        all_results,
        key=lambda name:
        all_results[name]["accuracy"],
    )

    best_accuracy = (
        all_results[
            best_model_name
        ]["accuracy"]
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison = {}

    for model_name, result in (
        all_results.items()
    ):

        comparison[
            model_name
        ] = {
            "cv_accuracy": result[
                "cv_accuracy_mean"
            ],

            "cv_std": result[
                "cv_accuracy_std"
            ],

            "accuracy": result[
                "accuracy"
            ],

            "precision": result[
                "precision_weighted"
            ],

            "recall": result[
                "recall_weighted"
            ],

            "f1_score": result[
                "f1_weighted"
            ],
        }

    # ========================================================
    # SAVE JSON
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_data = {
        "best_model": best_model_name,

        "best_accuracy": float(
            best_accuracy
        ),

        "models": comparison,
    }

    with open(
        MODEL_COMPARISON_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison_data,
            file,
            indent=4,
        )

    # ========================================================
    # SAVE COMPARISON REPORT
    # ========================================================

    with open(
        MODEL_COMPARISON_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PLANT LEAF DISEASE "
            "MODEL COMPARISON\n"
        )

        file.write(
            "=" * 75
            + "\n\n"
        )

        file.write(
            f"Best Model: "
            f"{best_model_name}\n"
        )

        file.write(
            f"Best Accuracy: "
            f"{best_accuracy:.2%}\n\n"
        )

        file.write(
            "-" * 75
            + "\n"
        )

        file.write(
            f"{'Model':<20}"
            f"{'CV':>12}"
            f"{'Accuracy':>12}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'F1':>12}\n"
        )

        file.write(
            "-" * 75
            + "\n"
        )

        for model_name, result in (
            all_results.items()
        ):

            file.write(
                f"{model_name:<20}"
                f"{result['cv_accuracy_mean']:>11.2%}"
                f"{result['accuracy']:>11.2%}"
                f"{result['precision_weighted']:>11.2%}"
                f"{result['recall_weighted']:>11.2%}"
                f"{result['f1_weighted']:>11.2%}\n"
            )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print(
        "MODEL COMPARISON"
    )
    print("=" * 60)

    print()

    for model_name, result in (
        all_results.items()
    ):

        print(
            f"{model_name:<18}"
            f"Accuracy: "
            f"{result['accuracy']:.2%}    "
            f"F1: "
            f"{result['f1_weighted']:.2%}"
        )

    print()
    print("-" * 60)

    print(
        f"BEST MODEL: "
        f"{best_model_name}"
    )

    print(
        f"BEST ACCURACY: "
        f"{best_accuracy:.2%}"
    )

    print("-" * 60)

    print()
    print(
        "Training completed successfully."
    )

    print()
    print(
        f"Comparison JSON:\n"
        f"{MODEL_COMPARISON_JSON}"
    )

    print()
    print(
        f"Comparison Report:\n"
        f"{MODEL_COMPARISON_REPORT}"
    )

    print()
    print(
        f"Models saved in:\n"
        f"{MODELS_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()