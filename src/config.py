"""Project configuration for Plant Leaf Disease Classification."""

from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (256, 256)


# ============================================================
# DATASET
# ============================================================

FEATURE_CSV = FEATURES_DIR / "leaf_features.csv"


# ============================================================
# K-MEANS SETTINGS
# ============================================================

KMEANS_CLUSTERS = 3
KMEANS_ATTEMPTS = 10
KMEANS_MAX_ITERATIONS = 100
KMEANS_EPSILON = 1.0

RANDOM_STATE = 42


# ============================================================
# GLCM SETTINGS
# ============================================================

GLCM_DISTANCE = 1
GLCM_ANGLE = 0
GLCM_LEVELS = 32


# ============================================================
# MODEL FILES
# ============================================================

# KNN
KNN_MODEL = MODELS_DIR / "knn_model.pkl"
KNN_SCALER = MODELS_DIR / "knn_scaler.pkl"


# Random Forest
RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest_model.pkl"


# Linear SVM
LINEAR_SVM_MODEL = MODELS_DIR / "linear_svm_model.pkl"
LINEAR_SVM_SCALER = MODELS_DIR / "linear_svm_scaler.pkl"


# RBF SVM
RBF_SVM_MODEL = MODELS_DIR / "rbf_svm_model.pkl"
RBF_SVM_SCALER = MODELS_DIR / "rbf_svm_scaler.pkl"


# Label encoder shared by all models
LABEL_ENCODER = MODELS_DIR / "label_encoder.pkl"


# ============================================================
# MODEL COMPARISON RESULTS
# ============================================================

MODEL_COMPARISON_JSON = (
    RESULTS_DIR / "model_comparison.json"
)

MODEL_COMPARISON_REPORT = (
    RESULTS_DIR / "model_comparison_report.txt"
)


# ============================================================
# INDIVIDUAL CLASSIFICATION REPORTS
# ============================================================

KNN_REPORT = (
    RESULTS_DIR / "knn_classification_report.txt"
)

RANDOM_FOREST_REPORT = (
    RESULTS_DIR / "random_forest_classification_report.txt"
)

LINEAR_SVM_REPORT = (
    RESULTS_DIR / "linear_svm_classification_report.txt"
)

RBF_SVM_REPORT = (
    RESULTS_DIR / "rbf_svm_classification_report.txt"
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

KNN_CONFUSION_MATRIX = (
    RESULTS_DIR / "knn_confusion_matrix.png"
)

RANDOM_FOREST_CONFUSION_MATRIX = (
    RESULTS_DIR / "random_forest_confusion_matrix.png"
)

LINEAR_SVM_CONFUSION_MATRIX = (
    RESULTS_DIR / "linear_svm_confusion_matrix.png"
)

RBF_SVM_CONFUSION_MATRIX = (
    RESULTS_DIR / "rbf_svm_confusion_matrix.png"
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "L_mean",
    "L_variance",
    "L_skewness",

    "a_mean",
    "a_variance",
    "a_skewness",

    "b_mean",
    "b_variance",
    "b_skewness",

    "contrast",
    "correlation",
    "entropy",
]