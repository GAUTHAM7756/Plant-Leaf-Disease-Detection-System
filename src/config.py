"""Central configuration for the Plant Leaf Disease Classification project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "color"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_CSV = FEATURES_DIR / "leaf_features.csv"
K_COMPARISON_CSV = RESULTS_DIR / "k_comparison.csv"
CLASSIFICATION_REPORT = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX = RESULTS_DIR / "confusion_matrix.png"
METRICS_JSON = RESULTS_DIR / "metrics.json"

KNN_MODEL = MODELS_DIR / "knn_model.pkl"
SCALER = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER = MODELS_DIR / "label_encoder.pkl"

IMAGE_SIZE = (256, 256)
KMEANS_CLUSTERS = 3
KMEANS_ATTEMPTS = 5
KMEANS_EPSILON = 1.0
KMEANS_MAX_ITERATIONS = 100
RANDOM_STATE = 42

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FEATURE_COLUMNS = [
    "L_mean", "L_variance", "L_skewness",
    "a_mean", "a_variance", "a_skewness",
    "b_mean", "b_variance", "b_skewness",
    "contrast", "correlation", "entropy",
]

K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
TEST_SIZE = 0.20
GLCM_DISTANCE = 1
GLCM_ANGLE = 0.0
GLCM_LEVELS = 32

EXPECTED_CLASS_COUNT = 38
