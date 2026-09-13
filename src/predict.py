"""Single-image prediction using the persisted scaler, encoder, and KNN model."""

from pathlib import Path

import joblib
import numpy as np

from .config import FEATURE_COLUMNS, KNN_MODEL, LABEL_ENCODER, SCALER
from .features import extract_features
from .preprocessing import preprocess_image
from .segmentation import segment_leaf


def _load_models():
    """Load all persisted prediction artifacts."""
    for path in (KNN_MODEL, SCALER, LABEL_ENCODER):
        if not path.exists():
            raise FileNotFoundError(
                f"Required model file is missing: {path}. Run training first."
            )

    return (
        joblib.load(KNN_MODEL),
        joblib.load(SCALER),
        joblib.load(LABEL_ENCODER),
    )


def parse_class_name(class_name: str) -> dict:
    """Convert a dataset class name into plant, condition, and status."""
    parts = class_name.split("___", 1)
    plant = parts[0].replace("_", " ").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) == 2 else "Unknown"

    healthy = condition.lower() in {"healthy", "health"}
    return {
        "class_name": class_name,
        "plant": plant,
        "condition": "Healthy" if healthy else condition,
        "status": "Healthy" if healthy else "Disease",
    }


def predict_image(image_path: str | Path) -> dict:
    """Run preprocessing, segmentation, feature extraction, and KNN prediction."""
    model, scaler, label_encoder = _load_models()

    processed = preprocess_image(image_path)
    segmentation = segment_leaf(processed["lab"])

    vector = extract_features(
        processed["lab"],
        processed["resized_rgb"],
        segmentation["leaf_mask"],
    )
    if vector.shape != (12,):
        raise ValueError(
            f"Prediction requires exactly 12 features; received {vector.shape}."
        )

    scaled = scaler.transform(vector.reshape(1, -1))
    encoded_prediction = model.predict(scaled)[0]
    class_name = label_encoder.inverse_transform([encoded_prediction])[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled)[0]

    result = parse_class_name(str(class_name))
    result.update({
        "original_rgb": processed["original_rgb"],
        "processed_rgb": processed["resized_rgb"],
        "segmented_rgb": segmentation["segmented_rgb"],
        "leaf_mask": segmentation["leaf_mask"],
        "masked_leaf": segmentation["masked_leaf"],
        "feature_vector": vector,
        "feature_names": FEATURE_COLUMNS,
        "scaled_features": scaled[0],
        "probabilities": probabilities,
        "background_cluster": segmentation["background_cluster"],
    })
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    prediction = predict_image(args.image)
    print(f"Predicted class: {prediction['class_name']}")
    print(f"Plant: {prediction['plant']}")
    print(f"Condition: {prediction['condition']}")
    print(f"Status: {prediction['status']}")
