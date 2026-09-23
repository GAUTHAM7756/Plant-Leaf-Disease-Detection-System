"""
Prediction pipeline for Plant Leaf Disease Classification.

One image is processed through:
Image -> Resize -> LAB -> K-Means -> Leaf Mask
-> 12 Features -> KNN / Random Forest / Linear SVM / RBF SVM
"""

import joblib
import numpy as np

from .config import (
    FEATURE_COLUMNS,

    KNN_MODEL,
    KNN_SCALER,

    RANDOM_FOREST_MODEL,

    LINEAR_SVM_MODEL,
    LINEAR_SVM_SCALER,

    RBF_SVM_MODEL,
    RBF_SVM_SCALER,

    LABEL_ENCODER,
)

from .features import extract_features
from .preprocessing import preprocess_image
from .segmentation import segment_leaf


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_models():

    paths = [
        KNN_MODEL,
        KNN_SCALER,
        RANDOM_FOREST_MODEL,
        LINEAR_SVM_MODEL,
        LINEAR_SVM_SCALER,
        RBF_SVM_MODEL,
        RBF_SVM_SCALER,
        LABEL_ENCODER,
    ]

    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required model file is missing:\n{path}\n\n"
                "Run train.py first."
            )

    return {
        "KNN": (
            joblib.load(KNN_MODEL),
            joblib.load(KNN_SCALER),
        ),

        "Random Forest": (
            joblib.load(RANDOM_FOREST_MODEL),
            None,
        ),

        "Linear SVM": (
            joblib.load(LINEAR_SVM_MODEL),
            joblib.load(LINEAR_SVM_SCALER),
        ),

        "RBF SVM": (
            joblib.load(RBF_SVM_MODEL),
            joblib.load(RBF_SVM_SCALER),
        ),
    }, joblib.load(LABEL_ENCODER)


# ============================================================
# CLASS NAME PARSER
# ============================================================

def parse_class_name(class_name):

    parts = class_name.split(
        "___",
        1,
    )

    plant = parts[0].replace(
        "_",
        " ",
    ).strip()

    if len(parts) == 2:

        condition = parts[1].replace(
            "_",
            " ",
        ).strip()

    else:

        condition = "Unknown"

    healthy = condition.lower() in {
        "healthy",
        "health",
    }

    return {
        "class_name": class_name,

        "plant": plant,

        "condition": (
            "Healthy"
            if healthy
            else condition
        ),

        "status": (
            "Healthy"
            if healthy
            else "Disease"
        ),
    }


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(image_path):

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models, label_encoder = load_models()


    # --------------------------------------------------------
    # IMAGE PREPROCESSING
    # --------------------------------------------------------

    processed = preprocess_image(
        image_path
    )


    # --------------------------------------------------------
    # K-MEANS SEGMENTATION
    # --------------------------------------------------------

    segmentation = segment_leaf(
        processed["lab"]
    )


    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    feature_vector = extract_features(
        processed["lab"],
        processed["resized_rgb"],
        segmentation["leaf_mask"],
    )


    if feature_vector.shape != (12,):

        raise ValueError(
            f"Expected 12 features, "
            f"received {feature_vector.shape}"
        )


    # --------------------------------------------------------
    # PREDICTIONS FROM ALL MODELS
    # --------------------------------------------------------

    predictions = {}

    for model_name, (
        model,
        scaler,
    ) in models.items():

        # Scale when required
        if scaler is not None:

            X = scaler.transform(
                feature_vector.reshape(1, -1)
            )

        else:

            X = feature_vector.reshape(
                1,
                -1,
            )


        # Prediction
        encoded_prediction = model.predict(
            X
        )[0]


        class_name = label_encoder.inverse_transform(
            [encoded_prediction]
        )[0]


        # Probability
        probabilities = None
        confidence = None

        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = model.predict_proba(
                X
            )[0]

            confidence = float(
                np.max(probabilities)
            )


        # Parse plant / disease
        information = parse_class_name(
            str(class_name)
        )


        predictions[model_name] = {

            **information,

            "confidence": confidence,

            "probabilities": probabilities,

        }


    # --------------------------------------------------------
    # RETURN EVERYTHING TO STREAMLIT
    # --------------------------------------------------------

    return {

        # Original images
        "original_rgb":
            processed["original_rgb"],

        "processed_rgb":
            processed["resized_rgb"],


        # LAB image
        "lab":
            processed["lab"],


        # Segmentation
        "labels":
            segmentation["labels"],

        "centers":
            segmentation["centers"],

        "background_cluster":
            segmentation["background_cluster"],

        "segmented_rgb":
            segmentation["segmented_rgb"],

        "leaf_mask":
            segmentation["leaf_mask"],

        "masked_leaf":
            segmentation["masked_leaf"],


        # Features
        "feature_vector":
            feature_vector,

        "feature_names":
            FEATURE_COLUMNS,


        # All model predictions
        "predictions":
            predictions,

    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Plant leaf disease prediction"
    )

    parser.add_argument(
        "image",
        help="Path to test image",
    )

    args = parser.parse_args()


    result = predict_image(
        args.image
    )


    print()
    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)


    for model_name, prediction in result[
        "predictions"
    ].items():

        print()
        print(
            f"{model_name}:"
        )

        print(
            f"  Class     : "
            f"{prediction['class_name']}"
        )

        print(
            f"  Plant     : "
            f"{prediction['plant']}"
        )

        print(
            f"  Condition : "
            f"{prediction['condition']}"
        )

        if prediction["confidence"] is not None:

            print(
                f"  Confidence: "
                f"{prediction['confidence']:.2%}"
            )