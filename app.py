import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import (
    FEATURE_COLUMNS,
    MODEL_COMPARISON_JSON,
    KNN_CONFUSION_MATRIX,
    RANDOM_FOREST_CONFUSION_MATRIX,
    LINEAR_SVM_CONFUSION_MATRIX,
    RBF_SVM_CONFUSION_MATRIX,
)
from src.predict import predict_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🌿 Plant Leaf Disease Detection")
st.caption(
    "Classical Computer Vision + Machine Learning"
)

st.write(
    "Upload a plant leaf image to identify the plant condition "
    "using the trained machine-learning models."
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.subheader("📷 Upload Leaf Image")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)


if uploaded is not None:

    temp_path = None

    try:

        # ----------------------------------------------------
        # Input image
        # ----------------------------------------------------

        original_image = Image.open(uploaded).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                original_image,
                caption=uploaded.name,
                use_container_width=True,
            )

        with right:

            st.write("")

            predict_button = st.button(
                "🔍 Predict Disease",
                type="primary",
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if predict_button:

            suffix = Path(uploaded.name).suffix.lower()

            if suffix not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                suffix = ".jpg"

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:

                temp_file.write(uploaded.getvalue())
                temp_path = Path(temp_file.name)

            with st.spinner(
                "Processing image and predicting disease..."
            ):

                result = predict_image(temp_path)

            predictions = result["predictions"]

            # ------------------------------------------------
            # Determine evaluation-best model
            # ------------------------------------------------

            best_model = "RBF SVM"
            best_accuracy = 0.8342

            if MODEL_COMPARISON_JSON.exists():

                try:

                    comparison_data = json.loads(
                        MODEL_COMPARISON_JSON.read_text(
                            encoding="utf-8"
                        )
                    )

                    best_model = comparison_data.get(
                        "best_model",
                        "RBF SVM",
                    )

                    best_accuracy = comparison_data.get(
                        "best_accuracy",
                        0.8342,
                    )

                except Exception:
                    pass

            best_prediction = predictions[best_model]

            # =================================================
            # MAIN RESULT
            # =================================================

            st.divider()
            st.subheader("🌿 Prediction Result")

            result_box = st.container(border=True)

            with result_box:

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Plant",
                        best_prediction["plant"],
                    )

                with c2:
                    st.metric(
                        "Condition",
                        best_prediction["condition"],
                    )

                with c3:
                    st.metric(
                        "Status",
                        best_prediction["status"],
                    )

                st.markdown(
                    f"**Predicted Class:** "
                    f"`{best_prediction['class_name']}`"
                )

                confidence = best_prediction["confidence"]

                if confidence is not None:

                    st.progress(
                        min(max(confidence, 0.0), 1.0)
                    )

                    st.write(
                        f"Model confidence: "
                        f"**{confidence:.2%}**"
                    )

                st.caption(
                    f"Final model: {best_model} "
                    f"(test accuracy: {best_accuracy:.2%})"
                )

            # =================================================
            # OTHER MODEL PREDICTIONS
            # =================================================

            with st.expander(
                "📊 Compare Predictions from All Models"
            ):

                rows = []

                for model_name, prediction in predictions.items():

                    rows.append(
                        {
                            "Model": model_name,
                            "Predicted Class":
                                prediction["class_name"],
                            "Plant":
                                prediction["plant"],
                            "Condition":
                                prediction["condition"],
                            "Confidence":
                                (
                                    f"{prediction['confidence']:.2%}"
                                    if prediction["confidence"]
                                    is not None
                                    else "N/A"
                                ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

            # =================================================
            # IMAGE PROCESSING
            # =================================================

            with st.expander(
                "🔍 View Image Processing Pipeline"
            ):

                st.write(
                    "The uploaded image passes through the "
                    "following preprocessing and segmentation steps."
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.image(
                        result["original_rgb"],
                        caption="Original Image",
                        use_container_width=True,
                    )

                with c2:
                    st.image(
                        result["processed_rgb"],
                        caption="Resized — 256 × 256",
                        use_container_width=True,
                    )

                st.markdown("**RGB → LAB**")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.image(
                        result["lab"][:, :, 0],
                        caption="L* Channel",
                        clamp=True,
                        use_container_width=True,
                    )

                with c2:
                    st.image(
                        result["lab"][:, :, 1],
                        caption="a* Channel",
                        clamp=True,
                        use_container_width=True,
                    )

                with c3:
                    st.image(
                        result["lab"][:, :, 2],
                        caption="b* Channel",
                        clamp=True,
                        use_container_width=True,
                    )

                st.markdown("**K-Means Segmentation**")

                c1, c2 = st.columns(2)

                with c1:
                    st.image(
                        result["segmented_rgb"],
                        caption="Segmented Image",
                        use_container_width=True,
                    )

                with c2:
                    st.image(
                        result["labels"],
                        caption="Cluster Map",
                        clamp=True,
                        use_container_width=True,
                    )

                st.caption(
                    f"K = 3 | Background cluster = "
                    f"{result['background_cluster']}"
                )

                st.markdown("**Leaf Extraction**")

                c1, c2 = st.columns(2)

                with c1:
                    st.image(
                        result["leaf_mask"],
                        caption="Leaf Mask",
                        clamp=True,
                        use_container_width=True,
                    )

                with c2:
                    st.image(
                        result["masked_leaf"],
                        caption="Masked Leaf",
                        use_container_width=True,
                    )

            # =================================================
            # FEATURES
            # =================================================

            with st.expander(
                "🧬 View Extracted Features"
            ):

                feature_df = pd.DataFrame(
                    {
                        "Feature": FEATURE_COLUMNS,
                        "Value": result["feature_vector"],
                    }
                )

                feature_df["Value"] = (
                    feature_df["Value"]
                    .astype(float)
                    .round(6)
                )

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "9 LAB color-moment features + "
                    "3 GLCM texture features = 12 features."
                )

            # =================================================
            # MODEL EVALUATION
            # =================================================

            with st.expander(
                "📈 View Model Evaluation"
            ):

                if MODEL_COMPARISON_JSON.exists():

                    try:

                        comparison_data = json.loads(
                            MODEL_COMPARISON_JSON.read_text(
                                encoding="utf-8"
                            )
                        )

                        rows = []

                        for model_name, values in (
                            comparison_data["models"].items()
                        ):

                            rows.append(
                                {
                                    "Model": model_name,
                                    "CV Accuracy":
                                        f"{values['cv_accuracy']:.2%}",
                                    "Test Accuracy":
                                        f"{values['accuracy']:.2%}",
                                    "Precision":
                                        f"{values['precision']:.2%}",
                                    "Recall":
                                        f"{values['recall']:.2%}",
                                    "F1-score":
                                        f"{values['f1_score']:.2%}",
                                }
                            )

                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.success(
                            f"Evaluation-best model: "
                            f"{comparison_data['best_model']} — "
                            f"{comparison_data['best_accuracy']:.2%} "
                            f"test accuracy"
                        )

                    except Exception as exc:

                        st.warning(
                            f"Could not load model evaluation: {exc}"
                        )

                else:

                    st.info(
                        "Model comparison results are not available."
                    )

            # =================================================
            # CONFUSION MATRICES
            # =================================================

            with st.expander(
                "📊 View Confusion Matrices"
            ):

                confusion_files = {
                    "KNN": KNN_CONFUSION_MATRIX,
                    "Random Forest":
                        RANDOM_FOREST_CONFUSION_MATRIX,
                    "Linear SVM":
                        LINEAR_SVM_CONFUSION_MATRIX,
                    "RBF SVM":
                        RBF_SVM_CONFUSION_MATRIX,
                }

                tabs = st.tabs(
                    list(confusion_files.keys())
                )

                for tab, (
                    model_name,
                    path,
                ) in zip(
                    tabs,
                    confusion_files.items(),
                ):

                    with tab:

                        if path.exists():

                            st.image(
                                str(path),
                                caption=(
                                    f"{model_name} "
                                    "Confusion Matrix"
                                ),
                                use_container_width=True,
                            )

                        else:

                            st.warning(
                                "Confusion matrix not found."
                            )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )

        st.exception(exc)

    finally:

        if temp_path is not None:

            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


# ============================================================
# FOOTER / ABOUT
# ============================================================

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This project classifies plant leaf diseases using
        classical computer vision and machine learning.

        Pipeline:
        Image → Resize → LAB → K-Means Segmentation
        → Leaf Mask → 12 Features → ML Classification
        """
    )

    st.write(
        """
        Models evaluated:
        KNN, Random Forest, Linear SVM and RBF SVM.

        The current evaluation identifies RBF SVM as the
        best-performing model with 83.42% test accuracy.
        """
    )
