"""Streamlit interface for Plant Leaf Disease Classification."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import (
    FEATURE_COLUMNS,
    K_COMPARISON_CSV,
    METRICS_JSON,
)
from src.predict import predict_image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Plant Leaf Disease Classification",
    page_icon="🌿",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("🌿 Plant Leaf Disease Classification")

st.caption("Machine Vision & Statistical Learning")

st.write(
    "A classical computer-vision and statistical-learning pipeline using "
    "LAB color features, K-Means segmentation, GLCM texture features, and KNN."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Project Information")

    st.metric("Classes", "38")
    st.metric("Features", "12")

    st.write("**Algorithm:** K-Nearest Neighbours (KNN)")
    st.write("**Segmentation:** K-Means (K=3)")
    st.write("**Feature space:** LAB color moments + GLCM")
    st.write("**Dataset:** Color images only")

    st.divider()

    st.subheader("Methodology")

    st.write(
        "Image → 256×256 → LAB → K-Means → leaf mask → "
        "9 color + 3 texture features → StandardScaler → KNN"
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded is not None:

    temp_path = None

    try:

        # -------------------------------------------------
        # Display the ORIGINAL uploaded image
        # -------------------------------------------------

        original_image = Image.open(uploaded).convert("RGB")

        st.subheader("Input Image")

        st.image(
            original_image,
            caption=f"Uploaded Image: {uploaded.name}",
            use_container_width=True,
        )

        # -------------------------------------------------
        # Save ORIGINAL uploaded bytes without
        # re-encoding through PIL.
        # -------------------------------------------------

        suffix = Path(uploaded.name).suffix.lower()

        if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_file.write(uploaded.getvalue())
            temp_path = Path(temp_file.name)

        # -------------------------------------------------
        # RUN THE EXACT SAME PREDICTION PIPELINE
        # -------------------------------------------------

        with st.spinner(
            "Processing image through preprocessing, "
            "segmentation, feature extraction and KNN..."
        ):

            result = predict_image(temp_path)

        st.success("Prediction completed successfully.")


        # =================================================
        # PROCESSING RESULTS
        # =================================================

        st.divider()

        st.subheader("Image Processing Pipeline")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(
                result["original_rgb"],
                caption="Original Image",
                use_container_width=True,
            )

        with c2:
            st.image(
                result["processed_rgb"],
                caption="Processed (256×256)",
                use_container_width=True,
            )

        with c3:
            st.image(
                result["segmented_rgb"],
                caption="K-Means Segmentation",
                use_container_width=True,
            )


        # -------------------------------------------------
        # MASK RESULTS
        # -------------------------------------------------

        c4, c5 = st.columns(2)

        with c4:
            st.image(
                result["leaf_mask"],
                caption="Leaf Mask",
                clamp=True,
                use_container_width=True,
            )

        with c5:
            st.image(
                result["masked_leaf"],
                caption="Masked Leaf",
                use_container_width=True,
            )


        # =================================================
        # PREDICTION RESULT
        # =================================================

        st.divider()

        st.subheader("🔍 Prediction Result")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "Plant",
                result["plant"],
            )

        with p2:
            st.metric(
                "Condition",
                result["condition"],
            )

        with p3:
            st.metric(
                "Status",
                result["status"],
            )

        st.info(
            f"**Predicted Class:** {result['class_name']}"
        )


        # =================================================
        # FEATURE VECTOR
        # =================================================

        st.divider()

        st.subheader("📊 12-Dimensional Feature Vector")

        feature_vector = result["feature_vector"]

        feature_df = pd.DataFrame(
            [feature_vector],
            columns=FEATURE_COLUMNS,
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
        )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.divider()

        st.subheader("🤖 Model Information")

        if METRICS_JSON.exists():

            metrics = json.loads(
                METRICS_JSON.read_text(
                    encoding="utf-8"
                )
            )

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    "Best K",
                    str(metrics["best_k"]),
                )

            with m2:
                st.metric(
                    "Accuracy",
                    f"{metrics['accuracy']:.2%}",
                )

            with m3:
                st.metric(
                    "Precision",
                    f"{metrics['precision_weighted']:.2%}",
                )

            with m4:
                st.metric(
                    "F1-score",
                    f"{metrics['f1_weighted']:.2%}",
                )

        else:

            st.warning(
                "Final evaluation metrics are not available yet. "
                "Train the model first."
            )


        # =================================================
        # DEBUG / VERIFICATION
        # =================================================

        with st.expander("Prediction Verification"):

            st.write(
                "The Streamlit interface uses the same "
                "`predict_image()` function as the command-line "
                "prediction system."
            )

            st.write(
                "**Predicted class returned by model:**"
            )

            st.code(
                result["class_name"]
            )

            st.write(
                "**Number of extracted features:**",
                len(result["feature_vector"]),
            )

            st.write(
                "**Feature order:**"
            )

            st.code(
                ", ".join(FEATURE_COLUMNS)
            )


    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )

        st.exception(exc)


    finally:

        # -------------------------------------------------
        # Remove temporary uploaded file
        # -------------------------------------------------

        if temp_path is not None:

            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


# =========================================================
# ABOUT PROJECT
# =========================================================

st.divider()

with st.expander("About the Project"):

    st.write(
        "This MSc project demonstrates the integration of "
        "unsupervised and supervised learning. K-Means performs "
        "unsupervised color-based image segmentation. Color moments "
        "and GLCM describe the segmented leaf. KNN then performs "
        "supervised 38-class plant health/disease classification."
    )

    st.write(
        "The project deliberately uses handcrafted computer-vision "
        "features rather than CNNs, transfer learning, TensorFlow, "
        "or PyTorch."
    )


# =========================================================
# K-VALUE EXPERIMENT
# =========================================================

if K_COMPARISON_CSV.exists():

    with st.expander("K-value Experiment"):

        st.dataframe(
            pd.read_csv(K_COMPARISON_CSV),
            use_container_width=True,
        )
