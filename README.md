# Plant Leaf Disease Classification Using Machine Vision and Statistical Learning

## 1. Project Overview

This MSc Computer Science project implements a classical computer-vision and statistical-machine-learning system for **38-class plant leaf health/disease classification**.

The system intentionally does not use CNNs, ResNet, EfficientNet, MobileNet, YOLO, TensorFlow, PyTorch, transfer learning, or other deep-learning approaches.

### Pipeline

```text
Input Leaf Image
      ↓
Image Loading
      ↓
Resize to 256 × 256
      ↓
RGB → LAB
      ↓
K-Means Segmentation (K=3)
      ↓
Border-based Background Detection
      ↓
Leaf Mask
      ↓
Color Moments (9)
      +
GLCM Texture (3)
      ↓
12-Dimensional Feature Vector
      ↓
StandardScaler
      ↓
KNN (Euclidean Distance)
      ↓
38-Class Prediction
      ↓
Plant + Condition + Health Status
      ↓
Streamlit
```

## 2. Objectives

- Build a reproducible classical computer-vision pipeline.
- Segment leaf regions using K-Means.
- Preserve disease-related color regions instead of assuming the leaf is green.
- Extract LAB color moments and GLCM texture descriptors.
- Produce exactly 12 numerical features.
- Train and compare KNN models for multiple K values.
- Select K using cross-validation on the training set only.
- Evaluate on a held-out stratified test set.
- Persist the model, scaler, and label encoder.
- Provide a Streamlit demonstration interface.

## 3. Technologies

- Python
- OpenCV
- NumPy
- SciPy
- scikit-image
- scikit-learn
- pandas
- matplotlib
- joblib
- Streamlit

## 4. Dataset Setup

Only the **color** dataset is used as the primary source.

Place it here:

```text
dataset/
└── color/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── ...
    └── Tomato___healthy/
```

The implementation expects 38 class folders for the final experiment.

Do not combine color, grayscale, and supplied segmented copies as independent training samples. Grayscale is generated internally for GLCM.

## 5. Installation

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Check the Dataset

```bash
python main.py dataset-info
```

For the final project, verify that the class count is 38 and the image counts look reasonable.

## 7. Test the Pipeline Before Full Processing

Start with 10 images:

```bash
python main.py features --max-images 10 --no-resume
```

Then test a larger subset:

```bash
python main.py features --max-images 100 --no-resume
```

Inspect generated:

```text
features/leaf_features.csv
```

The CSV contains:

```text
image
class
L_mean
L_variance
L_skewness
a_mean
a_variance
a_skewness
b_mean
b_variance
b_skewness
contrast
correlation
entropy
```

That is **14 columns total**, including image and class.

## 8. Full Feature Extraction

After the small tests are successful:

```bash
python main.py features
```

The extraction is sequential and does not load the complete image dataset into RAM.

The feature generator:

- continues after individual image failures;
- reports progress;
- writes rows incrementally;
- can resume from an existing CSV;
- records successful and failed images in its terminal output.

For a very large dataset, this step can take a substantial amount of time because K-Means is deliberately performed for each image.

## 9. Training

```bash
python main.py train
```

The training process:

1. loads the feature CSV;
2. separates X and y;
3. label-encodes the class names;
4. performs an 80/20 stratified split;
5. fits `StandardScaler` only on the training data;
6. performs 5-fold cross-validation on the training set;
7. compares K = 1, 3, 5, 7, 9, 11, 13, 15, 17, 19;
8. chooses the best K using training CV accuracy;
9. fits the final KNN on the complete scaled training set;
10. evaluates once on the held-out test set;
11. saves the artifacts.

### Data-leakage protection

Correct:

```text
Full data
   ↓
Train/Test split
   ↓
Fit scaler on training only
   ↓
Transform train and test
   ↓
CV on training only
   ↓
Select K
   ↓
Final test evaluation
```

The test set is not used to tune K.

## 10. Generated Model Files

Training creates:

```text
models/
├── knn_model.pkl
├── scaler.pkl
└── label_encoder.pkl
```

Results are written to:

```text
results/
├── k_comparison.csv
├── classification_report.txt
├── confusion_matrix.png
└── metrics.json
```

Actual experimental values must come from your run. The project does not fabricate accuracy, precision, recall, F1, best K, or image-processing counts.

## 11. Command-Line Prediction

After training:

```bash
python main.py predict path/to/leaf.jpg
```

The prediction pipeline repeats the same processing used during feature generation:

```text
Image
 ↓
Resize
 ↓
RGB → LAB
 ↓
K-Means
 ↓
Leaf Mask
 ↓
9 Color Features + 3 GLCM Features
 ↓
StandardScaler
 ↓
KNN
 ↓
LabelEncoder
 ↓
Original Class Name
```

## 12. Streamlit Application

Run:

```bash
streamlit run app.py
```

The application provides:

- image uploader;
- original image;
- 256×256 processed image;
- K-Means segmentation;
- leaf mask;
- masked leaf;
- predicted class;
- plant name;
- condition/disease;
- healthy/disease status;
- 12 feature values;
- selected K;
- actual evaluation metrics;
- K-value comparison;
- project methodology.

## 13. Human-Readable Classification

For:

```text
Tomato___Early_blight
```

the UI reports:

```text
Predicted Class: Tomato___Early_blight
Plant: Tomato
Condition: Early blight
Status: Disease
```

For:

```text
Tomato___healthy
```

the status is:

```text
Healthy
```

The 38 categories are not described as 38 diseases because the dataset includes healthy categories.

## 14. Feature Extraction

### Color Moments

LAB channels:

- L*
- a*
- b*

For each channel:

- mean;
- variance;
- skewness.

Therefore:

```text
3 channels × 3 statistics = 9 features
```

### Texture

Grayscale is generated internally from the resized RGB image.

GLCM configuration:

```text
distance = 1
angle = 0°
```

The project extracts:

- contrast;
- correlation;
- entropy.

Therefore:

```text
9 + 3 = 12 features
```

## 15. K-Means Segmentation

K-Means is an **unsupervised learning** method used here for image region separation.

The implementation uses:

- K = 3;
- OpenCV `cv2.kmeans`;
- K-Means++ initialization;
- multiple attempts;
- deterministic OpenCV RNG seed;
- border-pixel analysis.

Cluster IDs are arbitrary. The code therefore does not assume that cluster 0, 1, or 2 is always the leaf.

Instead, the cluster that dominates the image border is treated as background, and the other regions are retained as leaf candidates. Morphological operations and connected-component filtering clean the mask.

Disease colors such as brown, yellow, dark, gray, and reddish regions are not removed merely because they are not green.

## 16. Academic Interpretation

### Unsupervised learning

**K-Means**

Purpose:

> Image segmentation / region separation.

### Supervised learning

**KNN**

Purpose:

> 38-class plant leaf health/disease classification.

K-Means does not diagnose disease. KNN performs classification using the handcrafted features.

## 17. Project Structure

```text
Plant Disease Project/
│
├── dataset/
│   └── color/
│
├── features/
│   └── leaf_features.csv
│
├── models/
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── results/
│   ├── k_comparison.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── metrics.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── features.py
│   ├── create_features.py
│   ├── train.py
│   └── predict.py
│
├── tests/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## 18. Testing Strategy

### Stage 1 — 10 images

```bash
python main.py features --max-images 10 --no-resume
```

Check:

- all rows have 12 numeric features;
- no NaN/Inf values;
- segmentation can be visually inspected;
- CSV is generated.

### Stage 2 — 100 images

```bash
python main.py features --max-images 100 --no-resume
```

Then:

```bash
python main.py train
```

This validates the complete ML pipeline on a small dataset.

### Stage 3 — Full dataset

```bash
python main.py features
python main.py train
streamlit run app.py
```

## 19. Visual Segmentation Validation

Before full-scale extraction, inspect several representative images using the prediction/Streamlit interface.

The important questions are:

- Is the background removed?
- Is most of the leaf retained?
- Are damaged/discolored areas retained?
- Is the mask stable across different backgrounds?

A successful code execution alone is not evidence that segmentation is visually correct.

## 20. Troubleshooting

### `Dataset directory does not exist`

Make sure the dataset is at:

```text
dataset/color/
```

and that each class is a subdirectory.

### Wrong number of classes

The final experiment expects 38 class folders. Use:

```bash
python main.py dataset-info
```

to inspect the dataset.

### Feature CSV missing

Run:

```bash
python main.py features
```

before training.

### Model files missing

Run:

```bash
python main.py train
```

before prediction or Streamlit inference.

### OpenCV cannot decode an image

Check that the image is not corrupted and uses one of:

```text
jpg jpeg png bmp webp
```

### Extraction is slow

This is expected for a large dataset because K-Means is run per image. Use the 10/100-image workflow first. Full extraction should only begin after the pipeline has been validated.

### PowerShell activation is blocked

If local policy prevents virtual-environment activation, use Command Prompt or adjust the PowerShell execution policy according to your system's security policy.

## 21. Viva Questions to Understand

Be prepared to explain:

1. Why resize the image?
2. Why use LAB rather than RGB?
3. Why use K-Means?
4. Why identify background using border pixels?
5. Why not assume a fixed K-Means cluster ID is the leaf?
6. Why use color moments?
7. Why use GLCM?
8. Why exactly 12 features?
9. Why StandardScaler?
10. Why KNN?
11. Why experiment with different K values?
12. Why use stratified splitting?
13. Why keep a separate test set?
14. Why use cross-validation?
15. Why is K-Means unsupervised?
16. Why is KNN supervised?
17. Why must the grayscale/segmented copies not be treated as independent samples?
18. How is data leakage avoided?
19. Why use handcrafted features instead of a CNN?
20. Why use Streamlit?

## 22. Important Academic Limitation

The feature pipeline and KNN classifier depend strongly on the quality and consistency of segmentation and handcrafted descriptors. Performance may vary with illumination, camera conditions, background complexity, cultivar differences, and disease appearance.

The final report should present actual measured results from the final dataset run rather than a value copied from a reference.

## 23. Reproducibility

The main random state is:

```text
random_state = 42
```

K-Means also uses the fixed OpenCV random seed defined in `src/config.py`.

All important paths are project-relative using `pathlib.Path`.

## 24. Final Verification Checklist

Before submission, confirm:

- [ ] 38 final class folders are present.
- [ ] Only the color dataset is used as the primary dataset.
- [ ] Images are resized to 256×256.
- [ ] RGB is converted to LAB.
- [ ] K-Means uses K=3.
- [ ] Border-based background detection is used.
- [ ] Disease-related color regions are not removed simply for being non-green.
- [ ] Exactly 12 numerical features are generated.
- [ ] Feature order is identical in training and prediction.
- [ ] Train/test split uses 80/20.
- [ ] `random_state=42`.
- [ ] Split is stratified.
- [ ] StandardScaler is fitted only on training data.
- [ ] K is selected using training-only CV.
- [ ] Test data is reserved for final evaluation.
- [ ] KNN uses Euclidean distance.
- [ ] K values 1 through 19 in the specified list are tested.
- [ ] Accuracy, precision, recall, F1, and confusion matrix are generated.
- [ ] Model, scaler, and label encoder are persisted.
- [ ] Streamlit can load the saved artifacts.
- [ ] No experimental result is fabricated.
#   P l a n t - L e a f - D i s e a s e - D e t e c t i o n - S y s t e m  
 