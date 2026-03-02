# ================================
# IMPORTS
# ================================
import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Blood Group Prediction",
    page_icon="🩸",
    layout="centered"
)

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    svm = joblib.load("svm_model.pkl")
    knn = joblib.load("knn_model.pkl")
    meta = joblib.load("meta_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return svm, knn, meta, label_encoder

svm_model, knn_model, meta_model, label_encoder = load_models()

# ================================
# IMAGE PREPROCESSING
# ================================
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.createCLAHE(2.0, (8, 8)).apply(img)
    return img

# ================================
# HOG FEATURE EXTRACTION
# ================================
def extract_features(img):
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features.reshape(1, -1)

# ================================
# UI
# ================================
st.title("🩸 Blood Group Prediction from Fingerprint")
st.write("Ensemble Model: SVM + KNN → Meta Classifier")

uploaded = st.file_uploader(
    "Upload Fingerprint Image",
    type=["jpg", "png", "jpeg", "bmp"]
)

if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Original Fingerprint", use_column_width=True)

    processed = preprocess_image(img)
    st.image(processed, caption="Processed Fingerprint", use_column_width=True)

    if st.button("Predict Blood Group"):

        with st.spinner("Analyzing fingerprint..."):

            # Extract HOG features
            features = extract_features(processed)

            # Base model probabilities
            svm_prob = svm_model.predict_proba(features)
            knn_prob = knn_model.predict_proba(features)

            # Combine base outputs
            stacked_features = np.concatenate(
                (svm_prob, knn_prob),
                axis=1
            )

            # Meta model prediction
            final_prob = meta_model.predict_proba(stacked_features)
            pred_class = np.argmax(final_prob)
            confidence = np.max(final_prob) * 100

            blood_group = label_encoder.inverse_transform([pred_class])[0]

        st.success(f"🩸 Predicted Blood Group: **{blood_group}**")
        st.metric("Prediction Confidence", f"{confidence:.2f}%")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("© 2026 Blood Group Prediction | Developed by Kishanjee")
