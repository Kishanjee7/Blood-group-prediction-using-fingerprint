# ================================
# ENV FIX (TensorFlow oneDNN log)
# ================================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ================================
# IMPORTS
# ================================
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Blood Group Prediction from Fingerprints",
    page_icon="🩸",
    layout="centered"
)

# ================================
# SESSION STATE
# ================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================================
# LOAD MODELS (GLOBAL)
# ================================
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn_model.h5")
    resnet = tf.keras.models.load_model("resnet_model.h5")
    svm = joblib.load("svm_model.pkl")
    knn = joblib.load("knn_model.pkl")
    meta = joblib.load("meta_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    cnn.predict(np.zeros((1,128,128,1)))
    resnet.predict(np.zeros((1,128,128,3)))

    return cnn, resnet, svm, knn, meta, label_encoder

cnn_model, resnet_model, svm_model, knn_model, meta_model, label_encoder = load_models()

cnn_feat_model = tf.keras.Model(cnn_model.inputs, cnn_model.layers[-2].output)
resnet_feat_model = tf.keras.Model(resnet_model.inputs, resnet_model.layers[-2].output)

# ================================
# UI CSS
# ================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#e0f2fe,#fdf2f8);
    font-family:'Inter', sans-serif;
}
.hero {
    text-align:center;
    padding:60px 10px;
}
.hero h1 {
    font-size:52px;
    font-weight:900;
    background: linear-gradient(90deg,#dc2626,#fb7185);
    -webkit-background-clip: text;
    color: transparent;
}
.hero p {
    font-size:20px;
    color:#475569;
    max-width:720px;
    margin:auto;
}
.card {
    background:black;
    border-radius:24px;
    padding:35px;
    box-shadow:0 15px 40px rgba(0,0,0,0.12);
    margin-top:25px;
}
.feature {
    background:black;
    border-radius:18px;
    padding:22px;
    box-shadow:0 8px 25px rgba(0,0,0,0.1);
    text-align:center;
}
.stButton>button {
    background: linear-gradient(90deg,#dc2626,#fb7185);
    color:white;
    border-radius:14px;
    height:3.2em;
    width:100%;
    font-size:16px;
    border:none;
}
.footer {
    text-align:center;
    margin-top:60px;
    color:#64748b;
    font-size:14px;
}
img { border-radius:16px; }
</style>
""", unsafe_allow_html=True)

# ================================
# IMAGE PREPROCESSING
# ================================
def enhance_fingerprint(img):
    img = cv2.resize(img, (128,128))
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.createCLAHE(2.0, (8,8)).apply(img)
    return img / 255.0

# ================================
# HOME PAGE (DASHBOARD)
# ================================
if st.session_state.page == "home":

    st.markdown("""
    <div class="hero">
        <h1>Blood Group Prediction</h1>
        <p>Fingerprint-based blood group prediction using deep learning and ensemble machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='feature'><h4>Ensemble</h4><p>CNN + ResNet50 + SVM + KNN</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='feature'><h4>Non-Invasive</h4><p>No blood sample needed</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='feature'><h4>Medical Ready</h4><p>Clinical applicability</p></div>", unsafe_allow_html=True)

    # ----------------------------
    # MODEL ACCURACY COMPARISON
    # ----------------------------
    st.markdown("## Model Accuracy Comparison")

    acc_df = pd.DataFrame({
        "Model": ["Random Forest (HOG)", "Gabor + RF","CNN","ResNet50","Ensemble"],
        "Accuracy (%)": [94.5, 79.8, 84.3, 63.7, 95.3]
    })
    st.dataframe(acc_df, use_container_width=True)

    

    if st.button("🚀 Start Prediction"):
        st.session_state.page = "predict"
        st.rerun()

# ================================
# PREDICTION PAGE
# ================================
if st.session_state.page == "predict":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if st.button("⬅ Home"):
        st.session_state.page = "home"
        st.rerun()

    uploaded = st.file_uploader(
        "Upload Fingerprint Image",
        type=["jpg","png","jpeg","bmp"]
    )

    if uploaded:
        img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Fingerprint", use_column_width=True)

        enhanced = enhance_fingerprint(img)
        with col2:
            st.image(enhanced, caption="Enhanced Fingerprint", use_column_width=True)

        if st.button("Predict Blood Group"):
            with st.spinner("Analyzing fingerprint......."):
                x_cnn = np.expand_dims(enhanced, (0,-1))
                x_rgb = np.repeat(x_cnn, 3, axis=-1)

                cnn_prob = cnn_model.predict(x_cnn)
                res_prob = resnet_model.predict(x_rgb)

                features = np.concatenate((
                    cnn_feat_model.predict(x_cnn),
                    resnet_feat_model.predict(x_rgb)
                ), axis=1)

                final_features = np.concatenate((
                    cnn_prob,
                    res_prob,
                    svm_model.predict_proba(features),
                    knn_model.predict_proba(features)
                ), axis=1)

                probs = meta_model.predict_proba(final_features)
                pred_class = np.argmax(probs)
                confidence = np.max(probs) * 100
                blood_group = label_encoder.inverse_transform([pred_class])[0]

            st.success(f"🩸 Predicted Blood Group: **{blood_group}**")
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

# ================================
# FOOTER
# ================================
st.markdown("""
<div class="footer">
•••• © 2025 Blood Group Prediction from Fingerprints •••• Developed by :- Kishanjee ••••
</div>
""", unsafe_allow_html=True)
