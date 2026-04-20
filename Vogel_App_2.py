import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Vogel-Erkennung Pro",
    page_icon="🐦",
    layout="centered"
)

# -------------------------
# STYLE (Custom CSS)
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    text-align: center;
}
.result-box {
    padding: 15px;
    border-radius: 12px;
    margin-top: 15px;
}
.success-box {
    background-color: #1f3d2b;
}
.warning-box {
    background-color: #3d1f1f;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("🐦 Vogel-Erkennung")
st.markdown("Lade ein Bild hoch und die KI sagt dir, ob ein Vogel drauf ist.")

# -------------------------
# MODELL LADEN
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.applications.EfficientNetB0(weights="imagenet")

with st.spinner("🔄 Lade KI-Modell..."):
    model = load_model()

# -------------------------
# LABELS
# -------------------------
@st.cache_resource
def load_labels():
    import json
    import urllib.request
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url) as f:
        return json.load(f)

labels = load_labels()

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# -------------------------
# VOGEL-LISTE
# -------------------------
bird_keywords = [
    "bird", "hen", "cock", "rooster", "ostrich", "brambling", "goldfinch",
    "junco", "indigo_bunting", "robin", "bulbul", "jay", "magpie",
    "chickadee", "water_ouzel", "kite", "bald_eagle", "vulture",
    "great_grey_owl", "partridge", "quail", "peacock", "flamingo",
    "macaw", "toucan", "pelican", "king_penguin", "albatross"
]

# -------------------------
# UPLOAD UI
# -------------------------
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="📸 Dein Bild", use_column_width=True)

    with st.spinner("🧠 Analysiere Bild..."):
        processed = preprocess(image)
        prediction = model.predict(processed)

    # Top 5
    top_indices = prediction[0].argsort()[-5:][::-1]

    results = []
    for i in top_indices:
        label = labels[str(i)][1]
        confidence = float(prediction[0][i])
        results.append((label, confidence))

    # -------------------------
    # RESULT CHECK
    # -------------------------
    found_bird = False

    for label, confidence in results:
        if any(word in label.lower() for word in bird_keywords):
            st.markdown(f"""
            <div class="result-box success-box">
                <h3>🐦 Vogel erkannt!</h3>
                <p><b>{label}</b><br>
                Sicherheit: {round(confidence*100,2)}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence * 100))
            found_bird = True
            break

    if not found_bird:
        st.markdown("""
        <div class="result-box warning-box">
            <h3>❌ Kein Vogel erkannt</h3>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # DETAILS (aufklappbar)
    # -------------------------
    with st.expander("🔍 Details anzeigen (Top 5)"):
        for label, confidence in results:
            st.write(f"{label} – {round(confidence*100,2)}%")
