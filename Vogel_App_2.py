import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Vogel-Erkennung",
    page_icon="🐦",
    layout="centered"
)

# -------------------------
# STYLE (Instagram Look)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.insta-card {
    max-width: 420px;
    margin: auto;
    border-radius: 20px;
    overflow: hidden;
    background: #1a1d24;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.6);
}

.insta-header {
    padding: 12px;
    font-weight: bold;
    border-bottom: 1px solid #333;
    color: white;
}

.insta-image-container {
    position: relative;
}

.insta-image {
    width: 100%;
    display: block;
}

.overlay {
    position: absolute;
    bottom: 0;
    width: 100%;
    padding: 15px;
    background: linear-gradient(transparent, rgba(0,0,0,0.85));
    color: white;
}

.success {
    color: #4ade80;
    font-weight: bold;
    font-size: 18px;
}

.fail {
    color: #f87171;
    font-weight: bold;
    font-size: 18px;
}

.caption {
    font-size: 14px;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("🐦 Vogel-Erkennung")
st.markdown("Lade ein Bild hoch – wie eine Instagram Story ✨")

# -------------------------
# MODELL LADEN
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.applications.EfficientNetB0(weights="imagenet")

with st.spinner("🔄 KI wird geladen..."):
    model = load_model()

# -------------------------
# LABELS LADEN
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
# VOGEL KEYWORDS
# -------------------------
bird_keywords = [
    "bird", "hen", "cock", "rooster", "ostrich", "brambling", "goldfinch",
    "junco", "indigo_bunting", "robin", "bulbul", "jay", "magpie",
    "chickadee", "water_ouzel", "kite", "bald_eagle", "vulture",
    "great_grey_owl", "partridge", "quail", "peacock", "flamingo",
    "macaw", "toucan", "pelican", "king_penguin", "albatross"
]

# -------------------------
# UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🧠 Analysiere Bild..."):
        processed = preprocess(image)
        prediction = model.predict(processed)

    # Top-5
    top_indices = prediction[0].argsort()[-5:][::-1]

    results = []
    for i in top_indices:
        label = labels[str(i)][1]
        confidence = float(prediction[0][i])
        results.append((label, confidence))

    # -------------------------
    # VOGEL CHECK
    # -------------------------
    found_bird = False
    main_label = ""
    main_conf = 0

    for label, confidence in results:
        if any(word in label.lower() for word in bird_keywords):
            found_bird = True
            main_label = label
            main_conf = confidence
            break

    if not found_bird:
        main_label, main_conf = results[0]

    # -------------------------
    # IMAGE -> BASE64
    # -------------------------
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # -------------------------
    # INSTAGRAM CARD
    # -------------------------
    st.markdown(f"""
    <div class="insta-card">
        <div class="insta-header">📸 Vogel-KI Analyse</div>

        <div class="insta-image-container">
            <img src="data:image/jpeg;base64,{img_str}" class="insta-image"/>

            <div class="overlay">
                {"<div class='success'>🐦 Vogel erkannt</div>" if found_bird else "<div class='fail'>❌ Kein Vogel</div>"}
                <div class="caption">{main_label} ({round(main_conf*100,2)}%)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # DETAILS
    # -------------------------
    with st.expander("🔍 Top 5 Vorhersagen"):
        for label, confidence in results:
            st.write(f"{label} – {round(confidence*100,2)}%")
