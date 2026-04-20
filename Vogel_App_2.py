import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vogel-Erkennung Pro", layout="centered")
st.title("🐦 Vogel-Erkennung (bessere Version)")

# -------------------------
# MODELL LADEN
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.applications.EfficientNetB0(weights="imagenet")

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
# VOGEL-KLASSEN (ImageNet)
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
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild")

    processed = preprocess(image)
    prediction = model.predict(processed)

    # Top-5
    top_indices = prediction[0].argsort()[-5:][::-1]

    results = []
    for i in top_indices:
        label = labels[str(i)][1]
        confidence = float(prediction[0][i])
        results.append((label, confidence))

    # Debug anzeigen
    st.subheader("🔍 Top 5 Vorhersagen")
    for label, confidence in results:
        st.write(f"{label} – {round(confidence*100,2)}%")

    # Vogel-Erkennung
    found_bird = False

    for label, confidence in results:
        if any(word in label.lower() for word in bird_keywords):
            st.success(f"🐦 Vogel erkannt: {label} ({round(confidence*100,2)}%)")
            found_bird = True
            break

    if not found_bird:
        st.warning("❌ Kein Vogel erkannt")
