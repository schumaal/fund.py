import streamlit as st
from ultralytics import YOLO
from PIL import Image

# =========================================================
# MODEL (YOLOv8)
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # kleines, schnelles Modell

model = load_model()

# =========================================================
# UI
# =========================================================

st.title("🔎 YOLOv8 Objekt-Erkennung (Streamlit)")

st.write("Lade ein Bild hoch und lasse KI Objekte erkennen.")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Dein Bild", use_container_width=True)

    st.write("🔄 Analysiere Bild...")

    results = model(image)

    st.subheader("📦 Erkannte Objekte:")

    found_any = False

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]

            st.write(f"👉 {label} ({confidence*100:.1f}%)")
            found_any = True

    if not found_any:
        st.warning("Keine Objekte erkannt.")
