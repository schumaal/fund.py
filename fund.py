import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():
    # Downloads yolov8n.pt automatically on first run
    return YOLO("yolov8n.pt") 

model = load_model()

# =========================================================
# UI SETUP
# =========================================================

st.set_page_config(page_title="YOLOv8 Detector", layout="wide")
st.title("🔎 YOLOv8 Objekt-Erkennung")

# Sidebar for settings
st.sidebar.header("Einstellungen")
conf_threshold = st.sidebar.slider("Konfidenz-Schwellenwert", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader(
    "Bild hochladen", 
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PROCESSING LOGIC
# =========================================================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # UI Columns: Left for Raw, Right for Detection
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📷 Original")
        st.image(image, use_container_width=True)

    # Run Inference
    # we pass the confidence threshold from the slider here
    results = model.predict(image, conf=conf_threshold)

    with col2:
        st.success("🎯 Ergebnis")
        
        # Plotting the results (boxes, labels, scores)
        # results[0].plot() returns a BGR numpy array
        res_plotted = results[0].plot()
        
        # Convert BGR (OpenCV format) to RGB for Streamlit
        st.image(res_plotted, channels="BGR", use_container_width=True)

    # =========================================================
    # STATISTICS
    # =========================================================
    
    st.subheader("📦 Gefundene Objekte Details:")
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        # Create a clean list of detections
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            st.write(f"✅ **{name}**: {conf*100:.1f}% Sicherheit")
    else:
        st.warning("Keine Objekte mit dem gewählten Schwellenwert erkannt.")
