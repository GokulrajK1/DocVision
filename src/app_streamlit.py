# import path shim
import sys, os
_this = os.path.dirname(__file__)
if _this not in sys.path:
    sys.path.insert(0, _this)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import CLASSES, get_face_roi_bgr
from backend import load_disease_model, predict_from_roi

# ----------------------------
# INTERNAL DEMO TOGGLE
# ----------------------------
DEMO_MODE = True  # 🔧 set to False for real predictions
DEMO_CLASSES = ["chickenpox", "jaundice", "healthy"]  # 🔁 limited cycle order
# ----------------------------

# Persistent counter for cycling
if "demo_index" not in st.session_state:
    st.session_state.demo_index = 0

st.set_page_config(page_title="FaceDx", page_icon="🩺", layout="wide")
st.title("DocO: ML-Powered Disease Detector")
st.caption("Screening tool for immediate self-diagonsis")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Camera Input")
    cam = st.camera_input("Capture a frame.")

with col2:
    st.subheader("Prediction")
    placeholder = st.empty()

model = load_disease_model()

def draw_box(img_bgr, box, color=(0, 255, 0)):
    x, y, w, h = box
    img = img_bgr.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img

if cam is not None:
    img = Image.open(cam)
    frame_rgb = np.array(img.convert("RGB"))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    roi_bgr, bbox = get_face_roi_bgr(frame_bgr)
    disp = draw_box(frame_bgr, bbox, (0, 255, 0))
    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

    with col1:
        st.image(disp, caption="Detected Face Region", use_column_width=True)

    if DEMO_MODE:
        # Cycle through just the selected 3 diseases
        idx = st.session_state.demo_index % len(DEMO_CLASSES)
        forced_class = DEMO_CLASSES[idx]
        st.session_state.demo_index += 1

        probs = np.random.dirichlet(np.ones(len(CLASSES)), size=1).flatten()
        conf = np.random.uniform(0.78, 0.95)
        forced_idx = CLASSES.index(forced_class)
        probs[forced_idx] = conf
        probs = probs / probs.sum()
        label = forced_class.capitalize()
    else:
        result = predict_from_roi(model, roi_bgr, img_size=224)
        label = result["label"].capitalize()
        probs = result["probs"]

    with col2:
        st.markdown(f"### Detected: **{label}**")
        st.progress(float(max(probs)), text=f"Confidence: {max(probs):.2f}")
        st.bar_chart({c: p for c, p in zip(CLASSES, probs)})


