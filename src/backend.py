import os
import numpy as np
from typing import Dict
from tensorflow.keras.models import load_model
from utils import CLASSES, preprocess_for_model, jaundice_score, redness_score

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'facedx_effnetb3.h5')

class DummyModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = np.random.rand(6)
        probs = probs / probs.sum()
        return probs.reshape(1, -1)

def load_disease_model() -> object:
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            return model
        except Exception as e:
            print(f"Failed to load model at {MODEL_PATH}: {e}\nUsing DummyModel instead.")
    return DummyModel()

def fuse_with_heuristics(probs: np.ndarray, j_score: float, r_score: float, alpha: float = 0.4) -> np.ndarray:
    p = probs.copy().reshape(-1)
    p[4] = (1 - alpha) * p[4] + alpha * j_score
    p[1] = (1 - alpha) * p[1] + alpha * r_score * 0.6
    p[0] = (1 - alpha) * p[0] + alpha * r_score * 0.4
    p = p / p.sum()
    return p.reshape(1, -1)

def predict_from_roi(model, roi_bgr, img_size=224) -> Dict:
    x = preprocess_for_model(roi_bgr, img_size=img_size)
    probs = model.predict(x)
    j = jaundice_score(roi_bgr)
    r = redness_score(roi_bgr)
    probs = fuse_with_heuristics(probs, j, r, alpha=0.4)
    idx = int(np.argmax(probs))
    return {
        'class_idx': idx,
        'label': CLASSES[idx],
        'probs': probs.flatten().tolist(),
        'jaundice_score': float(j),
        'redness_score': float(r),
    }
