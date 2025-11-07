import argparse, os, cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import CLASSES, get_face_roi_bgr, preprocess_for_model

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'facedx_effnetb3.h5')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}. Train first with src/train.py.")

    model = load_model(MODEL_PATH)
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit("Could not read image.")
    roi, _ = get_face_roi_bgr(bgr)
    x = preprocess_for_model(roi, img_size=224)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    print("Prediction:", CLASSES[idx], "| Confidence:", float(probs[idx]))
    for i, c in enumerate(CLASSES):
        print(f"{c:>10s}: {probs[i]:.4f}")

if __name__ == '__main__':
    main()
