# FaceDx — Real-Time Facial Disease Detector

FaceDx is a hackathon-ready project that detects visible facial conditions (chickenpox, measles, acne, eczema, jaundice, healthy) in near-real-time using a webcam and a deep learning model (EfficientNetB3).

## Quick Start (Demo without a trained model)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app_streamlit.py
```

The app uses a **fallback heuristic** if `models/facedx_effnetb3.h5` is missing, so you can demo UI right away.

## Train
Prepare folders under `data/` per class, then:
```bash
python src/train.py --data_dir data --epochs 10 --img_size 224 --batch_size 16
```

## CLI
```bash
python src/infer.py --image path/to/photo.jpg
```

## Disclaimer
Screening tool for educational use only. Not a diagnostic device.
