import cv2
import numpy as np

CLASSES = ['chickenpox', 'measles', 'acne', 'eczema', 'jaundice', 'healthy']

def get_face_roi_bgr(frame_bgr, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    if len(faces) == 0:
        h, w = frame_bgr.shape[:2]
        return frame_bgr, (0,0,w,h)
    areas = [w*h for (x,y,w,h) in faces]
    i = int(np.argmax(areas))
    (x, y, w, h) = faces[i]
    roi = frame_bgr[max(y-10,0):y+h+10, max(x-10,0):x+w+10]
    return roi, (x, y, w, h)

def preprocess_for_model(roi_bgr, img_size=224):
    img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def jaundice_score(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 50, 50])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(mask.mean() / 255.0)

def redness_score(roi_bgr):
    b,g,r = cv2.split(roi_bgr.astype(np.float32))
    redness = (r - (g+b)/2.0)
    redness = np.clip(redness, 0, None)
    return float(redness.mean() / 255.0)
