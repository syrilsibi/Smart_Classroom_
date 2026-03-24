import cv2
import os
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# CONFIGURATION
DATASET_PATH = r"D:\New folder\SKILLPARK\SMART_CLASSROOOM\PROJECT\Dataset"
SAVE_PATH = r"D:\New folder\SKILLPARK\SMART_CLASSROOOM\encodings.pkl"

print("📂 Loading Models...")
detector = MTCNN()
embedder = FaceNet()

known_encodings = []
known_names = []

if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset folder not found at {DATASET_PATH}")
    exit()

print("🚀 Starting Encoding Process...")
for student_name in os.listdir(DATASET_PATH):
    student_dir = os.path.join(DATASET_PATH, student_name)
    if not os.path.isdir(student_dir): continue

    for img_name in os.listdir(student_dir):
        img_path = os.path.join(student_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img_rgb.shape

        try:
            results = detector.detect_faces(img_rgb)
            if results:
                x, y, w, h = results[0]['box']
                # Coordinate Clipping Fix
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x1 + w), min(h_img, y1 + h)
                
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    face = img_rgb[y1:y2, x1:x2]
                    face = cv2.resize(face, (160, 160))
                    face = np.expand_dims(face, axis=0)
                    
                    encoding = embedder.embeddings(face)[0]
                    known_encodings.append(encoding)
                    known_names.append(student_name)
                    print(f"✅ Encoded: {student_name}")
        except Exception:
            continue 

# SAVE DATABASE
if known_encodings:
    with open(SAVE_PATH, "wb") as f:
        pickle.dump({"encodings": np.array(known_encodings), "names": known_names}, f)
    print(f"🏁 Finished! Database saved to {SAVE_PATH}")