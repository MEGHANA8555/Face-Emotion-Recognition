"""
Face Emotion Recognition System (FERS)
=======================================
Full ML pipeline using OpenCV + SVM on FER2013

Structure:
  1. Dataset loading & preprocessing
  2. Feature extraction (HOG)
  3. SVM model training
  4. Evaluation (accuracy, precision, recall, F1, confusion matrix)
  5. Real-time webcam detection
  6. Static image emotion prediction
  7. Visualization (confusion matrix, accuracy charts, distribution)
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
IMG_SIZE     = 48                            # FER2013 standard
EMOTIONS     = ['angry','fear','happy','neutral','sad','surprise']
EMOTION_EMOJI = {'angry':'😠','fear':'😨','happy':'😄',
                 'neutral':'😐','sad':'😢','surprise':'😲'}
DATA_DIR     = "fer2013_images"              # path to extracted dataset
MODEL_PATH   = "fers_svm_model.pkl"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# HOG parameters
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS    = 9


# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────
def load_fer2013_images(data_dir: str):
    """
    Load FER2013 images from folder structure:
      data_dir/
        angry/   *.png
        fear/    *.png
        happy/   *.png
        ...
    Returns X (images as numpy arrays), y (emotion labels)
    """
    X, y = [], []
    print(f"\n[LOAD] Reading dataset from: {data_dir}")
    for emotion in EMOTIONS:
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_path):
            print(f"  ⚠ Missing folder: {emotion_path}")
            continue
        files = [f for f in os.listdir(emotion_path) if f.endswith(('.png','.jpg','.jpeg'))]
        for fname in files:
            img_path = os.path.join(emotion_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = preprocess_face(img)
            X.append(img)
            y.append(emotion)
        print(f"  ✓ {emotion:10s} — {len(files)} images loaded")
    print(f"\n[LOAD] Total: {len(X)} samples across {len(set(y))} classes")
    return np.array(X), np.array(y)


def preprocess_face(img: np.ndarray) -> np.ndarray:
    """
    Preprocess a face image for feature extraction:
      1. Convert to grayscale (if not already)
      2. Resize to 48×48
      3. Normalize pixel values to [0, 1]
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0    # normalize [0, 1]
    return img


# ─────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION — HOG
# ─────────────────────────────────────────────────────────────
def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) features
    from a batch of 48×48 grayscale images.
    """
    features = []
    for img in images:
        feat = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            feature_vector=True
        )
        features.append(feat)
    return np.array(features)


# ─────────────────────────────────────────────────────────────
# 3. MODEL TRAINING — SVM
# ─────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Build sklearn Pipeline:
      StandardScaler → SVM (RBF kernel)
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        ))
    ])


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    print("\n[TRAIN] Building SVM pipeline (StandardScaler + SVM RBF)...")
    model = build_pipeline()
    print("[TRAIN] Fitting model — this may take a few minutes on large datasets...")
    model.fit(X_train, y_train)
    print("[TRAIN] ✓ Training complete!")
    return model


def save_model(model, label_encoder, path=MODEL_PATH):
    joblib.dump({'model': model, 'le': label_encoder}, path)
    print(f"[SAVE] Model saved → {path}")


def load_model(path=MODEL_PATH):
    data = joblib.load(path)
    print(f"[LOAD] Model loaded ← {path}")
    return data['model'], data['le']


# ─────────────────────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, label_names):
    """
    Full evaluation:
      - Accuracy Score
      - Classification Report (Precision, Recall, F1)
      - Confusion Matrix
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "═"*60)
    print("  MODEL EVALUATION REPORT")
    print("═"*60)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("═"*60)
    print(classification_report(y_test, y_pred, target_names=label_names))

    return y_pred, acc


# ─────────────────────────────────────────────────────────────
# 5. FACE DETECTION — Haar Cascade
# ─────────────────────────────────────────────────────────────
class FaceDetector:
    """OpenCV Haar Cascade face detector wrapper."""

    def __init__(self, cascade_path=CASCADE_PATH, scale_factor=1.1,
                 min_neighbors=5, min_size=(30, 30)):
        self.detector    = cv2.CascadeClassifier(cascade_path)
        self.scale       = scale_factor
        self.min_nb      = min_neighbors
        self.min_size    = min_size

    def detect(self, frame: np.ndarray):
        """
        Detect faces in frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor  = self.scale,
            minNeighbors = self.min_nb,
            minSize      = self.min_size
        )
        return faces if len(faces) else []

    def extract_face(self, frame: np.ndarray, bbox):
        """Extract & preprocess a face region given (x,y,w,h)."""
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        face = gray[y:y+h, x:x+w]
        return preprocess_face(face)


# ─────────────────────────────────────────────────────────────
# 6. EMOTION PREDICTOR
# ─────────────────────────────────────────────────────────────
class EmotionPredictor:
    """Wraps the trained model for single-image prediction."""

    COLORS = {
        'happy':    (0, 212, 255),    # cyan-yellow
        'sad':      (255, 165, 50),   # blue-orange
        'angry':    (50, 80, 248),    # red
        'surprise': (180, 80, 255),   # purple
        'fear':     (50, 215, 130),   # green
        'neutral':  (180, 180, 180),  # grey
    }

    def __init__(self, model, label_encoder):
        self.model = model
        self.le    = label_encoder

    def predict(self, face_img: np.ndarray):
        """
        Predict emotion from a preprocessed 48×48 face image.
        Returns (emotion_str, confidence_float, proba_dict)
        """
        feat   = extract_hog_features([face_img])
        proba  = self.model.predict_proba(feat)[0]
        idx    = np.argmax(proba)
        label  = self.le.inverse_transform([idx])[0]
        conf   = proba[idx]
        proba_dict = {self.le.inverse_transform([i])[0]: proba[i]
                      for i in range(len(proba))}
        return label, conf, proba_dict

    def annotate_frame(self, frame: np.ndarray, bbox, emotion: str, conf: float):
        """
        Draw bounding box and emotion label on frame.
        """
        x, y, w, h = bbox
        color = self.COLORS.get(emotion, (0, 212, 255))
        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Corner brackets
        cs = 16
        for (px,py,dx,dy) in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(frame, (px,py),(px+dx*cs,py), color, 3)
            cv2.line(frame, (px,py),(px,py+dy*cs), color, 3)
        # Label background
        label_txt = f"{EMOTION_EMOJI.get(emotion,'')} {emotion.capitalize()}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x, y-30), (x+tw+12, y), color, -1)
        cv2.putText(frame, label_txt, (x+6, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        return frame


# ─────────────────────────────────────────────────────────────
# 7. REAL-TIME WEBCAM DETECTION
# ─────────────────────────────────────────────────────────────
def run_webcam_detection(predictor: EmotionPredictor, detector: FaceDetector):
    """
    Open webcam feed, detect faces, predict & display emotions in real-time.
    Press 'q' to quit, 's' to save screenshot.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("\n[WEBCAM] Starting live detection — press 'q' to quit, 's' to snapshot")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror
        faces = detector.detect(frame)
        frame_count += 1

        for bbox in faces:
            face_img = detector.extract_face(frame, bbox)
            emotion, conf, _ = predictor.predict(face_img)
            frame = predictor.annotate_frame(frame, bbox, emotion, conf)

        # HUD
        hud = f"Faces: {len(faces)}  Frame: {frame_count}  Model: SVM"
        cv2.putText(frame, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,229,255), 1)

        cv2.imshow("FERS — Face Emotion Recognition  [q: quit | s: snapshot]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"snapshot_{frame_count}.png"
            cv2.imwrite(fname, frame)
            print(f"[SNAP] Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[WEBCAM] Session ended.")


# ─────────────────────────────────────────────────────────────
# 8. STATIC IMAGE PREDICTION
# ─────────────────────────────────────────────────────────────
def predict_image(image_path: str, predictor: EmotionPredictor,
                  detector: FaceDetector, show=True):
    """
    Predict emotion from a static image file.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    faces = detector.detect(frame)
    print(f"\n[IMAGE] {image_path}")
    print(f"  Faces detected: {len(faces)}")

    for i, bbox in enumerate(faces):
        face_img          = detector.extract_face(frame, bbox)
        emotion, conf, proba = predictor.predict(face_img)
        frame             = predictor.annotate_frame(frame, bbox, emotion, conf)
        print(f"\n  Face #{i+1}:")
        print(f"    Detected emotion : {EMOTION_EMOJI.get(emotion,'')} {emotion.capitalize()}")
        print(f"    Confidence       : {conf*100:.1f}%")
        print("    All probabilities:")
        for em, p in sorted(proba.items(), key=lambda x:-x[1]):
            bar = '█' * int(p * 30)
            print(f"      {em:10s} {p*100:5.1f}%  {bar}")

    if show:
        cv2.imshow(f"FERS — {os.path.basename(image_path)}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return frame


# ─────────────────────────────────────────────────────────────
# 9. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """Seaborn heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title('Confusion Matrix — FERS SVM Model', fontsize=14, pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Confusion matrix saved → {save_path}")
    plt.show()


def plot_emotion_distribution(y, title="Emotion Distribution", save_path="emotion_dist.png"):
    """Bar chart of class distribution."""
    from collections import Counter
    counts = Counter(y)
    labels = [e for e in EMOTIONS if e in counts]
    values = [counts[e] for e in labels]
    colors = ['#fbbf24','#60a5fa','#f87171','#a78bfa','#34d399','#94a3b8']
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor='white', linewidth=0.5)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('Number of Samples')
    ax.set_facecolor('#0d1220'); fig.patch.set_facecolor('#050810')
    ax.tick_params(colors='white'); ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white'); ax.title.set_color('white')
    for spine in ax.spines.values(): spine.set_edgecolor('#1e2940')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Distribution saved → {save_path}")
    plt.show()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path="class_metrics.png"):
    """Grouped bar chart: Precision, Recall, F1 per class."""
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=class_names)
    x = np.arange(len(class_names)); width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, p, width, label='Precision', color='#00e5ff', alpha=0.85)
    ax.bar(x,         r, width, label='Recall',    color='#a855f7', alpha=0.85)
    ax.bar(x + width, f, width, label='F1-Score',  color='#ff4d8d', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylim(0.6, 1.0); ax.legend()
    ax.set_title('Per-Class Precision / Recall / F1', fontsize=13)
    ax.set_xlabel('Emotion'); ax.set_ylabel('Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Metrics chart saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  Face Emotion Recognition System (FERS)      ║")
    print("║  SVM + HOG + OpenCV Haar Cascade             ║")
    print("╚══════════════════════════════════════════════╝")

    # ── 1. Load or train model ──
    if os.path.exists(MODEL_PATH):
        print(f"\n[INFO] Found existing model: {MODEL_PATH}")
        model, le = load_model()
    else:
        # Load & preprocess dataset
        X, y_raw = load_fer2013_images(DATA_DIR)
        if len(X) == 0:
            print("\n[ERROR] No images found! Ensure FER2013 dataset is at:", DATA_DIR)
            print("        Download: https://www.kaggle.com/datasets/msambare/fer2013")
            sys.exit(1)

        # Encode labels
        le = LabelEncoder()
        y  = le.fit_transform(y_raw)

        # Extract features
        print("\n[FEATURES] Extracting HOG features...")
        X_feat = extract_hog_features(X)
        print(f"[FEATURES] Feature vector size: {X_feat.shape[1]}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[SPLIT] Train: {len(X_train)}  Test: {len(X_test)}")

        # Train
        model = train_model(X_train, y_train)
        save_model(model, le)

        # ── 2. Evaluate ──
        y_pred, acc = evaluate_model(
            model, X_test, y_test, le.classes_
        )

        # ── 3. Visualize ──
        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)
        plot_confusion_matrix(y_test_labels, y_pred_labels, EMOTIONS)
        plot_emotion_distribution(y_raw)
        plot_per_class_metrics(y_test_labels, y_pred_labels, EMOTIONS)

    # ── 4. Setup detector & predictor ──
    detector  = FaceDetector()
    predictor = EmotionPredictor(model, le)

    # ── 5. Run webcam or image ──
    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        img_path = sys.argv[2] if len(sys.argv) > 2 else "test.jpg"
        predict_image(img_path, predictor, detector)
    else:
        run_webcam_detection(predictor, detector)


if __name__ == "__main__":
    main()
