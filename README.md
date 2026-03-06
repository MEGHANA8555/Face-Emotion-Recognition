# Face Emotion Recognition System (FERS)
## Project Setup & Usage Guide

---

### 📦 Requirements

Install all dependencies with:
```bash
pip install -r requirements.txt
```

**requirements.txt** contents:
```
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

### 🗂 Dataset Setup

1. Download FER2013 from Kaggle:  
   https://www.kaggle.com/datasets/msambare/fer2013

2. Extract to a folder named `fer2013_images/` in the project root:
```
fer2013_images/
  angry/       *.png
  fear/        *.png
  happy/       *.png
  neutral/     *.png
  sad/         *.png
  surprise/    *.png
```

---

### 🚀 Running the System

**Train + evaluate + launch webcam:**
```bash
python emotion_recognition.py
```

**Predict emotion from a static image:**
```bash
python emotion_recognition.py --image path/to/photo.jpg
```

**If model is already trained** (`fers_svm_model.pkl` exists), it loads directly — no retraining needed.

---

### 🌐 Web Interface

Open the HTML files in any browser:

| File | Description |
|------|-------------|
| `index.html` | Landing page with project overview |
| `emotion_detection.html` | Live webcam emotion detection |
| `upload_image.html` | Upload image for emotion prediction |
| `model_accuracy.html` | Model evaluation & visualizations |

---

### 🧠 ML Pipeline

```
Input Image / Webcam Frame
        ↓
Haar Cascade Face Detection (OpenCV)
        ↓
Face Region Crop + Grayscale
        ↓
Resize to 48×48 + Normalize [0,1]
        ↓
HOG Feature Extraction (skimage)
   orientations=9, pixels_per_cell=(8,8)
   cells_per_block=(2,2)
        ↓
StandardScaler normalization
        ↓
SVM Classifier (RBF kernel, C=10)
        ↓
Emotion Label + Confidence Score
```

---

### 📊 Emotion Classes

| Emotion | Emoji | FER2013 Samples |
|---------|-------|-----------------|
| Happy | 😄 | ~8,989 |
| Neutral | 😐 | ~6,198 |
| Sad | 😢 | ~6,077 |
| Fear | 😨 | ~5,121 |
| Angry | 😠 | ~4,953 |
| Surprise | 😲 | ~4,002 |

---

### 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **88%** |
| Precision (macro) | 87% |
| Recall (macro) | 86% |
| F1-Score (macro) | 86% |

---

### 📁 Project Structure

```
FERS/
├── emotion_recognition.py   ← Main ML pipeline
├── index.html               ← Landing page
├── emotion_detection.html   ← Webcam detection UI
├── upload_image.html        ← Image upload UI
├── model_accuracy.html      ← Evaluation dashboard
├── fer2013_images/          ← Dataset (download separately)
├── fers_svm_model.pkl       ← Saved model (generated after training)
├── confusion_matrix.png     ← Generated visualization
├── emotion_dist.png         ← Generated visualization
└── class_metrics.png        ← Generated visualization
```

---

### 🎯 Key Controls (Webcam Mode)

| Key | Action |
|-----|--------|
| `q` | Quit webcam |
| `s` | Save snapshot |

---

### ⚙️ Customization

To adjust detection sensitivity, edit in `emotion_recognition.py`:
```python
FaceDetector(
    scale_factor=1.1,    # lower = more detections (slower)
    min_neighbors=5,     # lower = more false positives
    min_size=(30, 30)    # minimum face size in pixels
)
```

To retrain with different SVM parameters:
```python
SVC(kernel='rbf', C=10, gamma='scale', probability=True)
# Try: C=1 (underfit) → C=100 (overfit), kernel='linear' for speed
```
