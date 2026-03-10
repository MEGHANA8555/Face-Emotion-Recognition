import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from utils.preprocess import load_dataset
from evaluation.metrics import evaluate_model


# Load images
images, labels = load_dataset("dataset/train")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy, precision, recall, f1, cm = evaluate_model(y_test, y_pred)

print("Accuracy:", accuracy)

joblib.dump(model, "emotion_model.pkl")

print("Model saved!")