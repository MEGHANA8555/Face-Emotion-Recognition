import cv2
import numpy as np
import joblib

emotion_labels = ["Angry","Fear","Happy","Sad","Surprise","Neutral"]

model = joblib.load("emotion_model.pkl")


def predict_emotion(face_img):

    face_img = cv2.resize(face_img, (48,48))

    face_img = face_img / 255.0

    face_img = face_img.flatten().reshape(1,-1)

    prediction = model.predict(face_img)

    return emotion_labels[prediction[0]]