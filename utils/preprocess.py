import os
import cv2
import numpy as np


def load_dataset(dataset_path):

    images = []
    labels = []

    emotions = os.listdir(dataset_path)

    for label, emotion in enumerate(emotions):

        emotion_path = os.path.join(dataset_path, emotion)

        for img_name in os.listdir(emotion_path):

            img_path = os.path.join(emotion_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (48,48))

            img = img / 255.0

            images.append(img.flatten())
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels