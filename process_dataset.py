import os
import cv2
import numpy as np
import pickle
import mediapipe as mp

# Dataset directory
DATASET_DIR = "dataset"
OUTPUT_FILE = "data.pickle"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Prepare data storage
data = []
labels = []
label_map = {}

# Process each label (sign) in the dataset
for label_index, label in enumerate(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    print(f"Processing sign: {label} ({label_index})")
    label_map[label_index] = label  # Store label mapping

    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable images

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                feature_vector = []
                x_, y_ = [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    feature_vector.append(lm.x - min(x_))
                    feature_vector.append(lm.y - min(y_))

                data.append(feature_vector)
                labels.append(label_index)

# Save processed data
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'label_map': label_map}, f)

print(f"✅ Dataset preprocessing complete! Saved as '{OUTPUT_FILE}'")