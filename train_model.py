import cv2
import os
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = 'dataset'  # folder where subfolders with images exist

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# Train the recognizer
recognizer.train(x_train, np.array(y_labels))

# Save the trained model
os.makedirs('trainer', exist_ok=True)
recognizer.save('trainer/trainer.yml')

# Save the label dictionary (id:name)
with open('trainer/names.pkl', 'wb') as f:
    pickle.dump({v:k for k,v in label_ids.items()}, f)

print("Training done and files saved.")
