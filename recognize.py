import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Load face cascade and recognizer
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Load name labels
with open("trainer/names.pkl", "rb") as f:
    names = pickle.load(f)

# Attendance log to avoid duplicates
attendance_log = {}

def mark_attendance(id, name):
    global attendance_log
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    csv_path = f'attendance/{date_str}.csv'

    if not os.path.exists('attendance'):
        os.makedirs('attendance')

    # Load or create DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['ID', 'Name', 'Time'])

    # Check attendance_log for current session duplicates
    if id in attendance_log:
        last_seen = attendance_log[id]
        if now - last_seen < timedelta(minutes=10):
            return  # Ignore duplicate in current session

    # Check CSV file for duplicates in last 10 minutes
    recent_entries = df[(df['ID'] == id)]
    if not recent_entries.empty:
        last_time_str = recent_entries['Time'].values[-1]
        last_time = datetime.strptime(last_time_str, '%H:%M:%S')
        
        # Combine date and time to get full datetime for comparison
        last_datetime = datetime.strptime(f"{date_str} {last_time_str}", '%Y-%m-%d %H:%M:%S')

        if now - last_datetime < timedelta(minutes=10):
            return  # Ignore duplicate in file

    # Mark attendance
    attendance_log[id] = now
    new_row = {'ID': id, 'Name': name, 'Time': time_str}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Attendance marked for {name} at {time_str}")



def run_recognition():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)

            if conf < 70:
                name = names.get(id_, "Unknown")
                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                mark_attendance(id_, name)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
