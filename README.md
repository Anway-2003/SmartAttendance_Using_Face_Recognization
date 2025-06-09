# SmartAttendance_Using_Face_Recognization

# SmartAttendance 📸🧠
A Smart Attendance System using Face Recognition built as a Final Year Project to automate and digitize attendance processes with the help of Computer Vision and Python.

## 💡 Project Overview
SmartAttendance is a real-time face recognition system that marks attendance by detecting and recognizing faces through a webcam. It stores attendance records in a MySQL database and provides features like voice feedback, duplicate entry prevention, unknown face blurring, and daily CSV reports.

---

## ✨ Features

- 🔍 **Face Detection & Recognition** (using OpenCV & Haar cascades)
- 🗃️ **MySQL Integration** for secure and scalable storage
- 📢 **Text-to-Speech Notifications** with gTTS
- 📸 **Unknown Face Blurring** for Privacy
- 🔄 **Duplicate Entry Avoidance** within a time window
- 🗂️ **Daily Attendance Logs** in CSV format
- 📊 **Live Dashboard** using Matplotlib
- 🖥️ **GUI Interface** (Tkinter)
- ⚙️ **Auto-Start on Boot (Windows Batch Script)**

---

## 🛠️ Tech Stack

| Technology        | Purpose                         |
|------------------|---------------------------------|
| Python           | Core programming language       |
| OpenCV           | Face detection & recognition    |
| gTTS             | Text-to-speech conversion       |
| MySQL            | Database for user & logs        |
| Tkinter          | GUI interface                   |
| Matplotlib       | Visual dashboard                |
| SQLite & CSV     | Backup logging & reports        |

---

## 📂 Folder Structure

SmartAttendance/
│
├── dataset/       # Collected face images
├── trainer/       # Model training script
├── recognizer/    # Trained model storage
├── haarcascade_frontalface_default.xml
├── train_model.py
├── recognize.py
└── README.md

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SmartAttendance.git
   cd SmartAttendance

2.Install requirements
pip install -r requirements.txt   

3.Train the model
python trainer/train_model.py

4.Start attendance recognition
python recognize.py

📝 Requirements

Python 3.x
OpenCV
matplotlib
Tkinter (comes preinstalled)

🛡️ Security Features

Blur unknown faces
Prevent duplicate attendance within defined window
Logs all entries with timestamp
Password-protected admin dashboard (optional)

