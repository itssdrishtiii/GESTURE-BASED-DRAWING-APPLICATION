# 🖐️ Gesture-Based Drawing Application

A real-time computer vision project that allows users to draw on a virtual canvas using hand gestures instead of a mouse or touchscreen. Built using Python, OpenCV, and MediaPipe, this application tracks hand movements through a webcam and converts gestures into drawing actions.

## 🚀 Features
- Real-time hand gesture recognition
- Draw using index finger movement
- Erase using multi-finger gestures
- Color selection tool
- Draw shapes (circle, rectangle, line, etc.)
- Undo / Redo functionality
- Save drawing as image
- Live webcam preview
- Interactive virtual canvas interface

## 🛠️ Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

## 📁 Project Structure
Gesture-based-drawing-application/
│── gesture_app.py
│── run.bat
│── requirements.txt
│── .gitignore
│── README.md

## ⚙️ Installation

### 1. Clone Repository
git clone https://github.com/itssdrishtiii/Gesture-based-drawing-application.git

cd Gesture-based-drawing-application

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run Project
python gesture_app.py

OR double click:
run.bat

## 📦 Required Modules
- opencv-python
- mediapipe
- numpy
- protobuf==3.20.3

## 🎯 How It Works
- Webcam captures live video feed
- MediaPipe detects hand landmarks
- Finger positions are analyzed
- One finger gesture = Draw
- Multiple finger gesture = Erase / Select tools
- Drawing appears on virtual canvas in real time

## 🧠 Key Learnings
- Computer Vision fundamentals
- Hand Tracking with MediaPipe
- Real-time gesture recognition
- Interactive UI development
- Problem-solving with Python

## 👩‍💻 Author
Drishti  
Aspiring Software Engineer | QA | AI Enthusiast

## 📌 Note
This project was developed for learning, innovation, and practical implementation of AI-powered human-computer interaction.
