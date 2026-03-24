<<<<<<< HEAD
# Smart Classroom System

This project is a smart classroom system using face recognition for attendance.

## Features
- Face recognition attendance
- Real-time detection
- Student data management

## Tech Stack
- Python
- OpenCV
- Face Recognition

## How to Run
```bash
pip install -r requirements.txt
python main.py
=======
# Smart Classroom Attendance System 🎓

An AI-powered smart classroom attendance tracking system using Streamlit, OpenCV, MTCNN, and FaceNet.

## Features
- **Face Detection & Recognition**: Utilizes MTCNN for robust face detection and FaceNet for high-accuracy facial embeddings.
- **Live Attendance Dashboard**: A modern Streamlit interface with real-time camera feed and live metrics.
- **Automated Logging & Reporting**: Automatically records present individuals and allows downloading attendance reports in Excel and CSV format.
- **Sidebar Controls**: Adjustable recognition threshold mapping and easy navigation across log history.

## Setup Instructions

### 1. Requirements
Ensure you have Python 3.8+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Database
Store images of the students/individuals you want the system to recognize. 
Run the training script to generate the embeddings (`encodings.pkl`):
```bash
python train_system.py
```

### 3. Run the App
Start the Streamlit Web Application:
```bash
streamlit run app.py
```

## System Structure
- `app.py`: Main Streamlit web application with modern CSS and metric dashboard.
- `train_system.py`: Script to process images and create facial embeddings based on MTCNN+FaceNet pipeline.
- `requirements.txt`: Required Python dependencies.
- `.gitignore`: Setup to ignore ML artifacts (`encodings.pkl`) and local attendance logs (`*.xlsx`, `*.csv`).
>>>>>>> 393dbe7 (chore: Initialize Smart Classroom Attendance System repository)
