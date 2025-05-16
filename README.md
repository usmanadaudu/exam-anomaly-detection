# 🕵️‍♂️ Examination Proctoring Aid System

A light-weight proctoring system based on computer vision to detect anomalies during examinations. The system uses facial landmark tracking (MediaPipe), head pose estimation (MediaPipe) and object detection (YOLO) to monitorstudent behaviour and flag anomalous activities like:

- Looking left or right
- Speaking
- Presence of mobile phones

---

## 🚀 Features

| Feature                            | Emoji | Description                                     |
| ---------------------------------- | ----- | ----------------------------------------------- |
| Real-time face tracking            | 👤📍  | Tracks students face using MediaPipe           | 
| Gaze direction detection           | 👀🧭  | Detects if the student looks left/right        |
| Bounding box color-coded alerts    | 🟥🟩  | Red = Anomaly, Green = Normal                        |
| Mobile phone detection (YOLO)      | 📱🎯  | Detects mobile phones                          |
| Works on CPU                       | 🧠💻  | Runs efficiently without GPU                   |
| Multiple face detection            | 👥❗   | Detects presence of more than one face        |
| Alert system (future)              | 🚨🔔  | Trigger warnings on suspicious activity        |
| Video recording/logging            | 🎥📝  | Record examination sessions for review         |

## Installation
