# 🛡️ Drowsiness Detection

**Real-Time Eye Monitoring with Smart Alerting using Python & OpenCV**

> A computer vision project that tracks facial landmarks in real time to detect signs of drowsiness and trigger an audible alarm — built using OpenCV, dlib, and Pygame.

---

## 🚀 Project Overview

This project is a real-time **drowsiness detection system** designed to alert users when prolonged eye closure is detected — a key indicator of fatigue. It is particularly useful for applications like **driver monitoring systems**, **workplace fatigue detection**, and **smart productivity tools**.

Using a webcam feed and facial landmark detection, the system calculates the **Eye Aspect Ratio (EAR)** to determine whether the eyes are closing beyond a safe threshold. If drowsiness is detected, an audible alarm is played to alert the user.

---

## 💡 Key Features

✅ Real-time face and eye detection using webcam  
✅ 68-point facial landmark tracking (via dlib)  
✅ Eye Aspect Ratio (EAR) based drowsiness detection  
✅ Audio alarm system using Pygame  
✅ Visual feedback via OpenCV window  
✅ Threaded alarm triggering to avoid frame drops

---

## 🧠 Technologies Used

| Tech            | Purpose                                 |
|-----------------|-----------------------------------------|
| Python 3.x      | Core programming language               |
| OpenCV          | Real-time video capture and processing  |
| dlib            | Facial detection and landmark prediction|
| imutils         | Helper functions for image processing   |
| Pygame          | Audio alarm playback                    |
| Scipy           | Euclidean distance calculation          |
| Threading       | Async alarm trigger without freezing    |

---

## Download Required Model File
This project uses the `shape_predictor_68_face_landmarks.dat` file from the [dlib project](http://dlib.net/).

🔗 Download from: [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

After downloading, extract and place it in the root folder of the project.
