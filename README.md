
#  Sign Language Detection & Translation (A–Z)

A real-time hand sign recognition app that detects alphabets (A–Z) using webcam input and a trained machine learning model. It includes a Tkinter GUI, voice feedback, and sentence assembly from recognized gestures.

---

## 📸 Demo Preview

![Screenshot](./screenshot.png)  
*Live webcam feed, prediction display, and Speak/Clear/Start/Stop buttons.*

---

## 💡 Features

- ✅ Real-time **hand sign recognition** (A–Z) using MediaPipe
- 🔊 **Text-to-speech** for predicted alphabets
- 🖥️ **Tkinter GUI** with:
  - Start/Stop Webcam
  - Speak Prediction
  - Clear Assembled Sentence
- 🧠 Machine Learning Model trained on 21 hand landmarks (x, y) = 42 features

---

## 🛠️ Technologies Used

| Module       | Purpose                            |
|--------------|-------------------------------------|
| MediaPipe     | Hand landmark detection             |
| OpenCV        | Webcam access and image processing  |
| Tkinter       | GUI creation                        |
| Joblib        | Model persistence                   |
| NumPy         | Data processing                     |
| pyttsx3       | Text-to-speech                      |

---

## 📁 Project Structure

```
Sign-Language-Detection/
├── collect_data.py           # Collects hand landmark data
├── training.py               # Trains the model
├── sign_language_model.pkl   # Trained model and LabelEncoder
├── sign_language_data.csv    # Dataset with landmarks and labels
├── a_to_z_prediction.py      # Command-line prediction script
├── prediction.py             # GUI with voice and sentence features
├── screenshot.png            # App screenshot
```

---

## 🧪 How to Use

### 1. Install Dependencies

```bash
pip install opencv-python mediapipe numpy pyttsx3 joblib
```

### 2. Run the GUI Application

```bash
python prediction.py
```
