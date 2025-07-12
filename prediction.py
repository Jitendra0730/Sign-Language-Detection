import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pyttsx3
from threading import Thread

# Load the trained model and label encoder
model, label_encoder = joblib.load("sign_language_model.pkl")

# Text-to-speech setup
engine = pyttsx3.init()

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global variables
cap = None
running = False
last_prediction = ""
predicted_sentence = ""

# Start camera and prediction
def start_prediction():
    global cap, running, last_prediction, predicted_sentence
    cap = cv2.VideoCapture(0)
    running = True
    while running:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                if len(landmarks) == 42:
                    input_data = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(input_data)
                    predicted_label = label_encoder.inverse_transform(prediction)[0]
                    last_prediction = predicted_label
                    prediction_label.config(text=f"Predicted: {predicted_label}")

        cv2.imshow("Live Camera - Press Q in window to close", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_prediction()
            break

    cap.release()
    cv2.destroyAllWindows()

# Speak function
def speak_prediction():
    global last_prediction
    if last_prediction:
        engine.say(last_prediction)
        engine.runAndWait()
        update_sentence(last_prediction)

# Stop camera

def stop_prediction():
    global running
    running = False

# Add to sentence

def update_sentence(char):
    global predicted_sentence
    predicted_sentence += char
    sentence_text.set(predicted_sentence)

# Clear sentence
def clear_sentence():
    global predicted_sentence
    predicted_sentence = ""
    sentence_text.set("")

# Start thread
def start_thread():
    Thread(target=start_prediction).start()

# GUI setup
root = tk.Tk()
root.title("Hand Sign Detection Model")
root.geometry("500x400")
root.configure(bg="#e6f2ff")

heading = tk.Label(root, text="Hand Sign Detection Model", font=("Helvetica", 18, "bold"), bg="#e6f2ff")
heading.pack(pady=20)

btn_frame = tk.Frame(root, bg="#e6f2ff")
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="Start", command=start_thread, bg="#4CAF50", fg="white", width=10)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(btn_frame, text="Stop", command=stop_prediction, bg="#f44336", fg="white", width=10)
stop_btn.grid(row=0, column=1, padx=10)

speak_btn = tk.Button(btn_frame, text="Speak", command=speak_prediction, bg="#008CBA", fg="white", width=10)
speak_btn.grid(row=0, column=2, padx=10)

prediction_label = tk.Label(root, text="Predicted: ", font=("Helvetica", 14), bg="#e6f2ff")
prediction_label.pack(pady=10)

sentence_text = tk.StringVar()
sentence_entry = tk.Entry(root, textvariable=sentence_text, font=("Helvetica", 14), width=30, justify="center")
sentence_entry.pack(pady=10)

clear_btn = tk.Button(root, text="Clear", command=clear_sentence, bg="#ff9900", fg="white", width=10)
clear_btn.pack(pady=10)

root.mainloop()