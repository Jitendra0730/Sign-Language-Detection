import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

# Load trained model and label encoder
model, label_encoder = joblib.load("sign_language_model.pkl")

# Text-to-speech setup
engine = pyttsx3.init()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

print("📸 Real-time Sign Language Prediction Started (press Q to quit, S to speak)")

last_prediction = ""

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract X and Y only (42 values)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            if len(landmarks) == 42:
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                last_prediction = predicted_label

                # Show prediction on screen
                cv2.putText(frame, f'Prediction: {predicted_label}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Speak if 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and last_prediction:
        engine.say(last_prediction)
        engine.runAndWait()

    # Quit
    if key == ord('q'):
        break

    cv2.imshow("Sign Language Live Prediction", frame)

cap.release()
cv2.destroyAllWindows()
