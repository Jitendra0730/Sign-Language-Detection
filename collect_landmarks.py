import cv2
import mediapipe as mp
import csv
import numpy as np

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define label for the current class (e.g., A, B, C)
label = "Z"  # Change this to the label you're collecting

# Output CSV file
file = open("sign_language_data.csv", "a", newline='')
csv_writer = csv.writer(file)

cap = cv2.VideoCapture(0)

print("Collecting data for:", label)
print("Press 's' to save the frame. Press 'q' to quit.")

while True:
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

            if cv2.waitKey(1) & 0xFF == ord('s'):
                csv_writer.writerow(landmarks + [label])
                print(f"Saved {label} sample")

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
