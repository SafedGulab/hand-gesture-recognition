import cv2
import mediapipe as mp
import numpy as np

mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize hands
hands = mp_hand.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Finger tip IDs
Tip_ids = [4, 8, 12, 16, 20]

# Count fingers
def count_fingers(hand_landmarks, hand_label):
    fingers = []

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[Tip_ids[0]].x < hand_landmarks.landmark[Tip_ids[0]-1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if hand_landmarks.landmark[Tip_ids[0]].x > hand_landmarks.landmark[Tip_ids[0]-1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other fingers
    for tip_id in Tip_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Detect gestures
def get_gestures(fingers):
    total = sum(fingers)

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 1, 0, 0, 0]:
        return "Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [0, 1, 0, 0, 1]:
        return "Rock"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [0, 0, 1, 0, 0]:
        return "Middle Finger"
    else:
        return f"{total} Fingers"

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting Hand Gesture Recognition... Press 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to access camera")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Get hand label safely
            hand_label = "Unknown"
            if results.multi_handedness:
                hand_label = results.multi_handedness[idx].classification[0].label

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hand.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Process fingers
            fingers = count_fingers(hand_landmarks, hand_label)
            gesture = get_gestures(fingers)
            total_fingers = sum(fingers)

            # Wrist position
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)

            # Display gesture
            cv2.putText(frame, f"{hand_label}: {gesture}",
                        (cx - 60, cy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            # Display finger count
            cv2.putText(frame, f"Fingers: {total_fingers}",
                        (cx - 60, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2)

    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program Ended")