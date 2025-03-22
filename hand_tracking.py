import cv2
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

# Function to determine if hand is a fist
def is_fist(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    thumb_base = landmarks[1]
    index_base = landmarks[5]
    middle_base = landmarks[9]
    ring_base = landmarks[13]
    pinky_base = landmarks[17]
    return (thumb_tip.y > thumb_base.y and
            index_tip.y > index_base.y and
            middle_tip.y > middle_base.y and
            ring_tip.y > ring_base.y and
            pinky_tip.y > pinky_base.y)

# Function to determine if hand is an open palm
def is_open_palm(landmarks):
    return (landmarks[4].y < landmarks[3].y and
            landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[9].y and
            landmarks[16].y < landmarks[13].y and
            landmarks[20].y < landmarks[17].y)

# Function to determine if hand is pointing (index finger extended)
def is_pointing(landmarks):
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    return (index_tip.y < landmarks[6].y and thumb_tip.y > landmarks[3].y)

# Function to determine if hand is a thumbs up
def is_thumbs_up(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    return (thumb_tip.y < landmarks[3].y and
            index_tip.y > landmarks[6].y and
            middle_tip.y > landmarks[9].y and
            ring_tip.y > landmarks[13].y and
            pinky_tip.y > landmarks[17].y)

# Function to determine if hand is a peace sign
def is_peace_sign(landmarks):
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    thumb_tip = landmarks[4]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    return (index_tip.y < landmarks[6].y and
            middle_tip.y < landmarks[9].y and
            thumb_tip.y > landmarks[3].y and
            ring_tip.y > landmarks[13].y and
            pinky_tip.y > landmarks[17].y)

# Function to determine if hand is an OK sign
def is_ok_sign(landmarks):
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    return (index_tip.x > thumb_tip.x and
            index_tip.y > thumb_tip.y)

# Function to determine if hand is rock on (pinky and index extended)
def is_rock_on(landmarks):
    index_tip = landmarks[8]
    pinky_tip = landmarks[20]
    return (index_tip.y < landmarks[6].y and
            pinky_tip.y < landmarks[17].y and
            landmarks[4].y > landmarks[3].y and
            landmarks[12].y > landmarks[9].y and
            landmarks[16].y > landmarks[13].y)

# Open webcam
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check for gestures and display messages
            if is_fist(hand_landmarks.landmark):
                cv2.putText(frame, "Fist Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_open_palm(hand_landmarks.landmark):
                cv2.putText(frame, "Open Palm Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_pointing(hand_landmarks.landmark):
                cv2.putText(frame, "Pointing Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif is_thumbs_up(hand_landmarks.landmark):
                cv2.putText(frame, "Thumbs Up Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif is_peace_sign(hand_landmarks.landmark):
                cv2.putText(frame, "Peace Sign Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            elif is_ok_sign(hand_landmarks.landmark):
                cv2.putText(frame, "OK Sign Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif is_rock_on(hand_landmarks.landmark):
                cv2.putText(frame, "Rock On Gesture Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

    # Show the video output
    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
