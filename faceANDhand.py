import cv2
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)
previous_time = 0
current_time = 0

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (800, 600))
    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.flags.writeable = False
    results = holistic_model.process(Image)
    Image.flags.writeable = True
    Image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        Image, results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(0, 255, 255),
                               thickness=1,
                               circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 255),
                               thickness=1,
                               circle_radius=1))
    mp_drawing.draw_landmarks(Image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(Image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(Image, str(int(fps)) + "FPS", (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Face and Hand landmarks", Image)
    if cv2.waitKey(5) & 0xFF == ord('b'):
        break

capture.release()
cv2.destroyWindow()
for landmark in mp_holistic.HandLandmark:
    print(landmark, landmark.value)

print(mp_holistic.HandLandmark.WRIST.value)
