import cv2
import mediapipe as mp
import numpy as np

def detect_angry_face():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results_face = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                # Get landmarks for eyebrows and mouth
                left_eyebrow_upper = [face_landmarks.landmark[i] for i in [65, 66, 67, 68, 69, 70, 107]]
                right_eyebrow_upper = [face_landmarks.landmark[i] for i in [295, 296, 297, 298, 299, 300, 336]]
                mouth_upper = [face_landmarks.landmark[i] for i in [13, 14, 17, 39, 37, 40, 181, 82, 81, 80, 191]]
                mouth_lower = [face_landmarks.landmark[i] for i in [87, 84, 17, 314, 317, 402, 318, 415, 310, 311, 312]]

                # Calculate eyebrow and mouth distances
                left_eyebrow_dist = np.mean([abs(left_eyebrow_upper[0].y - left_eyebrow_upper[-1].y)])
                right_eyebrow_dist = np.mean([abs(right_eyebrow_upper[0].y - right_eyebrow_upper[-1].y)])
                mouth_dist = np.mean([abs(mouth_upper[0].y - mouth_lower[0].y)])

                # Angry detection logic (simplified)
                # if left_eyebrow_dist < 0.01 and right_eyebrow_dist < 0.01 and mouth_dist < 0.015: #Adjust these thresholds
                #     image_width = image.shape[1]  # gets the image width
                #     cv2.putText(image, "Angry Face Detected!", (image_width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                 (0, 0, 255), 2)
                #     cv2.putText(image, "Angry Face Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                image_width = image.shape[1]  # gets the image width
                cv2.putText(image, "Threat Detected", (image_width - 1900, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        cv2.imshow('Face Expression Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_angry_face()