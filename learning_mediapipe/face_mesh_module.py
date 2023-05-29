import cv2
import mediapipe as mp
import time
import numpy as np


class FaceMeshDetector():
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_draw = mp.solutions.drawing_utils # type: ignore
        self.mp_face_mesh = mp.solutions.face_mesh # type: ignore
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                    max_num_faces=max_num_faces,
                                                    min_detection_confidence=min_detection_confidence,
                                                    min_tracking_confidence=min_tracking_confidence)
        self.drawing_spec_landmark = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
        self.drawing_spec_connection = self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)

    def find_face_mesh(self, image, draw=False):
        # Get image dimensions
        image_height, image_width, image_channels = image.shape

        # CV2 loads images/videos in BGR format, convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(image_rgb)
        # print(results)

        # landmarks_pixels is an array of shape (468, 2) with x, y coordinates (as pixels) for each landmark
        landmarks_pixels = np.zeros((468, 2), dtype="int")

        face_detected = False

        if results.multi_face_landmarks:
            # Face(s) were detected
            # Loop through each face detected

            face_detected = True

            for face_landmarks in results.multi_face_landmarks:
                # face_landmarks is a list of landmarks for face

                if draw:
                    self.mp_draw.draw_landmarks(image, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS, self.drawing_spec_landmark,
                                                self.drawing_spec_connection)

                # Loop through each landmark
                for id, landmark in enumerate(face_landmarks.landmark):
                    # There are 468 landmarks in total, with x, y, z normalized coordinates
                    # print(id, landmark)

                    # Convert normalized coordinates to pixel coordinates (NOTE: z is currently unused)
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                    # print(id, x, y)

                    # Store pixel coordinates in array
                    landmarks_pixels[id] = (x, y)

        return face_detected, landmarks_pixels


def main():
    input_video = "videos/test_video.mp4"
    video = cv2.VideoCapture(input_video)
    previous_time = 0

    detector = FaceMeshDetector()

    # Loop through video frames
    while video.isOpened():
        # Get a frame of video
        ret, frame = video.read()
        if not ret:
            break

        # Find face mesh
        face_detected, landmarks_pixels = detector.find_face_mesh(image=frame, draw=True)

        # if not face_detected:
        #     print("Face not detected")

        # Calculate and overlay FPS

        current_time = time.time()
        # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Display frame

        cv2.imshow("Image", frame)
        cv2.waitKey(1)

    video.release()


if __name__ == "__main__":
    main()