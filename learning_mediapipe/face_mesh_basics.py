import cv2
import mediapipe as mp
import time

input_video = "videos/test_video.mp4"
video = cv2.VideoCapture(input_video)
previous_time = 0

mp_draw = mp.solutions.drawing_utils # type: ignore
mp_face_mesh = mp.solutions.face_mesh # type: ignore
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
drawing_spec_landmark = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
drawing_spec_connection = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)

# Loop through video frames
while video.isOpened():
    # Get a frame of video
    ret, frame = video.read()
    if not ret:
        break

    # CV2 loads images/videos in BGR format, convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame_rgb)
    # print(results)

    if results.multi_face_landmarks:
        # Face(s) were detected
        # Loop through each face detected
        for face_landmarks in results.multi_face_landmarks:
            # face_landmarks is a list of landmarks for face
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec_landmark, drawing_spec_connection)

            # Loop through each landmark
            for id, landmark in enumerate(face_landmarks.landmark):
                # There are 468 landmarks in total, with x, y, z normalized coordinates
                # print(id, landmark)

                # Get image dimensions
                image_height, image_width, image_channels = frame.shape

                # Convert normalized coordinates to pixel coordinates (NOTE: z is currently unused)
                x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                print(id, x, y)

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