import cv2
import mediapipe as mp
import time

input_video = "videos/test_video.mp4"
video = cv2.VideoCapture(input_video)
previous_time = 0

mp_face_detection = mp.solutions.face_detection # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)

# Loop through video frames
while video.isOpened():
    # Get a frame of video
    ret, frame = video.read()
    if not ret:
        break

    # CV2 loads images/videos in BGR format, convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(frame, detection)
            # print(id, detection)

            # Get image dimensions
            image_height, image_width, image_channels = frame.shape

            # Get bounding box coordinates
            # print(detection.location_data.relative_bounding_box)
            bounding_box_normalized = detection.location_data.relative_bounding_box
            bounding_box = int(bounding_box_normalized.xmin * image_width), int(bounding_box_normalized.ymin * image_height), \
                            int(bounding_box_normalized.width * image_width), int(bounding_box_normalized.height * image_height)
            cv2.rectangle(frame, bounding_box, (255, 0, 255), 2)

            cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (bounding_box[0], bounding_box[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    current_time = time.time()
    # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)

video.release()