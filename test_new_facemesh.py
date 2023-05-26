import urllib.request
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import os
from scipy.io import savemat

# # Get the face_landmarker_v2_with_blendshapes.task model file
# urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
#                            "face_landmarker_v2_with_blendshapes.task")


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks # type: ignore
        ])

        solutions.drawing_utils.draw_landmarks(  # type: ignore
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,  # type: ignore
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(  # type: ignore
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,  # type: ignore
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(  # type: ignore
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,  # type: ignore
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles  # type: ignore
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [
        face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [
        face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores,
                  label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(),
                 patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def cv2_imshow(img, window_name="Image Window"):
    """Convenience function to show an image"""
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_sample():
    # Download input image
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-assets/portrait.jpg", "image.jpg")

    img = cv2.imread("image.jpg")
    # Display the input image
    cv2_imshow(img)

    # STEP 2: Create an FaceLandmarker object.
    base_options = mp_python.BaseOptions(
        model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file("image.jpg")

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Display bar graph of face blendshapes category scores
    # plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

    # Print transformation matrixes
    # print(detection_result.facial_transformation_matrixes)

    return


def _read_binary_file(filepath):
        x_all, y_all, z_all, confidence_all = None, None, None, None

        with open(filepath, 'rb') as binary_file:
            x_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            y_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            z_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            confidence_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()

        return x_all, y_all, z_all, confidence_all


def _convert_camera_confidence_to_grayscale(confidence_array: np.ndarray) -> np.ndarray:
    """
    Convert the input confidence array to grayscale and scale down the brightness to help
    with face detection.

    Args:
        confidence_array: An (n, d) confidence image in the format outputted by the IMX520 camera.

    Returns:
        An (n, d) grayscale image containing grayscale intensity values in the range [0, 255].
    """

    divisor = 4
    
    grayscale_img = confidence_array.astype(float)
    grayscale_img = grayscale_img / divisor
    grayscale_img[np.where(grayscale_img > 255)] = 255
    grayscale_img = grayscale_img.astype('uint8')

    return grayscale_img

    # # This is a new implementation that I believe should be more resilient to
    # # changes in the lighting conditions of the scene.

    # # Normalize the confidence values to the range [0, 1]
    # min_val = np.min(confidence_array)
    # max_val = np.max(confidence_array)
    # normalized_data = (confidence_array - min_val) / (max_val - min_val)

    # # Map the normalized data to the range [0, 255]
    # grayscale_image = (normalized_data * 255).astype(np.uint8)

    # return grayscale_image

def _convert_grayscale_image_to_MediaPipe_image(grayscale_img: np.ndarray) -> mp.Image:
    """
    Create a MediaPipe Image object from a grayscale image.

    Args:
        grayscale_img: A NumPy array representing the grayscale image with shape (height, width).

    Returns:
        An instance of `mp.Image` representing the grayscale image in MediaPipe format.
    """    
    # Set the image format
    image_format = mp.ImageFormat.GRAY8

    # Create the MediaPipe Image object with the provided arguments
    image = mp.Image(image_format, grayscale_img)

    return image

def run_facemesh():
    # Get input images (frames of video(s))
    
    skvs_dir = os.path.join(os.getcwd(), 'skvs')
    input_dir=os.path.join(skvs_dir, "mat")
    output_filename="auto_bfsig_new"

    # The IMX520 sensor has a resolution of 640x480=307200 pixels per frame (width x height)
    # width = 640
    img_cols = 640
    # height = 480
    img_rows = 480

    num_ROIs = 7

    # Array of intensity signal arrays
    # Each element is (7, num_frames) = (7, 600) for 7 ROIs (regions of interest) and (likely) 600 frames per input video file
    intensity_signals = np.zeros((num_ROIs, 1))

    # Array of depth signal arrays
    depth_signals = np.zeros((num_ROIs, 1))

    # Not sure what this is for
    ear_signal = np.zeros((1))

    # Get list of all input files in input_mats_dir (./skvs/mat/)
    filelist = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.skv.bin'):
            # Remove the ".bin" suffix
            filename = filename[:-4]
            filelist.append(filename)

    # Load and process every input video file. Track and map face using MediaPipe.

    file_num = 0
    num_files_to_process = len(filelist)

    # Create an FaceLandmarker object
    base_options = mp_python.BaseOptions(
        model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    for filename in filelist:
        file_num = file_num + 1
        print(f"Processing file {file_num}/{num_files_to_process}: {filename}...")

        # Load the file
        filepath = os.path.join(input_dir, filename + '.bin')
        x_all, y_all, z_all, confidence_all = _read_binary_file(filepath)

        # Get number of frames (columns) in this video clip
        num_frames = np.shape(confidence_all)[1]

        # ROI indices:
        # 0: nose
        # 1: forehead
        # 2: cheek_and_nose
        # 3: left_cheek
        # 4: right_cheek
        # 5: low_forehead
        # 6: palm

        # Create arrays to store intensity and depth signals for all ROIs in this video clip (num_ROIs, num_frames) = (7, 600)
        intensity_signal_current = np.zeros((num_ROIs, num_frames))
        depth_signal_current = np.zeros((num_ROIs, num_frames))
        ear_signal_current = np.zeros(num_frames)

        # Each array is currently (height*width, num_frames) = (480*640, num_frames) = (307200, num_frames)
        # Reshape to (height, width, num_frames) = (480, 640, num_frames)
        x_all = x_all.reshape([img_rows, img_cols, num_frames])
        y_all = y_all.reshape([img_rows, img_cols, num_frames])
        z_all = z_all.reshape([img_rows, img_cols, num_frames])
        confidence_all = confidence_all.reshape([img_rows, img_cols, num_frames])
        
        # Loop through all frames
        for frame in range(num_frames):
            frame_x = x_all[:, :, frame]
            frame_y = y_all[:, :, frame]
            frame_z = z_all[:, :, frame]
            frame_confidence = confidence_all[:, :, frame]

            # Send image through MediaPipe to get face landmarks

            # Convert confidence array to grayscale image
            frame_grayscale = _convert_camera_confidence_to_grayscale(frame_confidence)

            # Display the image
            cv2.imshow("Grayscale Image", frame_grayscale)
            # Wait for a key press to close the window
            # cv2.waitKey(0)
            cv2.waitKey(10)

            # Load the input image
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_grayscale, cv2.COLOR_GRAY2RGB))
            # mp_img = mp.Image(image_format=mp.ImageFormat.GRAY8, data=frame_grayscale)

            # STEP 4: Detect face landmarks from the input image.
            detection_result = detector.detect(mp_img)

            # STEP 5: Process the detection result. In this case, visualize it.
            annotated_image = draw_landmarks_on_image(mp_img.numpy_view(), detection_result)
            cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()

        intensity_signals = np.concatenate((intensity_signals, intensity_signal_current), axis=1)
        depth_signals = np.concatenate((depth_signals, depth_signal_current), axis=1)
        ear_signal = np.concatenate((ear_signal, ear_signal_current),axis=0)
    
    intensity_signals = np.delete(intensity_signals, 0, 1)
    depth_signals = np.delete(depth_signals, 0, 1)
    ear_signal = np.delete(ear_signal,0,0)
    mdic = {"Depth": depth_signals, 'I_raw': intensity_signals, 'EAR': ear_signal} # EAR: eye aspect ratio
    savemat(os.path.join(input_dir, output_filename + '.mat'), mdic)

    print('finished')
    return

if __name__ == "__main__":
    # run_sample()

    run_facemesh()