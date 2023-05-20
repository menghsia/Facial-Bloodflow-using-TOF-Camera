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
        x_all, y_all, z_all, gray_all = None, None, None, None

        with open(filepath, 'rb') as binary_file:
            x_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            y_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            z_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            gray_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()

        return x_all, y_all, z_all, gray_all


def _mp_preprocess(frameTrk, divisor=4):
        frameTrk = frameTrk.astype(float)
        frameTrk = frameTrk / divisor
        frameTrk[np.where(frameTrk > 255)] = 255
        frameTrk = frameTrk.astype('uint8')
        image_3chnl = np.stack((frameTrk,) * 3, axis=-1)

        return image_3chnl


def _convert_camera_grayscale_to_3_channel_RGB(frame_gray: np.ndarray, illumination_multiplier: float = 0.25) -> np.ndarray:
    """
    Takes input an (n,d) grayscale image in the format outputted by IMX520.
    1. Converts it to a 3-channel RGB image where all three channels are equal.
    2. Multiplies the image by a constant to alter the brightness.
    
    Returns an (n,d,3) "RGB" image.
    """
    frameTrk = frame_gray.astype(float)
    frameTrk = frameTrk * illumination_multiplier
    frameTrk[np.where(frameTrk > 255)] = 255
    frameTrk = frameTrk.astype('uint8')
    image_3chnl = np.stack((frameTrk,) * 3, axis=-1)
    return image_3chnl


def _convert_camera_confidence_to_grayscale(confidence_array: np.ndarray) -> np.ndarray:
    """
    Takes input an (n,d) confidence image in the format outputted by the IMX520 camera.
    1. Converts input to grayscale.
    2. Stacks grayscale array to create (n,d,3) "RGB" array where all three channels
    are equal arrays that are all grayscale.
    
    Returns an (n,d,3) "RGB" array.
    """
    # Normalize the confidence values to the desired range
    min_val = np.min(confidence_array)
    max_val = np.max(confidence_array)
    normalized_data = (confidence_array - min_val) / (max_val - min_val)

    # Map the normalized data to the range [0, 255]
    grayscale_image = (normalized_data * 255).astype(np.uint8)

    return grayscale_image

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
        x_all, y_all, z_all, gray_all = _read_binary_file(filepath)

        # Get number of frames (columns) in this video clip
        num_frames = np.shape(gray_all)[1]

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
        gray_all = gray_all.reshape([img_rows, img_cols, num_frames])
        
        # Loop through all frames
        for frame in range(num_frames):
            frame_x = x_all[:, :, frame]
            frame_y = y_all[:, :, frame]
            frame_z = z_all[:, :, frame]
            frame_gray = gray_all[:, :, frame]

            # Send image through MediaPipe to get face landmarks

            # frame_gray_3_channel = _mp_preprocess(frame_gray, divisor=4)
            # frame_gray_3_channel = _convert_camera_grayscale_to_3_channel_RGB(frame_gray, illumination_multiplier=.25)
            frame_gray_3_channel = _convert_camera_confidence_to_grayscale(frame_gray)

            # Print the range of values in the array
            # print(f"frame_gray_3_channel range: [{np.min(frame_gray_3_channel)}, {np.max(frame_gray_3_channel)}]")

            # Display the actual image we are about to process with MediaPipe
            # Display the image
            cv2.imshow("Image", frame_gray_3_channel)

            # Wait for a key press to close the window
            # cv2.waitKey(0)
            cv2.waitKey(10)

            print("Image shown")

            # Load the input image
            # image = mp.Image(frame_gray)
        
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