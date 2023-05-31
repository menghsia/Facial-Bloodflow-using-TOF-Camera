import os
import time
import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.framework.formats import landmark_pb2
from PIL import Image, ImageDraw
from scipy.io import savemat
# from scipy.io import loadmat
from typing import Tuple, Union
from scipy.spatial import distance as dist
import concurrent.futures

# import h5py
# import hdf5storage

class FaceMeshDetector():
    """
    FaceMeshDetector is a class that performs face detection and landmark tracking using MediaPipe FaceMesh.

    Args:
        input_dir (str): Directory where input files are located.
        output_filename (str): Filename of the output .mat file.

    Attributes:
        input_dir (str): Directory where input files are located.
        output_filename (str): Filename of the output .mat file.
    """

    def __init__(self, input_dir: str, output_filename: str):
        """
        Initialize class variables.

        Args:
            input_dir: Directory where input files are located.
            output_filename: Filename of the output .mat file.
        """
        # Directory where input files are located (likely ./skvs/mat/)
        self.input_dir = input_dir

        # Filename of output .mat file (likely auto_bfsig.mat)
        self.output_filename = output_filename
    
    def run(self, visualize_ROI: bool = False, visualize_FaceMesh: bool = False) -> None:
        """
        Run the face mesh detection and intensity signal extraction.

        Args:
            visualize_ROI: Flag indicating whether to visualize the region(s) of interest (not sure what region(s) this is referring to).
            visualize_FaceMesh: Flag indicating whether to visualize the face mesh (the creepy mask-looking thing).
        """
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
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.skv.bin'):
                # Remove the ".bin" suffix
                filename = filename[:-4]
                filelist.append(filename)

        # Load and process every input video file. Track and map face using MediaPipe.

        file_num = 0
        num_files_to_process = len(filelist)
        
        # Define MediaPipe detectors
        my_mp_face_mesh = mp_face_mesh
        
        # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # This might be worth trying to increase accuracy:
        # Increasing min_tracking_confidence [0.0, 1.0] will generally improve the quality of the landmarks at the expense of a higher latency.
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        # with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        # Get the number of available threads
        num_threads = os.cpu_count()

        # If the CPU does not support multithreading, set num_threads to 1 (single-threaded)
        if num_threads is None:
            num_threads = 1
        elif num_threads < 1:
            num_threads = 1
        
        print(f"Using {num_threads} threads")

        tasks = []
        new_task = None
        
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
        
        with my_mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as my_face_mesh:
            # Loop through each file
            for filename in filelist:
                file_num = file_num + 1
                print(f"Processing file {file_num}/{num_files_to_process}: {filename}...")

                # Load the file
                filepath = os.path.join(self.input_dir, filename + '.bin')
                x_all, y_all, z_all, confidence_all = self._read_binary_file(filepath)

                # Get number of frames (columns) in this video clip
                # num_frames = np.size(gray_all, 1)
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

                    # Track face and extract intensity and depth for all ROIs in this frame
                    frame_grayscale = self._convert_camera_confidence_to_grayscale(frame_confidence)
                    
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    frame_grayscale.flags.writeable = False
                    frame_grayscale = cv2.cvtColor(frame_grayscale, cv2.COLOR_BGR2RGB)
                    results_face = my_face_mesh.process(frame_grayscale)
                    # results_hand = hands.process(frameTrk)

                    if hasattr(results_face, "multi_face_landmarks"):
                        face_landmarks = getattr(results_face, "multi_face_landmarks")[0]

                        # Queue each task using ThreadPoolExecutor.submit() so that the tasks are executed in parallel
                        new_task = thread_pool.submit(self._process_frame, frame_x, frame_y, frame_z, frame_confidence, frame,
                                                                                            img_rows, img_cols,
                                                                                            intensity_signal_current,
                                                                                            depth_signal_current,
                                                                                            ear_signal_current,
                                                                                            face_landmarks)
                        tasks.append(new_task)

                # Wait for the tasks to complete
                concurrent.futures.wait(tasks)

                intensity_signals = np.concatenate((intensity_signals, intensity_signal_current), axis=1)
                depth_signals = np.concatenate((depth_signals, depth_signal_current), axis=1)
                ear_signal = np.concatenate((ear_signal, ear_signal_current),axis=0)
        
        thread_pool.shutdown(wait=True, cancel_futures=False)
                    
        intensity_signals = np.delete(intensity_signals, 0, 1)
        depth_signals = np.delete(depth_signals, 0, 1)
        ear_signal = np.delete(ear_signal,0,0)
        mdic = {"Depth": depth_signals, 'I_raw': intensity_signals, 'EAR': ear_signal} # EAR: eye aspect ratio
        savemat(os.path.join(self.input_dir, self.output_filename + '.mat'), mdic)
        
        print('finished')

    def _process_frame(self, frame_x: np.ndarray, frame_y: np.ndarray, frame_z: np.ndarray,
                              frame_confidence: np.ndarray, frame: int, img_rows: int, img_cols: int,
                              intensity_signal_current: np.ndarray, depth_signal_current: np.ndarray,
                              ear_signal_current: np.ndarray, face_landmarks: landmark_pb2.NormalizedLandmarkList) -> None:
        """
        Processes a single frame to extract intensity and depth signals for each region of interest (ROI).

        Args:
            frame_x: X-coordinate values of the face mesh landmarks for the frame.
            frame_y: Y-coordinate values of the face mesh landmarks for the frame.
            frame_z: Z-coordinate values of the face mesh landmarks for the frame.
            frame_confidence: Confidence values of the face mesh landmarks for the frame.
            frame: Frame number.
            img_rows: Number of rows in the frame.
            img_cols: Number of columns in the frame.
            intensity_signal_current: Array to store the intensity signals for each ROI.
            depth_signal_current: Array to store the depth signals for each ROI.
            ear_signal_current: Array to store the eye aspect ratio (EAR) signals.
            face_landmarks: Face landmarks for the frame.
        """
        # print(f"{frame_num}: Worker starting...")

        # find the ROI vertices
        landmark_forehead = self._ROI_coord_extract(face_landmarks, 'forehead', img_rows, img_cols)
        mask_forehead = self._vtx2mask(landmark_forehead, img_cols, img_rows)
        landmark_nose = self._ROI_coord_extract(face_landmarks, 'nose', img_rows, img_cols)
        mask_nose = self._vtx2mask(landmark_nose, img_cols, img_rows)
        landmark_cheek_and_nose = self._ROI_coord_extract(face_landmarks, 'cheek_n_nose', img_rows, img_cols)
        mask_cheek_and_nose = self._vtx2mask(landmark_cheek_and_nose, img_cols, img_rows)
        landmark_left_cheek = self._ROI_coord_extract(face_landmarks, 'left_cheek', img_rows, img_cols)
        mask_left_cheek = self._vtx2mask(landmark_left_cheek, img_cols, img_rows)
        landmark_right_cheek = self._ROI_coord_extract(face_landmarks, 'right_cheek', img_rows, img_cols)
        mask_right_cheek = self._vtx2mask(landmark_right_cheek, img_cols, img_rows)
        landmark_low_forehead = self._ROI_coord_extract(face_landmarks, 'low_forehead', img_rows, img_cols)
        mask_low_forehead = self._vtx2mask(landmark_low_forehead, img_cols, img_rows)

        # calculate averaged intensity and depth for each ROI
        intensity_signal_current[0, frame] = np.average(frame_confidence[np.where(mask_nose > 0)])
        depth_signal_current[0, frame] = np.sqrt(np.average(frame_x[np.where(mask_nose > 0)]) ** 2 + np.average(frame_y[np.where(mask_nose > 0)]) ** 2 + np.average(frame_z[np.where(mask_nose > 0)]) ** 2)
        intensity_signal_current[1, frame] = np.average(frame_confidence[np.where(mask_forehead > 0)])
        depth_signal_current[1, frame] = np.sqrt(np.average(frame_x[np.where(mask_forehead > 0)]) ** 2 + np.average(frame_y[np.where(mask_forehead > 0)]) ** 2 + np.average(frame_z[np.where(mask_forehead > 0)]) ** 2)
        intensity_signal_current[2, frame] = np.average(frame_confidence[np.where(mask_cheek_and_nose > 0)])
        depth_signal_current[2, frame] = np.sqrt(np.average(frame_x[np.where(mask_cheek_and_nose > 0)]) ** 2 + np.average(frame_y[np.where(mask_cheek_and_nose > 0)]) ** 2 + np.average(frame_z[np.where(mask_cheek_and_nose > 0)]) ** 2)
        intensity_signal_current[3, frame] = np.average(frame_confidence[np.where(mask_left_cheek > 0)])
        depth_signal_current[3, frame] = np.sqrt(np.average(frame_x[np.where(mask_left_cheek > 0)]) ** 2 + np.average(frame_y[np.where(mask_left_cheek > 0)]) ** 2 + np.average(frame_z[np.where(mask_left_cheek > 0)]) ** 2)
        intensity_signal_current[4, frame] = np.average(frame_confidence[np.where(mask_right_cheek > 0)])
        depth_signal_current[4, frame] = np.sqrt(np.average(frame_x[np.where(mask_right_cheek > 0)]) ** 2 + np.average(frame_y[np.where(mask_right_cheek > 0)]) ** 2 + np.average(frame_z[np.where(mask_right_cheek > 0)]) ** 2)
        intensity_signal_current[5, frame] = np.average(frame_confidence[np.where(mask_low_forehead > 0)])
        depth_signal_current[5, frame] = np.sqrt(np.average(frame_x[np.where(mask_low_forehead > 0)]) ** 2 + np.average(frame_y[np.where(mask_low_forehead > 0)]) ** 2 + np.average(frame_z[np.where(mask_low_forehead > 0)]) ** 2)

        # PERCLOS
        landmark_leye = self._ROI_coord_extract(face_landmarks, 'left_eye', img_rows, img_cols)
        L_ear = self._eye_aspect_ratio(landmark_leye)
        landmark_reye = self._ROI_coord_extract(face_landmarks, 'right_eye', img_rows, img_cols)
        R_ear = self._eye_aspect_ratio(landmark_reye)
        ear_signal_current[frame] = (L_ear + R_ear) /2

        # # Show visualizations (Disabled to improve performance. Also, not currently working.)
        # if visualize_ROI:
        #     self._visualize_ROI(frameTrk, landmark_leye, landmark_reye)
        
        # if visualize_FaceMesh:
        #     self._visualize_FaceMesh(frameTrk, face_landmarks, results_face, mp_drawing_styles, mp_drawing, mp_face_mesh)
        
        # print(f"{frame_num}: Worker exiting...")

    def _visualize_ROI(self, frameTrk: np.ndarray, landmark_leye: list, landmark_reye: list) -> None:
        """
        Visualize the regions of interest (ROIs) on the image.

        Args:
            frameTrk: The frame image.
            landmark_leye: Landmark coordinates of the left eye.
            landmark_reye: Landmark coordinates of the right eye.
        """
        # Draw the face mesh annotations on the image and display
        frameTrk.flags.writeable = True
        image = cv2.cvtColor(frameTrk, cv2.COLOR_RGB2BGR)

        # pts = np.asarray(landmark_forehead)
        # pts = pts.reshape((-1, 1, 2))
        # img_showROI = cv2.polylines(image, [pts], True, color=(0, 0, 255), thickness=2)
        # pts = np.asarray(landmark_nose)
        # pts = pts.reshape((-1, 1, 2))
        # img_showROI = cv2.polylines(img_showROI, [pts], True, color=(0, 0, 255), thickness=2)
        pts = np.asarray(landmark_leye)
        pts = pts.reshape((-1, 1, 2))
        img_showROI = cv2.polylines(image, [pts], True, color=(0, 0, 255), thickness=2)
        pts = np.asarray(landmark_reye)
        pts = pts.reshape((-1, 1, 2))
        img_showROI = cv2.polylines(img_showROI, [pts], True, color=(0, 0, 255), thickness=2)

        # pts = np.asarray(landmark_palm)
        # pts = pts.reshape((-1, 1, 2))
        # img_showROI = cv2.polylines(img_showROI,[pts],True, color = (0,0,255), thickness = 2)

        cv2.imshow('ROI', img_showROI)
        cv2.waitKey(10)

        return

    def _visualize_FaceMesh(self, frameTrk: np.ndarray, face_landmarks: landmark_pb2.NormalizedLandmarkList,
                            results_face: mp_face_mesh.FaceMesh, mp_drawing_styles,
                            mp_drawing, mp_face_mesh: mp_face_mesh) -> None:
        """
        Visualize the FaceMesh annotations on the image.

        Args:
            frameTrk: The frame image.
            face_landmarks: Detected face landmarks.
            results_face: FaceMesh detection results.
            mp_drawing_styles: Drawing styles for FaceMesh annotations.
            mp_drawing: Drawing utilities.
            mp_face_mesh: FaceMesh solution.
        """
        # Draw the face mesh annotations on the image and display
        frameTrk.flags.writeable = True
        image = cv2.cvtColor(frameTrk, cv2.COLOR_RGB2BGR)

        if face_landmarks is not None:
            for face_landmarks_i in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_i,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_i,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_i,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        cv2.waitKey(10)

        return

    def _read_binary_file(self, filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read a binary file containing x, y, z coordinates, and confidence values.

        Args:
            filepath: The path to the binary file to be read.

        Returns:
            A tuple containing four NumPy arrays: x_all, y_all, z_all, and confidence_all.
            - x_all: An (n,d) array of x-coordinates.
            - y_all: An (n,d) array of y-coordinates.
            - z_all: An (n,d) array of z-coordinates.
            - confidence_all: An (n,d) array of confidence values.

        This method reads a binary file and extracts x, y, z coordinates, and confidence values.
        The binary file is assumed to have a specific structure where each array is stored sequentially.

        Note: This method assumes that the binary file is properly formatted and contains valid data.

        Example usage:
            x, y, z, confidence = _read_binary_file('data.bin')
        """
        x_all, y_all, z_all, confidence_all = None, None, None, None

        with open(filepath, 'rb') as binary_file:
            x_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            y_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            z_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()
            confidence_all = np.frombuffer(binary_file.read(600 * 307200 * 2), dtype=np.int16).reshape((600, 307200)).transpose()

        return x_all, y_all, z_all, confidence_all
    
    def _save_to_mat_file(self, x_all: np.ndarray, y_all: np.ndarray, z_all: np.ndarray, confidence_all: np.ndarray,
                          output_dir_path: str, filename: str) -> None:
        """
        Save the provided arrays to a MATLAB (.mat) file.

        Args:
            x_all: (n,d) array containing X-coordinate data.
            y_all: (n,d) array containing Y-coordinate data.
            z_all: (n,d) array containing Z-coordinate data.
            confidence_all: (n,d) array containing confidence values.
            output_dir_path: Path to the output directory.
            filename: Name of the output file (without extension).

        Returns:
            None
        """
        # mdic = {"Depth": D_signal, 'I_raw': I_signal, 'EAR': EAR} # EAR: eye aspect ratio
        # savemat(os.path.join(matpath, matname + '.mat'), mdic)

        mat_dict = {'x_all': x_all, 'y_all': y_all, 'z_all': z_all, 'confidence_all': confidence_all}
        savemat(os.path.join(output_dir_path, filename + '.mat'), mat_dict)
        # hdf5storage.write(mat_dict, output_dir_path, filename + '.mat', matlab_compatible=True)

        # Save using Matlab 7.3 to use HDF5 format (code does not currently work)
        # # make a dictionary to store the MAT data in
        # matfiledata = {}
        # # *** u prefix for variable name = unicode format, no issues thru Python 3.5
        # # advise keeping u prefix indicator format based on feedback despite docs ***
        # matfiledata[u'x_all'] = x_all
        # matfiledata[u'y_all'] = y_all
        # matfiledata[u'z_all'] = z_all
        # matfiledata[u'gray_all'] = gray_all
        # hdf5storage.write(matfiledata, output_dir_path, filename + '.mat', matlab_compatible=True)
        return

    def _line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        global x1, y1, x2, y2
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    def _toggle_selector(self, event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and self._toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            self._toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not self._toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            self._toggle_selector.RS.set_active(True)

    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int,
            image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def _ROI_coord_extract(self, face_landmarks, ROIwhich, image_rows, image_cols):
        # face_landmarks are the detected landmarks in a image
        if ROIwhich == 'full_face':
            ROI_vertex = [54, 284, 454, 365, 136, 234]
            # ROI_vertex = [137, 366, 365,152, 136]
        elif ROIwhich == 'left_face':
            ROI_vertex = [70, 135, 200, 8]
        elif ROIwhich == 'cheek_n_nose':
            ROI_vertex = [117, 346, 411, 187]
            # ROI_vertex = [116, 340, 433, 213]
        elif ROIwhich == 'left_cheek':
            ROI_vertex = [131, 165, 214, 50]
        elif ROIwhich == 'right_cheek':
            ROI_vertex = [372, 433, 358]
        elif ROIwhich == 'chin':
            ROI_vertex = [175, 148, 152, 377]
        elif ROIwhich == 'nose':
            ROI_vertex = [196, 419, 455, 235]
        elif ROIwhich == 'low_forehead':
            # ROI_vertex = [109,338,336,107]
            ROI_vertex = [108, 337, 8]
            # ROI_vertex = [109,338,9]

        elif ROIwhich == 'forehead':
            ROI_vertex = [109, 338, 9]

        elif ROIwhich == 'palm':
            ROI_vertex = [0, 5, 17]
        elif ROIwhich == 'left_eye':
            ROI_vertex = [33, 160, 159, 158, 133, 153, 145, 144]
        elif ROIwhich == 'right_eye':
            ROI_vertex = [263, 387, 386, 385, 362, 380, 374, 373]

        else:
            print('No such ROI')
            quit()
        # Landmarks can be found on https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

        # Facemesh detection

        # Extract coordinates of all pixels within the ROI polygon
        landmark_px = []

        for i, vtx in enumerate(ROI_vertex):
            landmark_current = self._normalized_to_pixel_coordinates(face_landmarks.landmark[vtx].x,
                                                                face_landmarks.landmark[vtx].y, image_cols, image_rows)
            landmark_px.append(landmark_current)
            # print(landmark_px)

        # n-by-2 2d array
        return landmark_px

    def _Chest_ROI_extract(self, image, chin_location, plot=False):
        # mp_pose = mp.solutions.pose
        # pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

        # image_3chnl = np.stack((image,) * 3, axis=-1)
        # image_3chnl = cv2.convertScaleAbs(image_3chnl)
        # shoulder_landmark = [11, 12]
        # landmark_px_rr = np.zeros([2, 4, 2])
        # image_height, image_width = image.shape
        # results = pose.process(image_3chnl)
        # body_points = results.pose_landmarks.landmark
        # shoulder_point_l = self._normalized_to_pixel_coordinates(body_points[11].x, body_points[11].y, image_width, image_height)
        # shoulder_point_r = self._normalized_to_pixel_coordinates(body_points[12].x, body_points[12].y, image_width, image_height)

        # shoulder_x = (shoulder_point_l[0] + shoulder_point_r[0]) / 2
        # shoulder_y = (shoulder_point_l[1] + shoulder_point_r[1]) / 2

        # neck_width = 2 * np.abs(chin_location[1][0] - chin_location[3][0])
        # neck_height = 0.5 * np.abs(shoulder_y - chin_location[2][1])

        # chest_width = np.abs(shoulder_point_l[0] - shoulder_point_r[0])
        # chest_height = 0.22 * chest_width

        # landmark_px_rr[0, :, 0] = [
        #     shoulder_x - 0.5 * neck_width,
        #     shoulder_x + 0.5 * neck_width,
        #     shoulder_x + 0.5 * neck_width,
        #     shoulder_x - 0.5 * neck_width
        # ]

        # landmark_px_rr[0, :, 1] = [
        #     shoulder_y - 1.1 * neck_height,
        #     shoulder_y - 1.1 * neck_height,
        #     shoulder_y - 0.1 * neck_height,
        #     shoulder_y - 0.1 * neck_height,
        # ]

        # # landmark_px_rr[0,:,0]=[chin_location[1][0]-0.8*neck_width,chin_location[3][0],
        # #                       chin_location[3][0],chin_location[1][0]-0.8*neck_width]
        # # landmark_px_rr[0,:,1]=[chin_location[1][1]+10,chin_location[3][1]+10,
        # #                       chin_location[3][1]+neck_height,chin_location[1][1]+neck_height]

        # landmark_px_rr[1, :, 0] = [
        #     shoulder_x - 0.3 * chest_width,
        #     shoulder_x + 0.3 * chest_width,
        #     shoulder_x + 0.3 * chest_width,
        #     shoulder_x - 0.3 * chest_width
        # ]

        # landmark_px_rr[1, :, 1] = [
        #     shoulder_y,
        #     shoulder_y,
        #     shoulder_y + chest_height,
        #     shoulder_y + chest_height
        # ]

        # # landmark_px_rr[1,:,0]=[shoulder_point_l[0]-25,shoulder_point_r[0]+25,
        # #                       shoulder_point_r[0]+25,shoulder_point_l[0]-25]
        # # landmark_px_rr[1,:,1]=[shoulder_point_l[1],shoulder_point_r[1],
        # #                       shoulder_point_r[1]+chest_height,shoulder_point_l[1]+chest_height]

        # np.clip(landmark_px_rr[0, :, 0], 0, image_width)
        # np.clip(landmark_px_rr[0, :, 1], 0, image_height)
        # np.clip(landmark_px_rr[1, :, 0], 0, image_width)
        # np.clip(landmark_px_rr[1, :, 1], 0, image_height)

        # if plot:
        #     plt.figure()
        #     plt.imshow(image, cmap='gray')
        #     plt.scatter(chin_location[1][0], chin_location[1][1], s=12, c='green', marker='x')
        #     plt.scatter(chin_location[3][0], chin_location[3][1], s=12, c='green', marker='x')

        #     plt.scatter(shoulder_point_l[0], shoulder_point_l[1], s=6, c='green', marker='o')
        #     plt.scatter(shoulder_point_r[0], shoulder_point_r[1], s=6, c='green', marker='o')
        #     for j in range(4):
        #         plt.scatter(landmark_px_rr[0][j][0], landmark_px_rr[0][j][1], s=8, c='red', marker='x')
        #         plt.scatter(landmark_px_rr[1][j][0], landmark_px_rr[1][j][1], s=1, c='black', marker='o')
        #         # plt.plot((landmark_px[k][j-1][0], landmark_px[k][j][0]),(landmark_px[k][j-1][1],landmark_px[k][j][1]), c='g', linewidth=1)
        # plt.show()
        # return landmark_px_rr
        return

    def _vtx2mask(self, vtx, image_cols, image_rows):
        """
        :param vtx: list of 2D coordinates of the polygon vertices
        :param image_cols: mask image columns
        :param image_rows: mask image rows
        :return: mask of polygon
        """
        maskimg = Image.new('L', (image_cols, image_rows), 0)
        ImageDraw.Draw(maskimg).polygon(vtx, outline=1, fill=1)
        mask = np.array(maskimg)

        return mask

    def _pre_whiten(self, signal):
        sig_avg = np.average(signal)
        sig_std = np.std(signal)

        sig_norm = (signal - sig_avg) / sig_std

        return sig_norm

    def _convert_camera_confidence_to_grayscale(self, confidence_array: np.ndarray) -> np.ndarray:
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

    def _eye_aspect_ratio(self, eye):
        # Vertical eye landmarks
        A = dist.euclidean(eye[1], eye[7])
        B = dist.euclidean(eye[2], eye[6])
        C = dist.euclidean(eye[3], eye[5])
        # Horizontal eye landmarks
        D = dist.euclidean(eye[0], eye[4])

        # The EAR Equation
        EAR = (A + B + C) / (3.0 * D)
        return EAR

if __name__ == "__main__":
    skvs_dir = os.path.join(os.getcwd(), 'skvs')

    myFaceMeshDetector = FaceMeshDetector(input_dir=os.path.join(skvs_dir, "mat"), output_filename="auto_bfsig")
    myFaceMeshDetector.run(visualize_ROI=False, visualize_FaceMesh=False)
