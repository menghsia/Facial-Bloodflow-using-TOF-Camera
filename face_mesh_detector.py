import os
import time
import math
import cv2
import numpy as np
# import mediapipe as mp
import matplotlib.pyplot as plt
# from mediapipe.python.solutions import face_mesh as mp_face_mesh
# from mediapipe.framework.formats import landmark_pb2
from PIL import Image, ImageDraw
from scipy.io import savemat
# from scipy.io import loadmat
from typing import Tuple, Union
from scipy.spatial import distance as dist
import concurrent.futures

from face_mesh_module import FaceMeshDetector

# import h5py
# import hdf5storage

class PhaseTwo():
    """
    PhaseTwo is a class that performs face detection and landmark tracking using MediaPipe FaceMesh.

    Args:
        input_dir (str): Directory where input files are located.
        output_filename (str): Filename of the output .mat file.

    Attributes:
        input_dir (str): Directory where input files are located.
        output_filename (str): Filename of the output .mat file.
    """

    def __init__(self, input_dir: str, output_filename: str, image_width: int = 640, image_height: int = 480):
        """
        Initialize class variables.

        Args:
            input_dir: Directory where input files are located.
            output_filename: Filename of the output .mat file.
            image_width: Width of input image (in pixels) (aka number of columns)
            image_height: Height of input image (in pixels) (aka number of rows)

        Note: The IMX520 sensor has an image resolution of 640x480=307200 pixels per frame (width x height)
        (aka 640 columns x 480 rows)
        """
        # Directory where input files are located (likely ./skvs/mat/)
        self.input_dir = input_dir

        # Filename of output .mat file (likely auto_bfsig.mat)
        self.output_filename = output_filename
        
        self.image_width = image_width
        self.image_height = image_height
    
    def run(self, visualize_ROI: bool = False, visualize_FaceMesh: bool = False) -> None:
        """
        Run the face mesh detection and intensity signal extraction.

        Args:
            visualize_ROI: Flag indicating whether to visualize the region(s) of interest (not sure what region(s) this is referring to).
            visualize_FaceMesh: Flag indicating whether to visualize the face mesh (the creepy mask-looking thing).
        """
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
        face_mesh_detector = FaceMeshDetector(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
        
        # with my_mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as my_face_mesh:
        
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
            x_all = x_all.reshape([self.image_height, self.image_width, num_frames])
            y_all = y_all.reshape([self.image_height, self.image_width, num_frames])
            z_all = z_all.reshape([self.image_height, self.image_width, num_frames])
            confidence_all = confidence_all.reshape([self.image_height, self.image_width, num_frames])

            # Used to calculate FPS
            previous_time = 0
            
            # Loop through all frames
            for frame_idx in range(num_frames):
                frame_x = x_all[:, :, frame_idx]
                frame_y = y_all[:, :, frame_idx]
                frame_z = z_all[:, :, frame_idx]
                frame_confidence = confidence_all[:, :, frame_idx]

                # Track face and extract intensity and depth for all ROIs in this frame

                # Convert the frame's confidence values to a grayscale image (n,d)
                frame_grayscale = self._convert_camera_confidence_to_grayscale(frame_confidence)

                # # To improve performance, optionally mark the image as not writeable to
                # # pass by reference.
                # frame_grayscale.flags.writeable = False

                # Convert grayscale image to "RGB" (n,d,3)
                frame_grayscale_rgb = cv2.cvtColor(frame_grayscale, cv2.COLOR_GRAY2RGB)

                # BEGIN OLD CODE

                # results_face = my_face_mesh.process(frame_grayscale)

                # if hasattr(results_face, "multi_face_landmarks"):
                #     face_landmarks = getattr(results_face, "multi_face_landmarks")[0]

                #     # Queue each task using ThreadPoolExecutor.submit() so that the tasks are executed in parallel
                #     new_task = thread_pool.submit(self._process_frame, frame_x, frame_y, frame_z, frame_confidence, frame,
                #                                                                         self.image_height, self.image_width,
                #                                                                         intensity_signal_current,
                #                                                                         depth_signal_current,
                #                                                                         ear_signal_current,
                #                                                                         face_landmarks)
                #     tasks.append(new_task)
                
                # END OLD CODE

                # Get pixel locations of all face landmarks
                face_detected, landmarks_pixels = face_mesh_detector.find_face_mesh(image=frame_grayscale_rgb, draw=True)

                # if face_detected:
                #     self._process_landmarks(landmarks_pixels)
                
                # Calculate and overlay FPS

                current_time = time.time()
                # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
                fps = 1 / (current_time - previous_time)
                previous_time = current_time
                cv2.putText(frame_grayscale_rgb, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                # Display frame

                cv2.imshow("Image", frame_grayscale_rgb)
                cv2.waitKey(1)

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
    
    def _process_face_landmarks(self, landmarks_pixels: np.ndarray):
        print("Processing landmarks (TODO)...")
        return
    
    def _get_ROI_bounding_box_pixels(self, landmarks_pixels: np.ndarray, roi_name: str) -> np.ndarray:
        """
        Takes in the pixel coordinates of all face landmarks and returns the pixel coordinates
        of the face landmarks that represent the bounding box for the requested ROI.

        Args:
            landmarks_pixels (np.ndarray): An array of shape (468, 2) representing the pixel coordinates (x, y)
                for each of the 468 total face landmarks detected. The i-th row corresponds to the i-th landmark
                (zero-indexed, so row 0 is landmark 1).
            roi_name (str): The name of the requested ROI. Choose from the following options:
                'full_face', 'left_face', 'cheek_n_nose', 'left_cheek', 'right_cheek',
                'chin', 'nose', 'low_forehead', 'forehead', 'palm', 'left_eye', 'right_eye'.

        Returns:
            np.ndarray: An array of shape (n, 2), where n is the number of landmarks
                that form the bounding box for the requested ROI. Each row represents the (x, y)
                coordinates of a landmark pixel.

        Raises:
            KeyError: If the provided roi_name does not match any of the predefined ROIs.

        Note:
            The returned bounding_box_pixels is in the same format as the input landmarks_pixels,
            with each array of shape (2,) representing the (x, y) coordinates of a landmark pixel.
        
        References:
        - Face Landmarks Key: https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/mesh_map.jpg
        """
        roi_definitions = {
            'full_face': [54, 284, 454, 365, 136, 234],
            'left_face': [70, 135, 200, 8],
            'cheek_n_nose': [117, 346, 411, 187],
            'left_cheek': [131, 165, 214, 50],
            'right_cheek': [372, 433, 358],
            'chin': [175, 148, 152, 377],
            'nose': [196, 419, 455, 235],
            'low_forehead': [108, 337, 8],
            'forehead': [109, 338, 9],
            'palm': [0, 5, 17],
            'left_eye': [33, 160, 159, 158, 133, 153, 145, 144],
            'right_eye': [263, 387, 386, 385, 362, 380, 374, 373]
        }

        bounding_box_pixels = np.array([])
        
        try:
            landmark_indices = np.array(roi_definitions[roi_name])
            # landmarks_pixels is zero-indexed, but the landmark indices are 1-indexed
            bounding_box_pixels = landmarks_pixels[landmark_indices - 1]
        except KeyError:
            raise KeyError(f"ERROR: The provided roi_name \"{roi_name}\" does not match any of the predefined ROIs.")

        return bounding_box_pixels
    
    def _get_pixels_within_ROI_bounding_box(self, bounding_box_pixels: np.ndarray) -> np.ndarray:
        """
        Takes in the pixel coordinates that represent the bounding box for an ROI and returns
        the pixel coordinates of all pixels within the bounding box.

        Args:
            bounding_box_pixels (np.ndarray): An array of shape (n, 2), where n is the number of landmarks
                that form the bounding box for the requested ROI. Each row represents the (x, y)
                coordinates of a landmark pixel.

        Returns:
            np.ndarray: An array of shape (image_height, image_width) representing the pixels within
                the bounding box. The pixels within the bounding box will have a value of 1, while
                the remaining pixels will have a value of 0. The output array has the same dimensions
                as the frame image.
        """

        # Create a black (0) grayscale image with the same dimensions as the frame
        # TODO: Try using mode='1' instead of mode='L' to save memory
        # (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)
        mask_canvas = Image.new('L', (self.image_width, self.image_height), 0)

        # Draw a polygon on the mask_canvas using the ROI bounding box pixel coordinates
        # The polygon will be filled in with pixels with value 1, and the outline will
        # be 1 pixel wide with a value of 1 as well.
        ImageDraw.Draw(mask_canvas).polygon(bounding_box_pixels.tolist(), fill=1, outline=1, width=1)

        # Convert the mask_canvas image, with the filled-in polygon on it, to a numpy array
        # The array will have a shape of (self.image_height, self.image_width)
        # TODO: Verify output shape
        pixels_in_ROI = np.array(mask_canvas)

        return pixels_in_ROI

    def _process_frame(self, frame_x: np.ndarray, frame_y: np.ndarray, frame_z: np.ndarray,
                              frame_confidence: np.ndarray, frame_idx: int,
                              intensity_signal_current: np.ndarray, depth_signal_current: np.ndarray,
                              ear_signal_current: np.ndarray, face_landmarks) -> None:
        """
        Processes a single frame to extract intensity and depth signals for each region of interest (ROI).

        Args:
            frame_x: X-coordinate values of the face mesh landmarks for the frame.
            frame_y: Y-coordinate values of the face mesh landmarks for the frame.
            frame_z: Z-coordinate values of the face mesh landmarks for the frame.
            frame_confidence: Confidence values of the face mesh landmarks for the frame.
            frame_idx: Frame index number.
            intensity_signal_current: Array to store the intensity signals for each ROI.
            depth_signal_current: Array to store the depth signals for each ROI.
            ear_signal_current: Array to store the eye aspect ratio (EAR) signals.
            face_landmarks: Face landmarks for the frame.
        """
        # print(f"{frame_num}: Worker starting...")

        # find the ROI vertices
        landmark_forehead = self._ROI_coord_extract(face_landmarks, 'forehead', self.image_height, self.image_width)
        mask_forehead = self._vtx2mask(landmark_forehead, self.image_width, self.image_height)
        landmark_nose = self._ROI_coord_extract(face_landmarks, 'nose', self.image_height, self.image_width)
        mask_nose = self._vtx2mask(landmark_nose, self.image_width, self.image_height)
        landmark_cheek_and_nose = self._ROI_coord_extract(face_landmarks, 'cheek_n_nose', self.image_height, self.image_width)
        mask_cheek_and_nose = self._vtx2mask(landmark_cheek_and_nose, self.image_width, self.image_height)
        landmark_left_cheek = self._ROI_coord_extract(face_landmarks, 'left_cheek', self.image_height, self.image_width)
        mask_left_cheek = self._vtx2mask(landmark_left_cheek, self.image_width, self.image_height)
        landmark_right_cheek = self._ROI_coord_extract(face_landmarks, 'right_cheek', self.image_height, self.image_width)
        mask_right_cheek = self._vtx2mask(landmark_right_cheek, self.image_width, self.image_height)
        landmark_low_forehead = self._ROI_coord_extract(face_landmarks, 'low_forehead', self.image_height, self.image_width)
        mask_low_forehead = self._vtx2mask(landmark_low_forehead, self.image_width, self.image_height)

        # calculate averaged intensity and depth for each ROI
        intensity_signal_current[0, frame_idx] = np.average(frame_confidence[np.where(mask_nose > 0)])
        depth_signal_current[0, frame_idx] = np.sqrt(np.average(frame_x[np.where(mask_nose > 0)]) ** 2 + np.average(frame_y[np.where(mask_nose > 0)]) ** 2 + np.average(frame_z[np.where(mask_nose > 0)]) ** 2)
        intensity_signal_current[1, frame_idx] = np.average(frame_confidence[np.where(mask_forehead > 0)])
        depth_signal_current[1, frame_idx] = np.sqrt(np.average(frame_x[np.where(mask_forehead > 0)]) ** 2 + np.average(frame_y[np.where(mask_forehead > 0)]) ** 2 + np.average(frame_z[np.where(mask_forehead > 0)]) ** 2)
        intensity_signal_current[2, frame_idx] = np.average(frame_confidence[np.where(mask_cheek_and_nose > 0)])
        depth_signal_current[2, frame_idx] = np.sqrt(np.average(frame_x[np.where(mask_cheek_and_nose > 0)]) ** 2 + np.average(frame_y[np.where(mask_cheek_and_nose > 0)]) ** 2 + np.average(frame_z[np.where(mask_cheek_and_nose > 0)]) ** 2)
        intensity_signal_current[3, frame_idx] = np.average(frame_confidence[np.where(mask_left_cheek > 0)])
        depth_signal_current[3, frame_idx] = np.sqrt(np.average(frame_x[np.where(mask_left_cheek > 0)]) ** 2 + np.average(frame_y[np.where(mask_left_cheek > 0)]) ** 2 + np.average(frame_z[np.where(mask_left_cheek > 0)]) ** 2)
        intensity_signal_current[4, frame_idx] = np.average(frame_confidence[np.where(mask_right_cheek > 0)])
        depth_signal_current[4, frame_idx] = np.sqrt(np.average(frame_x[np.where(mask_right_cheek > 0)]) ** 2 + np.average(frame_y[np.where(mask_right_cheek > 0)]) ** 2 + np.average(frame_z[np.where(mask_right_cheek > 0)]) ** 2)
        intensity_signal_current[5, frame_idx] = np.average(frame_confidence[np.where(mask_low_forehead > 0)])
        depth_signal_current[5, frame_idx] = np.sqrt(np.average(frame_x[np.where(mask_low_forehead > 0)]) ** 2 + np.average(frame_y[np.where(mask_low_forehead > 0)]) ** 2 + np.average(frame_z[np.where(mask_low_forehead > 0)]) ** 2)

        # PERCLOS
        landmark_leye = self._ROI_coord_extract(face_landmarks, 'left_eye', self.image_height, self.image_width)
        L_ear = self._eye_aspect_ratio(landmark_leye)
        landmark_reye = self._ROI_coord_extract(face_landmarks, 'right_eye', self.image_height, self.image_width)
        R_ear = self._eye_aspect_ratio(landmark_reye)
        ear_signal_current[frame_idx] = (L_ear + R_ear) /2

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

    def _visualize_FaceMesh(self, frameTrk: np.ndarray, face_landmarks,
                            results_face, mp_drawing_styles,
                            mp_drawing, mp_face_mesh) -> None:
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
        """
        Takes in all face landmarks, which ROI we want, and returns an array of
        pixel coordinates that represent the bounding box polygon of the ROI on the
        image.
        
        Within the function, the bounding box for each ROI is defined by the
        landmark ID numbers that represent vertices on the face mesh. These are
        listed in clockwise order, starting from the top left vertex.
        """
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
        # Landmarks can be found on https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/mesh_map.jpg
        # (old link: https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg)

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
        Takes in a list of 2D (x, y) coordinates that represent the vertices of a polygon.
        Then, creates a temporary image where the polygon is drawn onto.
        Then, returns a numpy array that contains the coordinates of every pixel that is
        inside the polygon.

        :param vtx: list of 2D coordinates of the polygon vertices
        :param image_cols: image columns
        :param image_rows: image rows
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

    myFaceMeshDetector = PhaseTwo(input_dir=os.path.join(skvs_dir, "mat"), output_filename="auto_bfsig")
    myFaceMeshDetector.run(visualize_ROI=False, visualize_FaceMesh=False)
