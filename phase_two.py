import os
import time
import cv2
import numpy as np
from scipy.io import savemat, loadmat
from scipy.spatial import distance as dist
import concurrent.futures
from PIL import Image, ImageDraw

from face_mesh_module import FaceMeshDetector
from chestROIReverseEngineering import ChestROI

from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import scipy

import csv


class PhaseTwo():
    """
    PhaseTwo is a class that performs face detection and landmark tracking using MediaPipe FaceMesh.
    """

    def __init__(self, input_dir: str, output_filename: str, image_width: int = 640, image_height: int = 480, visualize_FaceMesh=False, visualize_ROIs=False, doRR = False):
        """
        Initialize class variables.

        Args:
            input_dir: Directory where input files are located.
            output_filename: Filename of the output .mat file.
            image_width: Width of input image (in pixels) (aka number of columns)
            image_height: Height of input image (in pixels) (aka number of rows)
            visualize_FaceMesh: A boolean indicating whether to visualize the face mesh (the creepy mask-looking thing).
            visualize_ROIs: A boolean indicating whether to visualize the regions of interest.

        Note: The IMX520 sensor has an image resolution of 640x480=307200 pixels per frame (width x height)
        (aka 640 columns x 480 rows)
        
        References:
        - Face Landmarks Key: https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/mesh_map.jpg
        """
        # Directory where input files are located (likely ./skvs/mat/)
        self.input_dir = input_dir

        # Filename of output .mat file (likely auto_bfsig.mat)
        self.output_filename = output_filename
        
        self.image_width = image_width
        self.image_height = image_height

        self.visualize_FaceMesh=visualize_FaceMesh
        self.visualize_ROIs=visualize_ROIs

        self.chest_intensity = []
        self.chest_depth = []
        self.RR = None
        self.doRR = doRR


        # Define the landmarks that represent the vertices of the bounding box for each ROI
        # (used in _get_ROI_bounding_box_pixels())
        self.face_roi_definitions = {
            'nose': np.array([196, 419, 455, 235]),
            'forehead': np.array([109, 338, 9]),
            'cheek_n_nose': np.array([117, 346, 411, 187]), # CNN ROI 1: Gets 51.17 HR
            # 'cheek_n_nose': np.array([116, 340, 433, 213]), # CNN ROI 2: Gets 102.34 HR
            # 'cheek_n_nose': np.array([31, 228, 229, 230, 231, 232, 233, 245, 465, 453, 452, 451, 450, 449, 448, 340, 345, 352, 376, 411, 427, 426, 294, 278, 360, 363, 281, 5, 51, 134, 131, 102, 203, 206, 207, 187, 147, 123, 116, 111]), # CNN ROI 3: Gets 105.35 HR
            'left_cheek': np.array([131, 165, 214, 50]),
            'right_cheek': np.array([372, 433, 358]),
            # 'low_forehead': np.array([109,338,336,107]),
            'low_forehead': np.array([108, 337, 8]),
            # 'low_forehead': np.array([109, 338, 9]),
            'left_eye': np.array([33, 160, 159, 158, 133, 153, 145, 144]),
            'right_eye': np.array([263, 387, 386, 385, 362, 380, 374, 373])
        }

        # These were some other ROIs that were defined, but unused:
        # 'full_face': np.array([55, 285, 455, 366, 137, 235]),
        # 'left_face': np.array([71, 136, 201, 9]),
        # 'chin': np.array([176, 149, 153, 378]),
        # 'palm': np.array([1, 6, 18])

        # Define which ROIs we want to visualize
        self.ROIs_to_visualize = [
            # 'nose',
            # 'forehead',
            'cheek_n_nose',
            # 'left_cheek',
            # 'right_cheek',
            # 'low_forehead',
            # 'left_eye',
            # 'right_eye'
        ]

        # Create thread_pool
        num_threads = self._get_num_threads()
        # num_threads = 1

        if self.visualize_ROIs:
            num_threads = 1

        print(f"Using {num_threads} threads")
        # TODO: Consider disabling mutexes for thread pool and see if that improves performance
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

        # Set flag to indicate whether or not the class has been cleaned up
        # (either manually or automatically if the destructor was called by the garbage
        # collector)
        self.cleaned_up = False

        # Initialize output arrays
        self.intensity_signals = np.array([])
        self.depth_signals = np.array([])
        self.ear_signal = np.array([])

        # Initialize the array containing all 600 cheek_n_nose masks
        # self.cheek_n_nose_masks = np.zeros((600, 480, 640), dtype=np.uint8)
    
    def run(self) -> None:
        """
        Run the face mesh detection and intensity signal extraction.
        """

        num_ROIs = 7

        # Array of intensity signal arrays
        # Each element is (7, num_frames) = (7, 600) for 7 ROIs (regions of interest) and (likely) 600 frames per input video file
        self.intensity_signals = np.zeros((num_ROIs, 1))

        # Array of depth signal arrays
        self.depth_signals = np.zeros((num_ROIs, 1))

        # Array of eye aspect ratio signal values
        self.ear_signal = np.zeros((1))

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
        
        # with my_mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as my_face_mesh:
        
        # Loop through each file
        for filename in filelist:
            file_num = file_num + 1
            
            # Process the file
            self._process_file(file_num, num_files_to_process, filename, num_ROIs, face_mesh_detector)
        
        # # Load Alex data from alex_outputdata.mat
        # alex_outputdata = loadmat("alex_outputdata.mat")
        # self.intensity_signals = alex_outputdata['I_signal']
        # self.depth_signals = alex_outputdata['D_signal']
                    
        self.intensity_signals = np.delete(self.intensity_signals, 0, 1)
        self.depth_signals = np.delete(self.depth_signals, 0, 1)
        self.ear_signal = np.delete(self.ear_signal,0,0)

        with open("depth.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            firstCol = np.array(self.chest_depth)
            firstCol = firstCol.tolist()
            firstCol = [[value] for value in firstCol]  # Convert each value to a single
            writer.writerows(firstCol) 

        if self.doRR:
            self.RR = self.getRespitoryRate(self.chest_depth, outputFile=None, Savgof=False, Lowpass=True, Window=True, realFFT = True)


        # Save average intensities and depths for cheek_n_nose ROI as .mat files

        # Save row 2 of intensity_signals as a .mat file
        # savemat('main_intensity_signals.mat', {'intensity_signals': self.intensity_signals[2]})

        # Save row 2 of depth_signals as a .mat file
        # savemat('main_depth_signals.mat', {'depth_signals': self.depth_signals[2]})

        # Load the tablet_depth_signals.mat into a variable
        # tablet_depth_signals = loadmat('tablet_depth_signals.mat')
        # tablet_depth_signals = tablet_depth_signals['depth_signals']

        # Load the tablet_intensity_signals.mat into a variable
        # tablet_intensity_signals = loadmat('tablet_intensity_signals.mat')
        # tablet_intensity_signals = tablet_intensity_signals['intensity_signals'][0]

        # Set row 2 of intensity_signals to tablet_intensity_signals
        #self.intensity_signals[2] = tablet_intensity_signals

        # Set row 2 of depth_signals to tablet_depth_signals
        # self.depth_signals[2] = tablet_depth_signals

        # Load indices where tablet intensities and main intensities vary by over 1
        # tablet_intensity_diff_indx = loadmat('different_intensity_indx.mat')
        # tablet_intensity_diff_indx = tablet_intensity_diff_indx['idx'][0]

        # Make main intensities equal to tablet intensities where they vary by 1 and over 1
        # for idx in tablet_intensity_diff_indx:
        #     self.intensity_signals[2][idx-1] = tablet_intensity_signals[idx-1] 

        mdic = {"Depth": self.depth_signals, 'I_raw': self.intensity_signals, 'EAR': self.ear_signal} # EAR: eye aspect ratio
        savemat(os.path.join(self.input_dir, self.output_filename + '.mat'), mdic)

        # Save self.cheek_n_nose_masks to a .mat file
        # savemat('main_mask_cheek_n_nose_all_600.mat', {'masks': self.cheek_n_nose_masks})

        return
    
    def _process_file(self, file_num: int, num_files_to_process: int, filename: str, num_ROIs: int,
                  face_mesh_detector: FaceMeshDetector) -> None:
        """
        Processes a single file.proces
        
        Extracts intensity, depth, and eye aspect ratio signals for each region of interest (ROI)
        and updates the corresponding arrays.

        Args:
            file_num: The number of the current file being processed.
            num_files_to_process: The total number of files to be processed.
            filename: The name of the file to be processed.
            num_ROIs: The number of regions of interest (ROIs) for which to extract signals.
            face_mesh_detector: An instance of the FaceMeshDetector class for performing face mesh detection.

        Returns:
            None. Updates the self.intensity_signals, self.depth_signals, and self.ear_signal arrays
            of the class.
        """
        print(f"Processing file {file_num}/{num_files_to_process}: {filename}...")

        
        chest_detector = ChestROI()

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
        intensity_signal_current_file = np.zeros((num_ROIs, num_frames))
        depth_signal_current_file = np.zeros((num_ROIs, num_frames))
        ear_signal_current_file = np.zeros(num_frames)

        # Each array is currently (height*width, num_frames) = (480*640, num_frames) = (307200, num_frames)
        # Reshape to (height, width, num_frames) = (480, 640, num_frames)
        x_all = x_all.reshape([self.image_height, self.image_width, num_frames])
        y_all = y_all.reshape([self.image_height, self.image_width, num_frames])
        z_all = z_all.reshape([self.image_height, self.image_width, num_frames])
        confidence_all = confidence_all.reshape([self.image_height, self.image_width, num_frames])

        # Used to calculate FPS
        previous_time = 0
        start_time = time.time()

        multithreading_tasks = []
        
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

            # Get pixel locations of all face landmarks
            face_detected, landmarks_pixels = face_mesh_detector.find_face_mesh(image=frame_grayscale_rgb, draw=self.visualize_FaceMesh)

            if self.doRR:
                chest_ROIs = chest_detector._Chest_ROI_extract(image=frame_grayscale_rgb, face_landmarks=landmarks_pixels, draw = False)
                self.chestCalculations(chest_ROIs, frame_x, frame_y, frame_z, frame_confidence)

            if face_detected:
                multithreading_tasks.append(self.thread_pool.submit(self._process_face_landmarks, landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, intensity_signal_current_file, depth_signal_current_file, ear_signal_current_file, frame_grayscale_rgb))

            if self.visualize_FaceMesh or self.visualize_ROIs:
                # Calculate and overlay FPS

                current_time = time.time()
                # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
                fps = 1 / (current_time - previous_time)
                previous_time = current_time
                cv2.putText(frame_grayscale_rgb, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                # TODO: Overlay frame number in top right corner
                # cv2.putText(frame_grayscale_rgb, f'Frame {frame_idx + 1}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                text = f'{frame_idx + 1}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
                text_x = frame_grayscale_rgb.shape[1] - text_size[0] - 20  # Position text at the top right corner
                text_y = text_size[1] + 20
                cv2.putText(frame_grayscale_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                # Display frame

                cv2.imshow("Image", frame_grayscale_rgb)
                cv2.waitKey(1)
        
        # Calculate and print average FPS
        end_time = time.time()
        average_fps = num_frames / (end_time - start_time)
        print(f"Average FPS: {average_fps}")

        # Wait for all multithreading tasks to finish
        concurrent.futures.wait(multithreading_tasks)

        self.intensity_signals = np.concatenate((self.intensity_signals, intensity_signal_current_file), axis=1)
        self.depth_signals = np.concatenate((self.depth_signals, depth_signal_current_file), axis=1)
        self.ear_signal = np.concatenate((self.ear_signal, ear_signal_current_file),axis=0)

        # Save average intensities for each frame for cheek_n_nose to a .mat file. This is row idx 2 of the intensity_signals array.
        # savemat("intensity_signals.mat", {"intensity_signals": self.intensity_signals[2, :]})

        return
    
    def chestCalculations(self, corner_landmarks, frame_x, frame_y, frame_z, frame_confidence): 

        #takes corner points of chest ROIs as a 2 by 4 by 2 array where there are 2 ROIs which have 4 points
        #having an x,y coordinate each
        #then takes frame x,y,z data to determine chest depth and chest intensity
        #modifies chest_depth array and chest_intensity array by filling in depths and intensities for each ROI for each frame

        #seperates ROIs and converts to typeinteger
        chest_ROI_upper_pixels = corner_landmarks[0].astype(np.int32) 
        chest_ROI_lower_pixels = corner_landmarks[1].astype(np.int32)

        #gets all pixels within 4 corner outline
        chest_ROI_upper_pixels_in_ROI = self._get_pixels_within_ROI_bounding_box(chest_ROI_upper_pixels)
        chest_ROI_lower_pixels_in_ROI = self._get_pixels_within_ROI_bounding_box(chest_ROI_lower_pixels)

        #calculates intensity
        chest_ROI_upper_average_intensity = np.average(frame_confidence[np.where(chest_ROI_upper_pixels_in_ROI > 0)])
        chest_ROI_lower_average_intensity = np.average(frame_confidence[np.where(chest_ROI_lower_pixels_in_ROI > 0)])

        #calculates depth by using pythagorean theorm using x,y,z coords,
        chest_ROI_upper_average_depth = np.sqrt(
                np.average(frame_x[np.where(chest_ROI_upper_pixels_in_ROI > 0)]) ** 2 +
                np.average(frame_y[np.where(chest_ROI_upper_pixels_in_ROI > 0)]) ** 2 +
                np.average(frame_z[np.where(chest_ROI_upper_pixels_in_ROI > 0)]) ** 2
            )
        chest_ROI_lower_average_depth = np.sqrt(
                np.average(frame_x[np.where(chest_ROI_lower_pixels_in_ROI > 0)]) ** 2 +
                np.average(frame_y[np.where(chest_ROI_lower_pixels_in_ROI > 0)]) ** 2 +
                np.average(frame_z[np.where(chest_ROI_lower_pixels_in_ROI > 0)]) ** 2
            )
        
        #adds to arrays
        self.chest_depth.append(chest_ROI_lower_average_depth)
        #self.chest_depth.append([chest_ROI_lower_average_depth, chest_ROI_upper_average_depth])
        self.chest_intensity.append([chest_ROI_lower_average_intensity, chest_ROI_upper_average_intensity])

        #print(f"Chest ROI upper average intensity: {chest_ROI_upper_average_intensity}")
        #print(f"Chest ROI lower average intensity: {chest_ROI_lower_average_intensity}")
        #print(f"Chest ROI upper average depth: {chest_ROI_upper_average_depth}")
        #print(f"Chest ROI lower average depth: {chest_ROI_lower_average_depth}")

    def apply_lowpass_filter(self, signal, cutoff_frequency, sampling_rate, filter_order=5):
        nyquist_frequency = 0.5 * sampling_rate
        normalized_cutoff = cutoff_frequency / nyquist_frequency
        b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal


    def getRespitoryRate(self, data, outputFile=None, Savgof=False, Lowpass=True, Window=False, realFFT = True):
        num_frames = len(data)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        time = np.linspace(0, 60, len(data))

        axes[0].plot(time, data)
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Chest Depth")

        if Savgof:
            data = scipy.signal.savgol_filter(
                data, 79, 2, mode='nearest')

        if Lowpass:
            data = self.apply_lowpass_filter(data, 1, 30)

        if Window:
            min_val = np.min(data)
            max_val = np.max(data)
            data = (data - min_val) / (max_val - min_val)

            window = np.hanning(num_frames)
            data = data * window

        axes[1].plot(time, data)
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Chest Depth")

        T = 1/30
        if realFFT:
            yf_rr1 = abs(np.fft.rfft(data))
        # yf_rr1 = 2.0 / num_frames * yf_rr1[:num_frames // 2]
        # xf_rr1 = np.linspace(0.0, 1.0 / (2.0 * T), num_frames // 2)
            N = data.size
            xf_rr1 = np.fft.rfftfreq(N, d=T)
            xf_rr1 = xf_rr1 * 60
        #print(xf_rr1)

        else:
            yf_rr1 = abs(np.fft.fft(data))
            yf_rr1 = 2.0 / num_frames * yf_rr1[:num_frames // 2]
            xf_rr1 = np.linspace(0.0, 1.0 / (2.0 * T), num_frames // 2)      

        yf_rr1[np.where(xf_rr1 <= 5)] = 0
        yf_rr1[np.where(xf_rr1 >= 30)] = 0

        axes[2].plot(xf_rr1, yf_rr1)
        axes[2].set_xlim((5,30))
        #plt.ylim((0,2))
        axes[2].set_xlabel("Frequency (Breaths Per Minute)")
        axes[2].set_ylabel("Fourier Magnitude")


        peaks, properties = scipy.signal.find_peaks(yf_rr1)
        max_index = np.argmax(yf_rr1[peaks])

        RR = xf_rr1[peaks[max_index]]

        print("Respiratory Rate Measured", RR)

        if outputFile:
            plt.savefig(outputFile)

        else:
            pass

        return RR
    
    def _process_face_landmarks(
        self,
        landmarks_pixels: np.ndarray,
        frame_idx: int,
        frame_x: np.ndarray,
        frame_y: np.ndarray,
        frame_z: np.ndarray,
        frame_confidence: np.ndarray,
        intensity_signal_current_file: np.ndarray,
        depth_signal_current_file: np.ndarray,
        ear_signal_current_file: np.ndarray,
        frame_grayscale_rgb: np.ndarray
    ) -> None:
        """
        Processes the face landmarks for a single frame.

        Args:
            landmarks_pixels: An array of shape (468, 2) representing the pixel coordinates (x, y) for each of the
                468 total face landmarks detected. The i-th row corresponds to the i-th landmark.
            frame_idx: The index of the current frame.
            frame_x: An (n,d) array of x-coordinates.
            frame_y: An (n,d) array of y-coordinates.
            frame_z: An (n,d) array of z-coordinates.
            frame_confidence: An (n,d) array of confidence values.
            intensity_signal_current_file: An (n,d) array to store the intensity signals for all ROIs for all frames of this video clip.
            depth_signal_current_file: An (n,d) array to store the depth signals for all ROIs for all frames of this video clip.
            ear_signal_current_file: An (n,) array to store the eye aspect ratio (EAR) signals for all frames of this video clip.
            frame_grayscale_rgb: An (n,d,3) array representing the current frame in RGB format.

        Returns:
            None. The results are stored in the output arrays `intensity_signal_current_file`, `depth_signal_current_file`,
            and `ear_signal_current_file`.

        Description:
            This function processes the face landmarks for a single frame. It calculates the eye aspect
            ratio (EAR) for the left and right eyes, and the averaged intensity and depth signals for each
            region of interest (ROI) defined in the `self.face_roi_definitions` dictionary.
            The function iterates over each ROI, retrieves the bounding box pixels for the ROI from the
            `landmarks_pixels` array, and calculates the intensity and depth signals based on the
            corresponding pixel values in the frame. The results are then stored in the output arrays
            `intensity_signal_current_file` and `depth_signal_current_file`.
            The eye aspect ratio signals are calculated for the left and right eyes separately using the
            `self._get_eye_aspect_ratio` function, and the averaged value is stored in
            `ear_signal_current_file`.
        
        NOTE: If visualize_ROIs is True, all bounding boxes listed in `self.ROIs_to_visualize` will be drawn on the frame. After
        each ROI is drawn, the program will pause until the user presses any key. Pressing any key will continue to draw the next ROI.
        By drawing all ROIs on the same image, the user can see the relative sizes and positions of the ROIs, as well as the overall
        coverage of the face across all ROIs.

        If the user wants to visualize only a subset of the ROIs, they can set `self.ROIs_to_visualize` to a list of the ROIs they want
        to visualize. For example, if `self.ROIs_to_visualize = ["left_eye", "right_eye"]`, only the left and right eye ROIs will be
        drawn on the frame.
        """
        # Variables for calculating eye aspect ratio
        left_eye_aspect_ratio = 0.0
        right_eye_aspect_ratio = 0.0
        frame_with_ROIs_drawn = np.array([])

        if self.visualize_ROIs and len(self.ROIs_to_visualize) > 0:
            # Make a copy of the frame to draw ROIs on
            frame_with_ROIs_drawn = frame_grayscale_rgb.copy()

        # Loop through each ROI
        for roi_idx, roi_name in enumerate(self.face_roi_definitions.keys()):
            # Get bounding box of ROI in pixels
            roi_bounding_box_pixels = self._get_ROI_bounding_box_pixels(landmarks_pixels, roi_name)
            
            if self.visualize_ROIs and roi_name in self.ROIs_to_visualize:
                # Draw bounding box of ROI
                self._draw_ROI_bounding_box(roi_bounding_box_pixels, frame_with_ROIs_drawn, roi_name)

            if roi_name == "left_eye":
                # Calculate and save eye aspect ratio for the ROI
                left_eye_aspect_ratio = self._get_eye_aspect_ratio(roi_bounding_box_pixels)
            elif roi_name == "right_eye":
                # Calculate and save eye aspect ratio for the ROI
                right_eye_aspect_ratio = self._get_eye_aspect_ratio(roi_bounding_box_pixels)
            else:
                # Get pixels contained within ROI bounding box
                pixels_in_ROI = self._get_pixels_within_ROI_bounding_box(roi_bounding_box_pixels)

                # if roi_name == "cheek_n_nose" and frame_idx == 42:
                #     # Save the mask of pixels contained within the ROI bounding box to a .mat file
                #     savemat("main_mask_cheek_n_nose_43.mat", {"mask_cheek_n_nose_43": pixels_in_ROI})

                # if roi_name == "cheek_n_nose":
                #     # Save the mask of pixels contained within the ROI bounding box to self.cheek_n_nose_masks
                #     self.cheek_n_nose_masks[frame_idx] = pixels_in_ROI
                
                # Calculate and save averaged intensity for the ROI
                intensity_signal_current_file[roi_idx, frame_idx] = np.average(frame_confidence[np.where(pixels_in_ROI > 0)])
                
                # Calculate and save averaged depth for the ROI
                depth_signal_current_file[roi_idx, frame_idx] = np.sqrt(
                    np.average(frame_x[np.where(pixels_in_ROI > 0)]) ** 2 +
                    np.average(frame_y[np.where(pixels_in_ROI > 0)]) ** 2 +
                    np.average(frame_z[np.where(pixels_in_ROI > 0)]) ** 2
                )
        
        # Calculate and save eye aspect ratio for the ROI
        ear_signal_current_file[frame_idx] = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2
        
        return
    
    def _get_ROI_bounding_box_pixels(self, landmarks_pixels: np.ndarray, roi_name: str) -> np.ndarray:
        """
        Takes in the pixel coordinates of all face landmarks and returns the pixel coordinates
        of the face landmarks that represent the bounding box for the requested ROI.

        Args:
            landmarks_pixels (np.ndarray): An array of shape (468, 2) representing the pixel coordinates (x, y)
                for each of the 468 total face landmarks detected. The i-th row corresponds to the i-th landmark
                (zero-indexed, so row 0 is landmark 1).
            roi_name (str): The name of the requested ROI. Must be one of the predefined ROIs in
                `self.face_roi_definitions`.

        Returns:
            np.ndarray: An array of shape (n, 2), where n is the number of landmarks
                that form the bounding box for the requested ROI. Each row represents the (x, y)
                coordinates of a landmark pixel.

        Raises:
            KeyError: If the provided roi_name does not match any of the predefined ROIs.

        Note:
            The returned bounding_box_pixels is in the same format as the input landmarks_pixels,
            with each array of shape (2,) representing the (x, y) coordinates of a landmark pixel.
        """
        bounding_box_pixels = np.array([])
        
        try:
            landmark_indices = self.face_roi_definitions[roi_name]
            bounding_box_pixels = landmarks_pixels[landmark_indices]
        except KeyError:
            raise KeyError(f"ERROR: The provided roi_name \"{roi_name}\" does not match any of the predefined ROIs.")

        # if roi_name == 'cheek_n_nose':
        #     bounding_box_pixels[2][1] = bounding_box_pixels[2][1] - 2
        #     bounding_box_pixels[3][1] = bounding_box_pixels[3][1] - 2
        #     bounding_box_pixels[1][1] = bounding_box_pixels[1][1] + 2
        #     bounding_box_pixels[0][1] = bounding_box_pixels[0][1] + 2
        #     # bounding_box_pixels[0][0] = bounding_box_pixels[0][0] + 3
        #     # bounding_box_pixels[1][0] = bounding_box_pixels[1][0] - 3
        #     # bounding_box_pixels[2][0] = bounding_box_pixels[2][0] - 3
        #     # bounding_box_pixels[3][0] = bounding_box_pixels[3][0] + 3

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

        # Reformat to a list of 2-tuples
        pixels_passed_in = list(map(tuple, bounding_box_pixels.tolist()))

        # Draw a polygon on the mask_canvas using the ROI bounding box pixel coordinates
        # The polygon will be filled in with pixels with value 1, and the outline will
        # be 1 pixel wide with a value of 1 as well.
        ImageDraw.Draw(mask_canvas).polygon(pixels_passed_in, fill=1, outline=1, width=1)

        # Convert the mask_canvas image, with the filled-in polygon on it, to a numpy array
        # The array will have a shape of (self.image_height, self.image_width)
        pixels_in_ROI = np.array(mask_canvas)

        # # Display the image using matplotlib
        # plt.imshow(mask_canvas, cmap='gray')
        # plt.show()

        return pixels_in_ROI

        # # Create an empty binary mask with the same shape as the frame image
        # mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        # # Draw a filled polygon on the mask using the ROI bounding box pixel coordinates
        # cv2.fillPoly(mask, [bounding_box_pixels], color=1)

        # # Convert the mask to a binary array
        # pixels_in_ROI = mask.astype(np.uint8)

        # # # Display the image using matplotlib
        # # plt.imshow(mask, cmap='gray')
        # # plt.show()

        # return pixels_in_ROI

    def _get_eye_aspect_ratio(self, eye_bounding_box_pixels: np.ndarray) -> float:
        """
        Calculates the Eye Aspect Ratio (EAR) for an eye region represented by a bounding box.

        Args:
            eye_bounding_box_pixels: An array of shape (n, 2) containing the pixel coordinates
                (x, y) of the landmarks that form the bounding box for the eye ROI. Each row
                represents the (x, y) coordinates of a landmark pixel.

        Returns:
            The Eye Aspect Ratio (EAR) as a floating-point value.

        Description:
            The function takes a list of 2D (x, y) pixel coordinates that represent the vertices
            of the bounding box for a region of interest that represents an eye. It calculates
            the distances between certain landmarks of the eye, both vertically and
            horizontally, using the Euclidean distance formula. Finally, it computes the Eye
            Aspect Ratio (EAR) by applying the "eye aspect ratio equation".

            The function returns the calculated EAR as a floating-point value.
        """
        
        # TODO: See if this reference can help us work on this feature:
        # https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/

        # Vertical eye landmarks
        distance_a = dist.euclidean(eye_bounding_box_pixels[1], eye_bounding_box_pixels[7])
        distance_b = dist.euclidean(eye_bounding_box_pixels[2], eye_bounding_box_pixels[6])
        distance_c = dist.euclidean(eye_bounding_box_pixels[3], eye_bounding_box_pixels[5])

        # Horizontal eye landmarks
        distance_d = dist.euclidean(eye_bounding_box_pixels[0], eye_bounding_box_pixels[4])

        # Calculate eye aspect ratio using "eye aspect ratio equation"
        eye_aspect_ratio_value = (distance_a + distance_b + distance_c) / (3.0 * distance_d)

        return eye_aspect_ratio_value

    def _draw_ROI_bounding_box(self, bounding_box_pixels: np.ndarray, frame_grayscale_rgb: np.ndarray, roi_name: str) -> None:
        """
        Draws the bounding box for an ROI on the frame image along with the ROI name.

        Args:
            bounding_box_pixels (np.ndarray): An array of shape (n, 2), where n is the number of landmarks
                that form the bounding box for the requested ROI. Each row represents the (x, y)
                coordinates of a landmark pixel.
            frame_grayscale_rgb (np.ndarray): The frame image in grayscale RGB format.
            roi_name (str): The name of the requested ROI. This is the text that will be displayed
                above the bounding box.

        Returns:
            None
        
        NOTE: After drawing the bounding box and displaying the image, the function will wait for
        the user to press any key before continuing.

        NOTE: The usage of this function requires multi-threading to be disabled (aka num_threads=1).
        """

        # Draw the bounding box on the frame image
        cv2.polylines(frame_grayscale_rgb, [bounding_box_pixels], isClosed=True, color=(0, 0, 255), thickness=2)

        # Add roi_name as text above the bounding box
        text_position = (bounding_box_pixels[0, 0], bounding_box_pixels[0, 1] - 10)
        cv2.putText(frame_grayscale_rgb, roi_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the image
        cv2.imshow("ROI Bounding Boxes", frame_grayscale_rgb)
        cv2.waitKey(0)
    
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
        x_all, y_all, z_all, confidence_all = np.array([]), np.array([]), np.array([]), np.array([])

        NUM_FRAMES_PER_FILE = 600

        with open(filepath, 'rb') as binary_file:
            x_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
            y_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
            z_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
            confidence_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()

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
    
    def clean_up(self):
        """
        Clean up class variables.

        NOTE: This should be called when the class is no longer needed. If this is not
        called, the garbage collector will attempt to call the destructor automatically, but
        it should not be relied on.
        """
        self.__del__()

    def __del__(self):
        """
        Destructor to clean up class variables.

        NOTE: This should be called automatically by the garbage collector when the object
        is no longer needed, however it may be unreliable. It is recommended to call
        self.clean_up() manually when the class is no longer needed.
        """
        if not self.cleaned_up:
            # Shut down the thread pool
            self.thread_pool.shutdown(wait=True, cancel_futures=False)

            self.cleaned_up = True
    
    def _get_num_threads(self) -> int:
        """
        Get the number of threads to use for multi-threading. If the CPU does not support
        multi-threading, this will return 1.

        Returns:
            The number of threads to use for multi-threading.
        """
        
        # Get the number of available threads
        num_threads = os.cpu_count()

        # If the CPU does not support multithreading, set num_threads to 1 (single-threaded)
        if num_threads is None:
            num_threads = 1
        elif num_threads < 1:
            num_threads = 1
        
        return num_threads

if __name__ == "__main__":
    skvs_dir = os.path.join(os.getcwd(), 'skvs')

    myFaceMeshDetector = PhaseTwo(input_dir=os.path.join(skvs_dir, "mat"), output_filename="auto_bfsig", visualize_FaceMesh=False, visualize_ROIs=False)
    myFaceMeshDetector.run()
    myFaceMeshDetector.clean_up()