import os
import sys
import subprocess

# facial_skBF_facemeshTrak.py
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import scipy
import math
import mat73
from scipy.io import savemat, loadmat

import cv2
from typing import ChainMap, Tuple, Union
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import time
import os
from matplotlib.widgets import RectangleSelector
import shutil

# import tensorflow as tf

# facial_skBF_facemeshTrak.py
class FaceMeshDetector():
    def __init__(self, input_mats_dir, output_bfsig_name):
        self.input_mats_dir = input_mats_dir
        self.ouput_bfsig_name = output_bfsig_name
    
    def run(self):
        # matpath = 'C:/Users/MNI Lab/Documents/GitHub/prgrm/facial-blood-ToF/skv_mat/'
        # matname = '20221003-Sparsh_Fex_bfsig'

        # matpath = './skvs/mat/'
        # matname = '20221003-Sparsh_Fex_bfsig'

        matpath = self.input_mats_dir
        matname = self.ouput_bfsig_name

        # parameters initialization
        
        start_frame = 0
        img_cols = 640
        img_rows = 480
        I_signal = np.zeros((7, 1))
        D_signal = np.zeros((7, 1))
        EAR = np.zeros((1))
        filelist = []
        for filename in os.listdir(matpath):
            if filename.endswith('.skv.mat'):
                filelist.append(filename)
        
        counter = 0
        
        # defining mediapipe detector
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        mp_hands = mp.solutions.hands
        
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            with mp_hands.Hands(
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
        
                time1 = time.time()
        
                # load and process every .mat file, tracking done by mediapipe
        
                for filename in filelist:


                    try:
                        # mat_data = mat73.loadmat(matpath + filename)
                        mat_data = mat73.loadmat(os.path.join(matpath, filename))
                    except:
                        # mat_data = loadmat(matpath + filename)
                        mat_data = loadmat(os.path.join(matpath, filename))
                    print(filename)
                    gray_all = mat_data['grayscale']
                    x_all = mat_data['x_value']
                    y_all = mat_data['y_value']
                    z_all = mat_data['z_value']

                    counter = counter + 1
                    print(counter)
                    
                    # initializing output and intermediate variables
                    frame_num = np.size(gray_all, 1)
                    I_signal_current = np.zeros((7,
                                                frame_num))  # 1: nose;  2: forehead;   3: nose & cheek  4: left cheek   5: right cheek  6: lower forehead  7: palm
                    D_signal_current = np.zeros((7, frame_num))
                    EAR_current = np.zeros((frame_num))
                    time2 = time.time()
                    timeelps = time2 - time1
                    print('Data loading completed, time elapsed: %s seconds' % timeelps)

                    gray_all = np.reshape(gray_all, [img_rows, img_cols, frame_num])
                    x_all = np.reshape(x_all, [img_rows, img_cols, frame_num])
                    y_all = np.reshape(y_all, [img_rows, img_cols, frame_num])
                    z_all = np.reshape(z_all, [img_rows, img_cols, frame_num])

                    frame_num = np.size(gray_all, 2)
                    # tracking and extracting I, D frame by frame
                    for j in range(frame_num):
                        frameTrk = self._mp_preprocess(gray_all[:, :, j])
                        frameSig = gray_all[:, :, j]
                        xSig = x_all[:, :, j].astype('int32')
                        ySig = y_all[:, :, j].astype('int32')
                        zSig = z_all[:, :, j].astype('int32')
                        # To improve performance, optionally mark the image as not writeable to
                        # pass by reference.
                        frameTrk.flags.writeable = False
                        frameTrk = cv2.cvtColor(frameTrk, cv2.COLOR_BGR2RGB)
                        results_face = face_mesh.process(frameTrk)
                        results_hand = hands.process(frameTrk)

                        if results_face.multi_face_landmarks:
                            # print('results_face.multi_face_landmarks')
                            # print(len(results_face.multi_face_landmarks))
                            face_landmarks = results_face.multi_face_landmarks[0]

                            # find the ROI vertices
                            landmark_forehead = self._ROI_coord_extract(face_landmarks, 'forehead', img_rows, img_cols)
                            mask_forehead = self._vtx2mask(landmark_forehead, img_cols, img_rows)
                            landmark_nose = self._ROI_coord_extract(face_landmarks, 'nose', img_rows, img_cols)
                            mask_nose = self._vtx2mask(landmark_nose, img_cols, img_rows)
                            landmark_cn = self._ROI_coord_extract(face_landmarks, 'cheek_n_nose', img_rows, img_cols)
                            mask_cn = self._vtx2mask(landmark_cn, img_cols, img_rows)
                            landmark_lc = self._ROI_coord_extract(face_landmarks, 'left_cheek', img_rows, img_cols)
                            mask_lc = self._vtx2mask(landmark_lc, img_cols, img_rows)
                            landmark_rc = self._ROI_coord_extract(face_landmarks, 'right_cheek', img_rows, img_cols)
                            mask_rc = self._vtx2mask(landmark_rc, img_cols, img_rows)
                            landmark_lf = self._ROI_coord_extract(face_landmarks, 'low_forehead', img_rows, img_cols)
                            mask_lf = self._vtx2mask(landmark_lf, img_cols, img_rows)

                            # if results_hand.multi_hand_landmarks:
                            #     hand_landmarks = results_hand.multi_hand_landmarks[0]
                            #     landmark_palm =  ROI_coord_extract(hand_landmarks, 'palm', img_rows, img_cols)
                            #     mask_palm = vtx2mask(landmark_palm, img_cols, img_rows)
                            #
                            #     I_signal_current[6,j] = np.average(frameSig[np.where(mask_palm > 0)])
                            #     D_signal_current[6,j] = np.sqrt(np.average(xSig[np.where(mask_palm > 0)])**2 + np.average(ySig[np.where(mask_palm > 0)])**2+np.average(zSig[np.where(mask_palm > 0)])**2)

                            # calculate averaged I and D
                            I_signal_current[0, j] = np.average(frameSig[np.where(mask_nose > 0)])
                            D_signal_current[0, j] = np.sqrt(np.average(xSig[np.where(mask_nose > 0)]) ** 2 + np.average(
                                ySig[np.where(mask_nose > 0)]) ** 2 + np.average(zSig[np.where(mask_nose > 0)]) ** 2)
                            I_signal_current[1, j] = np.average(frameSig[np.where(mask_forehead > 0)])
                            D_signal_current[1, j] = np.sqrt(np.average(xSig[np.where(mask_forehead > 0)]) ** 2 + np.average(
                                ySig[np.where(mask_forehead > 0)]) ** 2 + np.average(zSig[np.where(mask_forehead > 0)]) ** 2)
                            I_signal_current[2, j] = np.average(frameSig[np.where(mask_cn > 0)])
                            D_signal_current[2, j] = np.sqrt(np.average(xSig[np.where(mask_cn > 0)]) ** 2 + np.average(
                                ySig[np.where(mask_cn > 0)]) ** 2 + np.average(zSig[np.where(mask_cn > 0)]) ** 2)
                            I_signal_current[3, j] = np.average(frameSig[np.where(mask_lc > 0)])
                            D_signal_current[3, j] = np.sqrt(np.average(xSig[np.where(mask_lc > 0)]) ** 2 + np.average(
                                ySig[np.where(mask_lc > 0)]) ** 2 + np.average(zSig[np.where(mask_lc > 0)]) ** 2)
                            I_signal_current[4, j] = np.average(frameSig[np.where(mask_rc > 0)])
                            D_signal_current[4, j] = np.sqrt(np.average(xSig[np.where(mask_rc > 0)]) ** 2 + np.average(
                                ySig[np.where(mask_rc > 0)]) ** 2 + np.average(zSig[np.where(mask_rc > 0)]) ** 2)
                            I_signal_current[5, j] = np.average(frameSig[np.where(mask_lf > 0)])
                            D_signal_current[5, j] = np.sqrt(np.average(xSig[np.where(mask_lf > 0)]) ** 2 + np.average(
                                ySig[np.where(mask_lf > 0)]) ** 2 + np.average(zSig[np.where(mask_lf > 0)]) ** 2)

                            # PERCLOS
                            landmark_leye = self._ROI_coord_extract(face_landmarks, 'left_eye', img_rows, img_cols)
                            L_ear = self._eye_aspect_ratio(landmark_leye)
                            landmark_reye = self._ROI_coord_extract(face_landmarks, 'right_eye', img_rows, img_cols)
                            R_ear = self._eye_aspect_ratio(landmark_reye)
                            EAR_current[j] = (L_ear + R_ear) /2


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

                    I_signal = np.concatenate((I_signal, I_signal_current), axis=1)
                    D_signal = np.concatenate((D_signal, D_signal_current), axis=1)
                    EAR = np.concatenate((EAR, EAR_current),axis=0)
        I_signal = np.delete(I_signal, 0, 1)
        D_signal = np.delete(D_signal, 0, 1)
        EAR = np.delete(EAR,0,0)
        mdic = {"Depth": D_signal, 'I_raw': I_signal, 'EAR': EAR} # EAR: eye aspect ratio
        savemat(os.path.join(matpath, matname + '.mat'), mdic)
        
        print('finished')

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
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

        image_3chnl = np.stack((image,) * 3, axis=-1)
        image_3chnl = cv2.convertScaleAbs(image_3chnl)
        shoulder_landmark = [11, 12]
        landmark_px_rr = np.zeros([2, 4, 2])
        image_height, image_width = image.shape
        results = pose.process(image_3chnl)
        body_points = results.pose_landmarks.landmark
        shoulder_point_l = self._normalized_to_pixel_coordinates(body_points[11].x, body_points[11].y, image_width, image_height)
        shoulder_point_r = self._normalized_to_pixel_coordinates(body_points[12].x, body_points[12].y, image_width, image_height)

        shoulder_x = (shoulder_point_l[0] + shoulder_point_r[0]) / 2
        shoulder_y = (shoulder_point_l[1] + shoulder_point_r[1]) / 2

        neck_width = 2 * np.abs(chin_location[1][0] - chin_location[3][0])
        neck_height = 0.5 * np.abs(shoulder_y - chin_location[2][1])

        chest_width = np.abs(shoulder_point_l[0] - shoulder_point_r[0])
        chest_height = 0.22 * chest_width

        landmark_px_rr[0, :, 0] = [
            shoulder_x - 0.5 * neck_width,
            shoulder_x + 0.5 * neck_width,
            shoulder_x + 0.5 * neck_width,
            shoulder_x - 0.5 * neck_width
        ]

        landmark_px_rr[0, :, 1] = [
            shoulder_y - 1.1 * neck_height,
            shoulder_y - 1.1 * neck_height,
            shoulder_y - 0.1 * neck_height,
            shoulder_y - 0.1 * neck_height,
        ]

        # landmark_px_rr[0,:,0]=[chin_location[1][0]-0.8*neck_width,chin_location[3][0],
        #                       chin_location[3][0],chin_location[1][0]-0.8*neck_width]
        # landmark_px_rr[0,:,1]=[chin_location[1][1]+10,chin_location[3][1]+10,
        #                       chin_location[3][1]+neck_height,chin_location[1][1]+neck_height]

        landmark_px_rr[1, :, 0] = [
            shoulder_x - 0.3 * chest_width,
            shoulder_x + 0.3 * chest_width,
            shoulder_x + 0.3 * chest_width,
            shoulder_x - 0.3 * chest_width
        ]

        landmark_px_rr[1, :, 1] = [
            shoulder_y,
            shoulder_y,
            shoulder_y + chest_height,
            shoulder_y + chest_height
        ]

        # landmark_px_rr[1,:,0]=[shoulder_point_l[0]-25,shoulder_point_r[0]+25,
        #                       shoulder_point_r[0]+25,shoulder_point_l[0]-25]
        # landmark_px_rr[1,:,1]=[shoulder_point_l[1],shoulder_point_r[1],
        #                       shoulder_point_r[1]+chest_height,shoulder_point_l[1]+chest_height]

        np.clip(landmark_px_rr[0, :, 0], 0, image_width)
        np.clip(landmark_px_rr[0, :, 1], 0, image_height)
        np.clip(landmark_px_rr[1, :, 0], 0, image_width)
        np.clip(landmark_px_rr[1, :, 1], 0, image_height)

        if plot:
            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.scatter(chin_location[1][0], chin_location[1][1], s=12, c='green', marker='x')
            plt.scatter(chin_location[3][0], chin_location[3][1], s=12, c='green', marker='x')

            plt.scatter(shoulder_point_l[0], shoulder_point_l[1], s=6, c='green', marker='o')
            plt.scatter(shoulder_point_r[0], shoulder_point_r[1], s=6, c='green', marker='o')
            for j in range(4):
                plt.scatter(landmark_px_rr[0][j][0], landmark_px_rr[0][j][1], s=8, c='red', marker='x')
                plt.scatter(landmark_px_rr[1][j][0], landmark_px_rr[1][j][1], s=1, c='black', marker='o')
                # plt.plot((landmark_px[k][j-1][0], landmark_px[k][j][0]),(landmark_px[k][j-1][1],landmark_px[k][j][1]), c='g', linewidth=1)
        plt.show()
        return landmark_px_rr

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

    def _mp_preprocess(self, frameTrk, divisor=4):
        frameTrk = frameTrk.astype(float)
        frameTrk = frameTrk / divisor
        frameTrk[np.where(frameTrk > 255)] = 255
        frameTrk = frameTrk.astype('uint8')
        image_3chnl = np.stack((frameTrk,) * 3, axis=-1)

        return image_3chnl

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




# - Use Automotive Suite to record a video clip (.skv file)
# - Copy skv file(s) to /skvs/
# - Open imx520_sample project in Visual Studio (use the long .exe profile)
# 	- We are debugging imx520_sample.exe (C:\Users\MNI Lab\Documents\GitHub\prgrm\facial-blood-ToF\imx520_sample\.build\output\imx520_sample.exe)
# - Run facial_skBF_facemeshTrak.py
# 	- Outputs to facial-blood-ToF/skv_mat/20221003-Sparsh_Fex_bfsig.mat
# - copy the bfsig output file to matlab_code
# 	- where??
# - open process_thermal.m
# - this outputs charts of the heart rate




# Record skv video file
#   - Open Automotive Suite to let user manually record a video clip (.skv file)

def record_skv():
    """
        Launch Automotive Suite and wait until the user exits it.
        Once the user is done recording, they will close the program.
        After it is closed, move all of the newly recorded files to ./skvs/ and
        return the absolute path to ./skvs.
        If no new files, terminate the script.
    """
    # Find the folder path that matches the pattern "./automotive_suite_recorder_viewer*"
    # print(os.getcwd())
    folder_name = next(filter(lambda x: x.startswith('automotive_suite_recorder_viewer'), os.listdir()), None)
    # print(f"folder_name: {folder_name}")
    
    if not folder_name:
        print('Automotive Suite not found at automotive_suite_recorder_viewer*/automotive_suite.exe')
        sys.exit()
    else:
        folder_path = os.path.join(os.getcwd(), folder_name)
        exe_path = os.path.join(folder_path, 'automotive_suite.exe')
        # print(f"folder_path: {folder_path}")
        # print(f"exe_path: {exe_path}")
    
        # Get set of files in ./automotive_suite_recorder_viewer*
        #   - Put all *.skv filenames and the datetime they were created in a set of tuples {("filename.skv", datetime_created))}
        #   - Ignore folders

        recordings_path = os.path.join(folder_path, 'RecordedMovies')

        skvs_before_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(recordings_path, x))), os.listdir(recordings_path))))

        # skvs_before_recording = set()
        # for filename in os.listdir(recordings_path):
        #     print(f"filename: {filename}")
        #     if os.path.isfile(os.path.join(recordings_path, filename)) and filename.endswith('.skv'):
        #         # get the datetime the file was created
        #         datetime_created = os.path.getctime(os.path.join(recordings_path, filename))
        #         # add the filename and datetime_created to the set as a tuple
        #         skvs_before_recording.add((filename, datetime_created))




        # print("skvs before recording")
        # print(skvs_before_recording)
    
        # Launch the program and wait until the user exits it
        process = subprocess.run(exe_path, shell=True)
    
        skvs_after_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(recordings_path, x))), os.listdir(recordings_path))))
        # print("skvs after recording")
        # print(skvs_after_recording)
    
        # If sets are different, then new .skv file(s) were created
        # skvs_before_recording - skvs_after_recording = set of new .skv files
        if skvs_before_recording != skvs_after_recording:
            # Move the new .skv file(s) to ./skvs/
            new_skvs = skvs_after_recording - skvs_before_recording
            # print("new_skvs")
    
            # # Get absolute path to the new .skv file
            # new_skv_path = os.path.join(recordings_path, new_skv[0])

            for new_skv in new_skvs:
                # print(new_skv)
                new_skv_path = os.path.join(recordings_path, new_skv[0])
                # print(new_skv_path)
                shutil.move(new_skv_path, os.path.join(os.getcwd(), 'skvs'))
                print('Automotive Suite recorded new file: ' + new_skv_path)
            
            skv_dir = os.path.join(os.getcwd(), 'skvs')
    
            return skv_dir
        else:
            print('No new .skv file was recorded')
            sys.exit()

# Launch ./automotive_suite_recorder_viewer_v0.0.0/automotive_suite.exe
# Wait until the user exits the program (automotive_suite.exe) and then continue

# # Launch the program
# process = subprocess.Popen("./automotive_suite_recorder_viewer_v0.0.0/automotive_suite.exe")

# # Wait for the program to exit
# process.wait()


#   - Once the user is done recording, they will close the program
#   - After it is closed, the .skv file will be processed into a .mat file


# Convert .skv video file into .mat file
#   - imx520_sample.exe -i skvs/input_skv.skv -o skvs/mat/output.mat


# Generate bloodflow signature .mat file
#   - facial_skBF_facemeshTrak.py
#   - Outputs to facial-blood-ToF/skv_mat/20221003-Sparsh_Fex_bfsig.mat


# Generate plots of heart rate from bloodflow signature .mat file
#   - process_thermal.m
#   - Outputs plots of the heart rate

def check_for_skvs(skvs_dir):
    skvs_before_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(skvs_dir, x))), os.listdir(skvs_dir))))

    # if skvs_before_recording empty, exit
    if not skvs_before_recording:
        print('No .skv files found in ./skvs/')
        sys.exit()

def skv_to_mat(skvs_dir):
    ## For each .skv file in skvs_dir, convert to .mat file using imx520_sample.exe and save to ./skvs/mat/
    #for skv_filename in os.listdir(skvs_dir):
    #    # print("skv_filename:")
    #    # print(skv_filename)
    #    skv_path = os.path.join(skvs_dir, skv_filename)
    #    # print("skv_path:")
    #    # print(skv_path)
    #
    #    # Convert .skv video file into .mat file
    #    # Get absolute path to imx520_sample.exe
    #    imx520_sample_exe_path = os.path.join(os.getcwd(), "skv_to_mat/compiled_releases/r1/imx520_sample.exe")
    #    # Get absolute path to output .mat file
    #    output_mat_path = os.path.join(os.getcwd(), "skvs/mat/" + skv_filename + ".mat")
    #    # Run imx520_sample.exe
    #    process = subprocess.run([imx520_sample_exe_path, "-i", skv_path, "-o", output_mat_path], shell=True)

    # Convert all .skv video files in ./skvs/ into .mat files using imx520_sample.exe and save to ./skvs/mat/
    # Get absolute path to imx520_sample.exe
    imx520_sample_exe_path = os.path.join(os.getcwd(), "skv_to_mat/compiled_releases/r2_2/imx520_sample.exe")
    # Get absolute path to dir to save output .mat files to
    output_mat_dir = os.path.join(skvs_dir, "mat")
    # Run imx520_sample.exe
    # ./imx520_sample.exe -i ./skvs/ -o ./skvs/mat/ -d
    process = subprocess.run([imx520_sample_exe_path, "-i", skvs_dir, "-o", output_mat_dir, "-d"], shell=True)

def mat_to_bfsig(skvs_dir):
    # Tag regions in face and generate bloodflow signature .mat file
    # myFaceMeshDetector = FaceMeshDetector(input_mats_dir="./skvs/mat/", output_bfsig_name="auto_bfsig")
    myFaceMeshDetector = FaceMeshDetector(input_mats_dir=os.path.join(skvs_dir, "mat"), output_bfsig_name="auto_bfsig")
    myFaceMeshDetector.run()

def bfsig_to_plot():
    # Run plotting matlab script
    # Create path to matlab script
    matlab_script_path = os.path.join(os.getcwd(), "auto_matlab/process_thermal_SINGLE.m")
    process = subprocess.run(["matlab", "-r", "run('" + matlab_script_path + "');"], shell=True)

if __name__ == '__main__':
    # print(tf.config.list_physical_devices('GPU'))






    # Get the path to the new .skv file
    # skvs_dir = record_skv()

    skvs_dir = os.path.join(os.getcwd(), 'skvs')

    check_for_skvs(skvs_dir)

    skv_to_mat(skvs_dir)

    mat_to_bfsig(skvs_dir)

    bfsig_to_plot()

    print('Done!')