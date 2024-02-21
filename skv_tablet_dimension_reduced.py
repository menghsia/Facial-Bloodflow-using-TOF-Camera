import pickle
import mediapipe as mp
from typing import ChainMap, Tuple, Union
from PIL import Image, ImageDraw
import numpy as np
import cv2, time
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import FastICA
from skimage.measure import block_reduce
import re
import scipy
from scipy.signal import savgol_filter
from scipy.ndimage import zoom
from skimage.transform import resize

import os
import serial
import csv

from processHR import ProcessHR
import run
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
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


def ROI_coord_extract(image, ROIwhich, img_plt = False):
    # Input image is a 2D nparray representing a grayscale image
    image_rows, image_cols = image.shape
    if ROIwhich == 'full_face':
        ROI_vertex = [54, 284, 365, 136]
    elif ROIwhich == 'left_face':
        ROI_vertex = [70, 135, 200, 8]
    elif ROIwhich == 'cheek_n_nose':
        ROI_vertex = [117, 346, 411, 187]
        #ROI_vertex = [116, 340, 433, 213]
    elif ROIwhich == 'left_cheek':
        ROI_vertex = [131, 165,214, 50]
    elif ROIwhich == 'right_cheek':
        ROI_vertex = [372, 433, 358]
    elif ROIwhich == 'chin':
        ROI_vertex = [175,148,152,377]
    else:
        print('No such ROI')
        quit()

    image_3chnl = np.stack((image,)*3, axis=-1)
    #print(image_3chnl.shape)

    # Facemesh detection

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    results = face_mesh.process(image_3chnl)
    face_landmarks = results.multi_face_landmarks[0]

    # Extract coordinates of all pixels within the ROI polygon
    landmark_px = []

    for i, vtx in enumerate(ROI_vertex):
        landmark_current = _normalized_to_pixel_coordinates(face_landmarks.landmark[vtx].x, face_landmarks.landmark[vtx].y, image_cols, image_rows)
        landmark_px.append(landmark_current)
        # print(landmark_px)

    if img_plt:
        #plt.figure()
        #plt.imshow(image, cmap='gray')
        for j,_ in enumerate(ROI_vertex):
            plt.scatter(landmark_px[j][0], landmark_px[j][1], s=1, c='red', marker='o')
            plt.plot((landmark_px[j-1][0], landmark_px[j][0]),(landmark_px[j-1][1],landmark_px[j][1]), c='g', linewidth=1)
        #plt.show()

    # n-by-2 2d array
    return landmark_px


def vtx2mask(vtx, image_cols, image_rows):
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


def pre_whiten(signal):
    sig_avg = np.average(signal)
    sig_std = np.std(signal)

    sig_norm = (signal - sig_avg) / sig_std

    return sig_norm


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def distcomp (roimean1, distmean1, power_range=np.arange(0.3,4.1,0.1), lco_range=np.arange(0.2,5.025,0.025),time_window=1, Fs=10):
    #distmean1= moving_average(distmean1, 9);
    timewindow=int(time_window*Fs);  #number of points in 2s
    L=len(roimean1)
    num_window=math.floor(L/timewindow)
    neutralized_pre=np.zeros(len(roimean1))
    neutralized=np.zeros(len(roimean1))

    for i in range(num_window):
        neutralized_pre[i*timewindow:(i+1)*timewindow-1] = roimean1[i*timewindow:(i+1)*timewindow-1]*(distmean1[i*timewindow:(i+1)*timewindow-1]**0.5)
        correlation=np.corrcoef(neutralized_pre[i*timewindow:(i+1)*timewindow-1],distmean1[i*timewindow:(i+1)*timewindow-1])
        correlation_pre=abs(correlation[1,0])
        for ii in power_range:
            for iii in lco_range:
                neutralized[i*timewindow:(i+1)*timewindow-1] = roimean1[i*timewindow:(i+1)*timewindow-1]*(iii*(distmean1[i*timewindow:(i+1)*timewindow-1]**ii))
                correlation=np.corrcoef( neutralized[i*timewindow:(i+1)*timewindow-1],distmean1[i*timewindow:(i+1)*timewindow-1])
                if abs(correlation[1,0])<correlation_pre:
                    neutralized_pre[i*timewindow:(i+1)*timewindow-1]=neutralized[i*timewindow:(i+1)*timewindow-1]
                    correlation_pre=abs(correlation[1,0]);
        neutralized_pre[i*timewindow:(i+1)*timewindow-1]=(neutralized_pre[i*timewindow:(i+1)*timewindow-1]-np.mean(neutralized_pre[i*timewindow:(i+1)*timewindow-1]))/np.std(neutralized_pre[i*timewindow:(i+1)*timewindow-1])
    if L%timewindow !=0:
        if L%timewindow >=2:
            neutralized_pre[int(L-L%timewindow):L] = roimean1[int(L-L%timewindow):L]*((distmean1[int(L-L%timewindow):L]**0.5))
            correlation=np.corrcoef(neutralized_pre[int(L-L%timewindow):L],distmean1[int(L-L%timewindow):L])
            correlation_pre=abs(correlation[1,0])
            for ii in power_range:
                for iii in lco_range:
                    neutralized[int(L-L%timewindow):L] = roimean1[int(L-L%timewindow):L]*(iii*(distmean1[int(L-L%timewindow):L]**ii))
                    correlation=np.corrcoef( neutralized[int(L-L%timewindow):L],distmean1[int(L-L%timewindow):L])
                    if abs(correlation[1,0])<correlation_pre:
                        neutralized_pre[int(L-L%timewindow):L]=neutralized[int(L-L%timewindow):L]
                        correlation_pre=abs(correlation[1,0]);
        neutralized_pre[int(L-L%timewindow):L]=(neutralized_pre[int(L-L%timewindow):L]-np.mean(neutralized_pre[int(L-L%timewindow):L]))/np.std(neutralized_pre[int(L-L%timewindow):L])
    else:
        neutralized_pre[L-1]=neutralized_pre[L-2]
    return neutralized_pre

def read_temp(portname):
    ser = serial.Serial(portname, 1000000)
    i=0
    temp = 0
    while True:
        cc=str(ser.readline())
        if 'Temperature Measured' in cc:
            i+=1
            # print(cc[2:][:-5])
            temp += float(cc[24:28])
            if i > 4:
                break
    avg_temp = temp/i
    print('Average temperature = '+str(avg_temp))

def Chest_ROI_extract(image, chin_location, plot=False):
    mp_pose = mp.solutions.pose
    pose= mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6,  min_tracking_confidence=0.6)

    image_3chnl = np.stack((image,)*3, axis=-1)
    image_3chnl = cv2.convertScaleAbs(image_3chnl)
    shoulder_landmark=[11,12]
    landmark_px_rr=np.zeros([2,4,2])
    image_height, image_width= image.shape
    results = pose.process(image_3chnl)
    body_points=results.pose_landmarks.landmark
    shoulder_point_l= _normalized_to_pixel_coordinates(body_points[11].x, body_points[11].y, image_width, image_height)
    shoulder_point_r= _normalized_to_pixel_coordinates(body_points[12].x, body_points[12].y, image_width, image_height)
    
    shoulder_x=(shoulder_point_l[0]+shoulder_point_r[0])/2
    shoulder_y=(shoulder_point_l[1]+shoulder_point_r[1])/2

    neck_width = 2*np.abs(chin_location[1][0]-chin_location[3][0])
    neck_height = 0.5*np.abs(shoulder_y-chin_location[2][1])

    chest_width= np.abs(shoulder_point_l[0]-shoulder_point_r[0])
    chest_height= 0.22*chest_width

    landmark_px_rr[0, :, 0] = [
        shoulder_x - 0.5*neck_width,
        shoulder_x + 0.5*neck_width,
        shoulder_x + 0.5*neck_width,
        shoulder_x - 0.5*neck_width
    ]

    landmark_px_rr[0, :, 1] = [
        shoulder_y - 1.1*neck_height,
        shoulder_y - 1.1*neck_height,
        shoulder_y - 0.1*neck_height,
        shoulder_y - 0.1*neck_height,
    ]

    #landmark_px_rr[0,:,0]=[chin_location[1][0]-0.8*neck_width,chin_location[3][0],
    #                       chin_location[3][0],chin_location[1][0]-0.8*neck_width]
    #landmark_px_rr[0,:,1]=[chin_location[1][1]+10,chin_location[3][1]+10,
    #                       chin_location[3][1]+neck_height,chin_location[1][1]+neck_height]

    landmark_px_rr[1, :, 0] = [
        shoulder_x - 0.3*chest_width,
        shoulder_x + 0.3*chest_width,
        shoulder_x + 0.3*chest_width,
        shoulder_x - 0.3*chest_width
    ]

    landmark_px_rr[1, :, 1] = [
        shoulder_y, 
        shoulder_y,
        shoulder_y + chest_height,
        shoulder_y + chest_height
    ]

    #landmark_px_rr[1,:,0]=[shoulder_point_l[0]-25,shoulder_point_r[0]+25,
    #                       shoulder_point_r[0]+25,shoulder_point_l[0]-25]
    #landmark_px_rr[1,:,1]=[shoulder_point_l[1],shoulder_point_r[1],
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
            #plt.plot((landmark_px[k][j-1][0], landmark_px[k][j][0]),(landmark_px[k][j-1][1],landmark_px[k][j][1]), c='g', linewidth=1)
    plt.show()
    return landmark_px_rr
def read_binary_file (filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    print(f'Bin file path: {filepath}')
    with open(filepath, 'rb') as binary_file:
        x_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
        y_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
        z_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
        confidence_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()

    return x_all, y_all, z_all, confidence_all
def bin_to_numpy(bin_file_path):
    """
    turn bin into numpy arrays of x,y,z, intensity
    """
    print(f'Processing {bin_file_path}')
    filepath = bin_file_path

    x_all, z_all, y_all, confidence_all = read_binary_file(filepath)
    
    frame_num = 600
    x_value = x_all.reshape([480, 640, frame_num])
    y_value = y_all.reshape([480, 640, frame_num])
    depth = z_all.reshape([480, 640, frame_num])
    intensity = confidence_all.reshape([480, 640, frame_num])

    # plt.figure()
    # plt.imshow(intensity[:,:,0], cmap='gray', aspect='auto')
    # plt.show()
    # I want to transpose so it's in the form (frame_num, 480, 640)
    x_value = np.transpose(x_value, (2, 0, 1))
    y_value = np.transpose(y_value, (2, 0, 1))
    depth = np.transpose(depth, (2, 0, 1))
    intensity = np.transpose(intensity, (2, 0, 1))
    
    # plt.figure()
    # plt.imshow(intensity[0, :, :], cmap='gray')
    # plt.show()
    print(f'shape of x_value: {x_value.shape}')
    print(f'shape of y_value: {y_value.shape}')
    print(f'shape of depth: {depth.shape}')
    print(f'shape of intensity: {intensity.shape}')

    return x_value, y_value, depth, intensity, frame_num

if __name__ == "__main__":

    skv_dir = os.path.join(os.getcwd(), 'skvs')
    print(skv_dir)
    # run.check_for_skvs(skv_dir)
    run.skv_to_bin(skv_dir)
    print('converted skv to bin')
    bin_dir = os.path.join(os.getcwd(), 'skvs/mat')
    bin_files = [f for f in os.listdir(bin_dir) if f.endswith('.bin')]
    # print(bin_files)
    for filename in bin_files:
        start = time.time()

        bin_dir = os.path.join(os.getcwd(), f'skvs/mat/{filename}')
        x_value, y_value, depth, intensity, frame_num = bin_to_numpy(bin_dir)

        x_value = x_value[20:320, :, :]
        y_value = y_value[20:320, :, :]
        depth = depth[20:320, :, :]
        intensity = intensity[20:320, :, :]
        frame_num = 300
        # resize using opencv 
        resized_x = np.zeros((300, 171, 224))
        resized_y = np.zeros((300, 171, 224))
        resized_depth = np.zeros((300, 171, 224))
        resized_intensity = np.zeros((300, 171, 224))
        for i in range(300):
            resized_x[i] = resize(x_value[i], (171, 224))
            resized_y[i] = resize(y_value[i], (171, 224))
            resized_depth[i] = resize(depth[i], (171, 224))
            resized_intensity[i] = resize(intensity[i], (171, 224))

        
        x_value = resized_x
        y_value = resized_y
        depth = resized_depth
        intensity = resized_intensity



        # pil_image = Image.fromarray(intensity[0, :, :])
        # plt.figure()
        # plt.imshow(pil_image, cmap='gray')
        # plt.show()


        # resized_image = pil_image.resize((224, 171), Image.LANCZOS)

        # resized_x = np.zeros((300, 171, 224))
        # resized_y = np.zeros((300, 171, 224))
        # resized_depth = np.zeros((300, 171, 224))
        # resized_intensity = np.zeros((300, 171, 224))

        # for i in range(300):
        #     pil_image = Image.fromarray(x_value[i].astype('uint8'))
        #     resized_x[i] = np.array(pil_image.resize((224, 171), Image.LANCZOS))
        #     pil_image = Image.fromarray(y_value[i].astype('uint8'))
        #     resized_y[i] = np.array(pil_image.resize((224, 171), Image.LANCZOS))
        #     pil_image = Image.fromarray(depth[i].astype('uint8'))
        #     resized_depth[i] = np.array(pil_image.resize((224, 171), Image.LANCZOS))
        #     pil_image = Image.fromarray(intensity[i].astype('uint8'))
        #     resized_intensity[i] = np.array(pil_image.resize((224, 171), Image.LANCZOS))

        # x_value = resized_x
        # y_value = resized_y
        # depth = resized_depth
        # intensity = resized_intensity
        
        # zoom_factor_1 = 171 / 480
        # zoom_factor_2 = 224 / 640
        # x_value = zoom(x_value, (1, zoom_factor_1, zoom_factor_2))
        # y_value = zoom(y_value, (1, zoom_factor_1, zoom_factor_2))
        # depth = zoom(depth, (1, zoom_factor_1, zoom_factor_2))
        # intensity = zoom(intensity, (1, zoom_factor_1, zoom_factor_2))
        # plt.figure()
        # plt.imshow(intensity[0, :, :], cmap='gray')
        # plt.show()
        print(f'shape of x_value: {x_value.shape}')
        print(f'shape of y_value: {y_value.shape}')
        print(f'shape of depth: {depth.shape}')
        print(f'shape of intensity: {intensity.shape}')
        # cut 300 frames inside the 600 frames from 20th frame

        I_signal = np.zeros(frame_num)
        D_signal = np.zeros(frame_num)
        image_rows = 171
        image_cols = 224

        # Extract initial ROI landmark locations
        start_frame=0
        frameSig = intensity[start_frame,:,:]
        frameTrk = intensity[start_frame,:,:]
        frameTrk[np.where(frameTrk>255)] = 255
        frameTrk = np.uint8(frameTrk)

        # print(f'confidence all 1st frame: {frameSig[10]}')
        plt.figure()
        plt.imshow(frameTrk, cmap='gray')
        plt.show()

        ROIcoords_full  = ROI_coord_extract(frameTrk,'full_face',img_plt = True)
        ROIcoords_sig = ROI_coord_extract(frameTrk,'cheek_n_nose',img_plt=False)

        xframe = x_value[start_frame,:,:]
        yframe = y_value[start_frame,:,:]
        zframe = depth[start_frame,:,:]
        '''
        ##### CHEST ROI #####
        D_signal_RR = np.zeros((2,frame_num))

        chin_edge = ROI_coord_extract(frameTrk, 'chin')
        ROIcoord_RR= Chest_ROI_extract(frameTrk, chin_edge, plot=True)
        alpha = [tuple(i) for i in ROIcoord_RR[0,:,:].tolist()]
        beta = [tuple(i) for i in ROIcoord_RR[1,:,:].tolist()]
        ini_ROImask_RR_neck = vtx2mask(alpha,image_cols,image_rows)
        ini_ROImask_RR_chest = vtx2mask(beta,image_cols,image_rows)
        non_zero_pnts_neck=np.where(ini_ROImask_RR_neck>0)
        non_zero_pnts_chest=np.where(ini_ROImask_RR_chest>0)
        D_signal_RR[0,0] = np.average(zframe[non_zero_pnts_neck[0],non_zero_pnts_neck[1]])
        D_signal_RR[1,0] = np.average(zframe[non_zero_pnts_chest[0],non_zero_pnts_chest[1]])

        ##### CHEST ROI #####
        '''
        # Corner feature detection for tracking the ROI
        # Harris corner detector parameters
        feature_params = dict( maxCorners = 100,
                                qualityLevel = 0.01,
                                minDistance = 10,
                                blockSize = 6 )

        mask = vtx2mask(ROIcoords_full, image_cols, image_rows)
        p0 = cv2.goodFeaturesToTrack(frameTrk, mask= mask, **feature_params)

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (21,21),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.005))
        frame_new=np.zeros((171,224)).astype('uint8')

        ini_ROImask = vtx2mask(ROIcoords_sig,image_cols,image_rows)
        # print(f'ini_ROImask: \n {ini_ROImask}')
        ini_ROIcoords_sig = np.asarray(ROIcoords_sig).T
        ini_ROIcoords_sig = np.vstack((ini_ROIcoords_sig,np.ones(ini_ROIcoords_sig.shape[1])))
        old_ROIcoords = ini_ROIcoords_sig

        xframe[np.where(ini_ROImask>0)]**2 + yframe[np.where(ini_ROImask>0)]**2 + zframe[np.where(ini_ROImask>0)]

        # print(f'xframe: {xframe}')
        # print(f'yframe: {yframe}')
        # print(f'zframe: {zframe}')
        # print(f'ini_ROImask: {ini_ROImask}')
        # Initial value
        I_signal[0] = np.average(frameSig[np.where(ini_ROImask>0)])
        # D_signal[0] = np.sqrt(np.average(xframe[np.where(ini_ROImask>0)]**2 +
        #     yframe[np.where(ini_ROImask>0)]**2 + zframe[np.where(ini_ROImask>0)]**2))
        D_signal[0] = np.sqrt(
            np.average(xframe[np.where(ini_ROImask>0)])**2 +
            np.average(yframe[np.where(ini_ROImask>0)])**2 +
            np.average(zframe[np.where(ini_ROImask>0)])**2
        )
        print(f'D_signal[0]: {D_signal[0]}')
        # D_signal[0] = np.average(np.sqrt((xframe[np.where(ini_ROImask>0)])**2 +
        #     (yframe[np.where(ini_ROImask>0)])**2 + (zframe[np.where(ini_ROImask>0)])**2))

        # Create 3 length buffer to calculate RoI means
        # roi_history = np.zeros(tuple([3] + list(old_ROIcoords.shape)))
        # roi_history[0, :, :] = old_ROIcoords
        # roi_history[1, :, :] = old_ROIcoords
        # roi_history[2, :, :] = old_ROIcoords

        for i in range(frame_num-1):
            frameSig = intensity[i+1,:,:]
            frame_new = intensity[i+1,:,:]/4
            frame_new = np.uint8(frame_new)


            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(frameTrk, frame_new, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # Now update the previous frame and previous points
            print(f'good_new: {good_new}')
            print(f'good_old: {good_old}')


            img_track = frame_new.copy()
            frameTrk = frame_new.copy()
            xframe = x_value[i+1,:,:]
            yframe = y_value[i+1,:,:]
            zframe = depth[i+1,:,:]

            p0 = good_new.reshape(-1,1,2)
            xform_matrix, inliers = cv2.estimateAffinePartial2D(good_old,good_new)

            ##################
            #D_signal_RR[0,i+1] = np.average(zframe[non_zero_pnts_neck[0],non_zero_pnts_neck[1]])
            #D_signal_RR[1,i+1] = np.average(zframe[non_zero_pnts_chest[0],non_zero_pnts_chest[1]])

            ##################

            # No averaging of ROI
            new_ROIcoords = np.dot(xform_matrix,old_ROIcoords)
            old_ROIcoords = np.vstack((new_ROIcoords,np.ones(new_ROIcoords.shape[1])))

            # New method with 3 frame averaging of ROI coordinates
            # old_ROIcoords = roi_history[2, :, :]
            # roi_history[0, :, :] = roi_history[1, :, :]
            # roi_history[1, :, :] = roi_history[2, :, :]
            # knrc = np.dot(xform_matrix,old_ROIcoords)
            # roi_history[2, :, :] = np.vstack((knrc, np.ones(knrc.shape[1])))
            # new_ROIcoords = np.mean(roi_history, axis=0)
            # new_ROIcoords = new_ROIcoords[:-1, :]
            # roi_history[2, :, :] = np.vstack((new_ROIcoords, np.ones(new_ROIcoords.shape[1])))

            # record ROI signal
            ROImask = vtx2mask(list(np.reshape(new_ROIcoords.T,-1)),image_cols,image_rows)
            I_signal[i+1] = np.average(frameSig[np.where(ROImask>0)])
            D_signal[i+1] = np.average(np.sqrt(xframe[np.where(ROImask>0)]**2 +
                yframe[np.where(ROImask>0)]**2 + zframe[np.where(ROImask>0)]**2))
            # if i > 50 and i < 60:
            #     print(f'frame {i} I_signal: {I_signal[i+1]}')
            #     print(f'frame {i} D_signal: {D_signal[i+1]}') 
            # show tracking performance

            ROIpts = np.transpose(new_ROIcoords)
            ROIpts = np.int32(ROIpts)
            img_track = np.stack((img_track,)*3, axis=-1)
            img_track = cv2.polylines(img_track,[ROIpts],True, color = (0,0,255), thickness = 1)
            for kk, points in enumerate(good_new[:,1]):
                img_track = cv2.circle(img_track, (int(good_new[kk,0]),int(good_new[kk,1])), radius=2, color=(0, 255, 255), thickness=-1)


            # cv2.imshow('frame', img_track)
            # cv2.waitKey(30)


        cv2.destroyAllWindows()

        # export D_signal and I_signal to csv files
        # if file doesn't exist create it, otherwise append
        # path = '/datas/csv/'
        # os.chmod(path, 0o777)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        
        # give write permission to the file
        # print(f'shape of I_signal: {I_signal.shape}, length: {len(I_signal)}')
        # print(f'shape of D_signal: {D_signal.shape}, length: {len(D_signal)}')
        with open('PLY/tablet_csv/intensity.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(I_signal.reshape(-1,1))
        with open('PLY/tablet_csv/depth.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(D_signal.reshape(-1,1))

        D_signal_smooth=scipy.signal.savgol_filter(D_signal,9,2,mode='nearest')
        I_signal_smooth=scipy.signal.savgol_filter(I_signal,5,2,mode='nearest')
        
        print(f'D signal shape after smoothing: {D_signal_smooth.shape}')
        print(f'I signal shape after smoothing: {I_signal_smooth.shape}')
        
        I_compensated = distcomp(I_signal_smooth/200, D_signal_smooth,time_window=1, Fs = 10)
        print(f'I_compensated shape: {I_compensated.shape}')
        fps = 10
        T = 1.0 / fps
        yf_hr = abs(fft(I_compensated)) # this is spectrum
        yf_hr=2.0 / frame_num * yf_hr[:frame_num // 2] # frame_num=L
        xf_hr = np.linspace(0.0, 1.0 / (2.0 * T), frame_num // 2) # xf_hr = f
        xf_hr = xf_hr * 60
        yf_hr[np.where(xf_hr<=40 )]=0 
        yf_hr[np.where(xf_hr>=200)]=0

        #isiah uncompensated hr BEGIN
        yf_hrun = abs(fft(I_signal_smooth))
        yf_hrun = 2.0 / frame_num * yf_hrun[:frame_num // 2]
        xf_hrun = np.linspace(0.0, 1.0 / (2.0 * T), frame_num // 2)
        xf_hrun = xf_hrun * 60
        yf_hrun[np.where(xf_hrun <= 40)] = 0
        yf_hrun[np.where(xf_hrun >= 150)] = 0

        peaks, properties = scipy.signal.find_peaks(yf_hrun)
        max_index = np.argmax(yf_hrun[peaks])
        HR_UNCOMP = xf_hrun[peaks[max_index]]
        #isiah uncompensated hr END

        peaks, properties = scipy.signal.find_peaks(yf_hr)
        max_index=np.argmax(yf_hr[peaks])
        HR_comp = xf_hr[peaks[max_index]]

            
        # error_rate=abs(HR_comp-HR_ref)/abs(HR_ref)*100
        # success_label=error_rate<11
        # success_index=np.where(error_rate<10)
        # success_rate=np.shape(success_index)[1]/len(HR_comp)   
        # plt.figure()
        # plt.plot(D_signal)
        # plt.show()
        # f1 = plt.figure()
        # ax1 = f1.add_subplot(221)
        # ax1.plot(I_signal_smooth)
        # plt.figure() 
        # plt.plot(D_signal_smooth)
        # plt.show()

        # #please work - isiah
        # np.savetxt('test.out',D_signal, delimiter = ',')



        # ax2 = f1.add_subplot(222)
        # ax2.plot(D_signal_smooth)
        # ax3 = f1.add_subplot(223)
        # ax3.plot(I_compensated)
        # ax4 = f1.add_subplot(224)
        # ax4.plot(xf, yf)
        # plt.xlim(0, 200)
        # plt.ylim(0,0.5)

        print("COMPENSATED Heart Rate Measured", HR_comp)
        print("UNCOMPENSATED Heart Rate Measured", HR_UNCOMP)

        with open('PLY/tablet_csv/tablet_code_results.csv', 'a', newline='') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(['HR_comp', HR_comp])
            write.writerow(['HR_UNCOMP', HR_UNCOMP])
        # print(D_signal_RR.shape)
        # plt.figure()
        # plt.plot(D_signal_RR[0,:])
        # plt.plot(D_signal_RR[1,:])
        # plt.show()



        # plt.figure()
        # plt.plot(I_signal)
        # plt.show()


        # transformer = FastICA(n_components=2, max_iter=500, whiten=True, tol=5e-3)
        # X_transformed = transformer.fit_transform(D_signal_RR.T)

        #plt.figure()
        #plt.plot(X_transformed[:,0])
        #plt.plot(X_transformed[:,1])
        #plt.show()

        '''
        yf_rr1 = abs(fft(X_transformed[:,0]))
        yf_rr1=2.0 / frame_num * yf_rr1[:frame_num // 2]
        xf_rr1 = np.linspace(0.0, 1.0 / (2.0 * T), frame_num // 2)
        xf_rr1 = xf_rr1 * 60
        yf_rr1[np.where(xf_rr1<=5 )]=0
        yf_rr1[np.where(xf_rr1>=30)]=0

        peaks, properties = scipy.signal.find_peaks(yf_rr1)
        max_index_1=np.argmax(yf_rr1[peaks])

        yf_rr2 = abs(fft(X_transformed[:,1]))
        yf_rr2=2.0 / frame_num * yf_rr1[:frame_num // 2]
        xf_rr2 = np.linspace(0.0, 1.0 / (2.0 * T), frame_num // 2)
        xf_rr2 = xf_rr2 * 60
        yf_rr2[np.where(xf_rr2<=5 )]=0
        yf_rr2[np.where(xf_rr2>=30)]=0

        peaks, properties = scipy.signal.find_peaks(yf_rr2)
        max_index_2=np.argmax(yf_rr2[peaks])

        if yf_rr1[peaks[max_index_1]]>yf_rr2[peaks[max_index_2]]:
            RR_ica=xf_rr1[peaks[max_index_1]]
        else:
            RR_ica=xf_rr2[peaks[max_index_2]]

        print("Respiratory Rate Measured", RR_ica)
        '''


        # Read temperature
        # read_temp('/dev/ttyUSB0')
        # plt.figure()
        # plt.plot(X_transformed)
        # plt.show()

        # plt.figure()
        # plt.plot(xf_rr, yf_rr)
        # plt.show()

        # run phase 3 main code
        print(f'processing main code for file {i}')
        processHR = ProcessHR(input_file=f"PLY/{i}.ply")
        Depth = D_signal
        I_raw = I_signal        
        print('I_raw shape', I_raw.shape)
        print('Depth shape', Depth.shape)
        
        # for i in range(Depth.shape[1]-1, -1, -1):
        #     if Depth[0, i] == 0:
        #         Depth = np.delete(Depth, i, axis=1)
        #         I_raw = np.delete(I_raw, i, axis=1)
        #     else:
        #         break
        # Depth = np.delete(Depth, 6, axis=0)
        # I_raw = np.delete(I_raw, 6, axis=0)
        # Depth = scipy.signal.savgol_filter(Depth, 9, 2, mode='nearest')    
        # I_raw = scipy.signal.savgol_filter(I_raw, 5, 2, mode='nearest')
        # print('I_raw shape', I_raw.shape)
        # print('Depth shape', Depth.shape)
        # # I want 7 times of the I_raw
        # I_raw = np.repeat(I_raw, 7, axis=0)
        # Depth = np.repeat(Depth, 7, axis=0)
        # I_comp = processHR.depthComp(I_raw, Depth, 2, 10) # depthComp is good
        # print('I_comp shape', I_comp.shape)
        # HRsig = I_comp[2,:]
        HRsig = distcomp(I_raw/200, Depth, time_window=1, Fs=30)
        HRsigRaw = I_raw
        HR_comp = processHR.getHR(HRsig, 300, Window=False)
        HR_ND = processHR.getHR(HRsigRaw, 300, Window=False)
        print(f'Main HR: {HR_comp}')
        print(f'Main HR_ND: {HR_ND}')