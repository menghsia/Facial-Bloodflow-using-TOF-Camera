import mediapipe as mp
from typing import ChainMap, Tuple, Union
from PIL import Image, ImageDraw
import numpy as np
import cv2
import time
from scipy.fftpack import fft
import scipy.io
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import FastICA
import subprocess
import scipy
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
# import serial
import os


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


def ROI_coord_extract(image, ROIwhich, img_plt=False):
    # Input image is a 2D nparray representing a confidence_all image
    image_rows, image_cols = image.shape
    if ROIwhich == 'full_face':
        ROI_vertex = [54, 284, 365, 136]
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
    else:
        print('No such ROI')
        quit()

    image_3chnl = np.stack((image,)*3, axis=-1)

    # Facemesh detection

    mp_face_mesh = mp.solutions.face_mesh # type: ignore
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    results = face_mesh.process(image_3chnl)
    face_landmarks = results.multi_face_landmarks[0]

    # Extract coordinates of all pixels within the ROI polygon
    landmark_px = []

    for i, vtx in enumerate(ROI_vertex):
        landmark_current = _normalized_to_pixel_coordinates(
            face_landmarks.landmark[vtx].x, face_landmarks.landmark[vtx].y, image_cols, image_rows)
        landmark_px.append(landmark_current)
        # print(landmark_px)

    if img_plt:
        plt.figure()
        plt.imshow(image, cmap='gray')
        for j, _ in enumerate(ROI_vertex):
            plt.scatter(landmark_px[j][0], landmark_px[j]
                        [1], s=1, c='red', marker='o')
            plt.plot((landmark_px[j-1][0], landmark_px[j][0]),
                     (landmark_px[j-1][1], landmark_px[j][1]), c='g', linewidth=1)
        plt.show()

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


def compensate_intensity_using_depth(intensities, depths, b_values=np.arange(0.2, 5.1, 0.1), a_values=[1.0], sub_clip_length=2, frames_per_second=30):
    """
    Args:
        intensities: 1D array of intensity signals
        depths: 1D array of depth signals
        b_values: array of b values to try
        a_values: array of a values to try
        sub_clip_length: length of each subclip that the full clip gets split into (seconds)
        frames_per_second: frames per second of the video
    Returns:
        intensity_compensated: 1D array of depth-compensated intensity signals
    """
    # distmean1= moving_average(distmean1, 9);
    num_frames_window = int(sub_clip_length*frames_per_second)  # number of points in 2s
    num_frames = len(intensities)
    window_idx = math.floor(num_frames/num_frames_window)
    intensity_compensated = np.zeros(len(intensities))
    intensity_compensated_temp = np.zeros(len(intensities))

    for i in range(window_idx):
        intensity_compensated[i*num_frames_window:(i+1)*num_frames_window] = intensities[i*num_frames_window:(i+1)*num_frames_window]*(depths[i*num_frames_window:(i+1)*num_frames_window]**0.5)
        correlation_temp = np.corrcoef(intensity_compensated[i*num_frames_window:(i+1)*num_frames_window], depths[i*num_frames_window:(i+1)*num_frames_window])
        correlation_best = abs(correlation_temp[1, 0])
        for ii in b_values:
            for iii in a_values:
                intensity_compensated_temp[i*num_frames_window:(i+1)*num_frames_window] = intensities[i*num_frames_window:(i+1)*num_frames_window]*(iii*(depths[i*num_frames_window:(i+1)*num_frames_window]**ii))
                correlation_temp = np.corrcoef(intensity_compensated_temp[i*num_frames_window:(i+1)*num_frames_window], depths[i*num_frames_window:(i+1)*num_frames_window])
                if abs(correlation_temp[1, 0]) < correlation_best:
                    intensity_compensated[i*num_frames_window:(i+1)*num_frames_window] = intensity_compensated_temp[i*num_frames_window:(i+1)*num_frames_window]
                    correlation_best = abs(correlation_temp[1, 0])
        intensity_compensated[i*num_frames_window:(i+1)*num_frames_window] = (intensity_compensated[i*num_frames_window:(i+1)*num_frames_window]-np.mean(
            intensity_compensated[i*num_frames_window:(i+1)*num_frames_window]))/np.std(intensity_compensated[i*num_frames_window:(i+1)*num_frames_window])
    if num_frames % num_frames_window != 0:
        # This is an edge case when the clip does not divide evenly into subclips
        # In this case, do max number of full time windows and then with the final subclip, do the same thing as above except only use the number of frames in the subclip
        if num_frames % num_frames_window >= 2:
            intensity_compensated[int(num_frames-num_frames % num_frames_window):num_frames] = intensities[int(num_frames-num_frames % num_frames_window):num_frames]*((depths[int(num_frames-num_frames % num_frames_window):num_frames]**0.5))
            correlation_temp = np.corrcoef(intensity_compensated[int(num_frames-num_frames % num_frames_window):num_frames], depths[int(num_frames-num_frames % num_frames_window):num_frames])
            correlation_best = abs(correlation_temp[1, 0])
            for ii in b_values:
                for iii in a_values:
                    intensity_compensated_temp[int(num_frames-num_frames % num_frames_window):num_frames] = intensities[int(num_frames-num_frames % num_frames_window):num_frames]*(iii*(depths[int(num_frames-num_frames % num_frames_window):num_frames]**ii))
                    correlation_temp = np.corrcoef(intensity_compensated_temp[int(num_frames-num_frames % num_frames_window):num_frames], depths[int(num_frames-num_frames % num_frames_window):num_frames])
                    if abs(correlation_temp[1, 0]) < correlation_best:
                        intensity_compensated[int(num_frames-num_frames % num_frames_window):num_frames] = intensity_compensated_temp[int(num_frames-num_frames % num_frames_window):num_frames]
                        correlation_best = abs(correlation_temp[1, 0])
        intensity_compensated[int(num_frames-num_frames % num_frames_window):num_frames] = (intensity_compensated[int(num_frames-num_frames % num_frames_window):num_frames]-np.mean(intensity_compensated[int(num_frames-num_frames % num_frames_window):num_frames]))/np.std(intensity_compensated[int(num_frames-num_frames % num_frames_window):num_frames])
    else:
        intensity_compensated[num_frames-1] = intensity_compensated[num_frames-2]
    return intensity_compensated

def Chest_ROI_extract(image, chin_location, plot=False):
    mp_pose = mp.solutions.pose # type: ignore
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.6,  min_tracking_confidence=0.6)

    image_3chnl = np.stack((image,)*3, axis=-1)
    image_3chnl = cv2.convertScaleAbs(image_3chnl)
    shoulder_landmark = [11, 12]
    landmark_px_rr = np.zeros([2, 4, 2])
    image_height, image_width = image.shape
    results = pose.process(image_3chnl)
    body_points = results.pose_landmarks.landmark
    shoulder_point_l = _normalized_to_pixel_coordinates(
        body_points[11].x, body_points[11].y, image_width, image_height)
    shoulder_point_r = _normalized_to_pixel_coordinates(
        body_points[12].x, body_points[12].y, image_width, image_height)

    shoulder_x = (shoulder_point_l[0]+shoulder_point_r[0])/2
    shoulder_y = (shoulder_point_l[1]+shoulder_point_r[1])/2

    neck_width = 2*np.abs(chin_location[1][0]-chin_location[3][0])
    neck_height = 0.5*np.abs(shoulder_y-chin_location[2][1])

    chest_width = np.abs(shoulder_point_l[0]-shoulder_point_r[0])
    chest_height = 0.22*chest_width

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

    # landmark_px_rr[0,:,0]=[chin_location[1][0]-0.8*neck_width,chin_location[3][0],
    #                       chin_location[3][0],chin_location[1][0]-0.8*neck_width]
    # landmark_px_rr[0,:,1]=[chin_location[1][1]+10,chin_location[3][1]+10,
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
        plt.scatter(chin_location[1][0], chin_location[1]
                    [1], s=12, c='green', marker='x')
        plt.scatter(chin_location[3][0], chin_location[3]
                    [1], s=12, c='green', marker='x')

        plt.scatter(
            shoulder_point_l[0], shoulder_point_l[1], s=6, c='green', marker='o')
        plt.scatter(
            shoulder_point_r[0], shoulder_point_r[1], s=6, c='green', marker='o')
        for j in range(4):
            plt.scatter(
                landmark_px_rr[0][j][0], landmark_px_rr[0][j][1], s=8, c='red', marker='x')
            plt.scatter(
                landmark_px_rr[1][j][0], landmark_px_rr[1][j][1], s=1, c='black', marker='o')
            # plt.plot((landmark_px[k][j-1][0], landmark_px[k][j][0]),(landmark_px[k][j-1][1],landmark_px[k][j][1]), c='g', linewidth=1)
    plt.show()
    return landmark_px_rr

####### FUNCTIONS FROM MAIN CODE FOR TESTING #######

def depthComp(I_raw, Depth, timeWindow=2, Fs=30):
    """
    depthComp finds compensated intensity using the equation in the research paper

    Args:
        I_raw (2D Array of ints): Raw intensities at each ROI
        Depth (2D Array of ints): Raw depths at each ROI
        timeWindow (int): Every time window to iterate for finding best b value, in seconds
        Fs (int): frames per second

    Returns:
        I_comp (2D Array of ints): Compensated intesities (7x1800 for 60s)
        
    """

    I_comp = np.ones_like(I_raw)

    # best: scalar variable to find best b value
    best = 1
    # best_rem: scalar variable to find best b value for the remainder of the clip less than 20s
    best_rem = 1

    # Iterate through the different ROIs
    for ROI in range(I_raw.shape[0]):
        # I_comp_ROI: 2D array of ints with the compensated intensities for the ROI
        I_comp_ROI = np.ones(I_raw.shape[1])
        # i: scalar variable to iterate through each clip(time window)
        i = 1

        # Iterate through every clip...so every 60 frames
        while (i * (timeWindow * Fs)) < len(I_raw[ROI, :]):
            # cor: the lowest correlation coefficient that we compare to/reset (we start at 1 because that is highest possible value)
            cor = 2

            # For each clip, iterate through different b values with a = 1
            for bi in np.arange(0.2, 5.1, 0.1):
                bI_comp = I_raw[ROI, ((i - 1) * (timeWindow * Fs)) : ((i * (timeWindow * Fs)))] / ((Depth[ROI, ((i - 1) * (timeWindow * Fs)) : ((i * (timeWindow * Fs)))]) ** (-bi))
                # Find correlation between bI_comp and Depth
                corr_v = np.corrcoef(bI_comp, Depth[ROI, ((i - 1) * (timeWindow * Fs)) : ((i * (timeWindow * Fs)))])
                # Take absolute value of correlation coefficients
                corr_ = abs(corr_v[1, 0])

                # If the new correlation coeff is less than the old one, reset cor value and best I_comp
                if corr_ < cor:
                    cor = corr_
                    best = bI_comp

            # Normalize data using z-scores
            I_comp_ROI[((i - 1) * (timeWindow * Fs)) : ((i * (timeWindow * Fs)))] = (best - np.mean(best))/np.std(best)
            i += 1

        # For the remainder of the clip if it is 
        cor = 2
        for bii in np.arange(0.2, 5.1, 0.1):
            bI_comp = I_raw[ROI, (((i - 1) * (timeWindow * Fs))) :] / (Depth[ROI, (((i - 1) * (timeWindow * Fs))) :]) ** (-bii)
            # Find correlation between bI_comp and Depth
            corr_v = np.corrcoef(bI_comp, Depth[ROI, (((i - 1) * (timeWindow * Fs)) ) :])
            # Take absolute value of correlation coefficients
            corr_ = abs(corr_v[1, 0])

            # If the new correlation coeff is less than the old one, reset cor value and I_comp
            if corr_ < cor:
                cor = corr_
                best_rem = bI_comp

        # Normalize data
        I_comp_ROI[(((i - 1) * (timeWindow * Fs))) :] = (best_rem - np.mean(best_rem))/np.std(best_rem)
        # Append to final output matrix
        I_comp[ROI, :] = I_comp_ROI

    return I_comp

def getHR(HRsig, L, trial=None): 
    ###  NEEDS FIXING ###

    # Prepare Parameters
    Fs = 30
    T = 1/Fs

    # Get HR
    spectrum = abs(fft(HRsig))
    spectrum = 2.0 / L * spectrum[:L // 2]
    f = np.linspace(0.0, 1.0 / (2.0 * T), L // 2) * 60
    # f_Filtered_range = np.logical_or(f < 40, f > 150)
    spectrum[np.where(f < 40)] = 0
    spectrum[np.where(f > 150)] = 0

    # HR peak locate
    pks, properties = find_peaks(spectrum.squeeze())
    maxindex = np.argmax(spectrum[pks])
    HR = f[pks[maxindex]]

    plt.figure()
    plt.plot(f, spectrum)
    plt.xlim((40, 150))
    
    return HR

####### END MAIN CODE FUNCTIONS #######


if __name__ == '__main__':
    start = time.time()

    mat_data = {
        'x_all': [],
        'y_all': [],
        'z_all': [],
        'confidence_all': []
    }

    # loaded_matfile = scipy.io.loadmat("skvs/mat/sk_automotive_20230525_175029.skv.mat")

    skvs_dir = os.path.join(os.getcwd(), 'skvs')
    bins_and_mats_dir = os.path.join(skvs_dir, 'mat')

    loaded_matfile = scipy.io.loadmat(os.path.join(bins_and_mats_dir, 'sk_automotive_20230706_140354.skv.mat'))

    mat_data['x_all'] = loaded_matfile['x_all']
    mat_data['y_all'] = loaded_matfile['y_all']
    mat_data['z_all'] = loaded_matfile['z_all']
    mat_data['confidence_all'] = loaded_matfile['confidence_all']

    num_frames = int(np.size(mat_data['confidence_all'])/(640*480))

    x_all = np.reshape(np.transpose(
        mat_data['x_all']), (num_frames, 480, 640)).astype('float')
    y_all = np.reshape(np.transpose(
        mat_data['y_all']), (num_frames, 480, 640)).astype('float')
    z_all = np.reshape(np.transpose(
        mat_data['z_all']), (num_frames, 480, 640)).astype('float')
    confidence_all = np.reshape(np.transpose(
        mat_data['confidence_all']), (num_frames, 480, 640)).astype('int')

    print("Data loading complete\n")

    intensity_signals = np.zeros(num_frames)
    depth_signals = np.zeros(num_frames)
    image_rows = 480
    image_cols = 640

    # Get grayscale image for first frame
    frame_confidence = confidence_all[0, :, :]
    frame_grayscale = confidence_all[0, :, :]/4
    frame_grayscale[np.where(frame_grayscale > 255)] = 255
    frame_grayscale = np.uint8(frame_grayscale)

    # Save frame_grayscale to a .mat file
    # scipy.io.savemat('frame_grayscale.mat', {'frame_grayscale': frame_grayscale})

    pixels_ROI_full_face = ROI_coord_extract(
        frame_grayscale, 'full_face', img_plt=False)
    pixels_ROI_cheek_n_nose = ROI_coord_extract(
        frame_grayscale, 'cheek_n_nose', img_plt=False)

    frame_x = x_all[0, :, :]
    frame_y = y_all[0, :, :]
    frame_z = z_all[0, :, :]

    ##### CHEST ROI #####
    D_signal_RR = np.zeros((2, num_frames))

    pixels_ROI_chin = ROI_coord_extract(frame_grayscale, 'chin')
    ROIcoord_RR = Chest_ROI_extract(
        frame_grayscale, pixels_ROI_chin, plot=False)
    alpha = [tuple(i) for i in ROIcoord_RR[0, :, :].tolist()]
    beta = [tuple(i) for i in ROIcoord_RR[1, :, :].tolist()]
    ini_ROImask_RR_neck = vtx2mask(alpha, image_cols, image_rows)
    ini_ROImask_RR_chest = vtx2mask(beta, image_cols, image_rows)
    non_zero_pnts_neck = np.where(ini_ROImask_RR_neck > 0)
    non_zero_pnts_chest = np.where(ini_ROImask_RR_chest > 0)
    D_signal_RR[0, 0] = np.average(
        frame_z[non_zero_pnts_neck[0], non_zero_pnts_neck[1]])
    D_signal_RR[1, 0] = np.average(
        frame_z[non_zero_pnts_chest[0], non_zero_pnts_chest[1]])

    ##### CHEST ROI #####

    # Corner feature detection for tracking the ROI
    # Harris corner detector parameters
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.01,
                          minDistance=10,
                          blockSize=6)

    mask_full_face = vtx2mask(pixels_ROI_full_face, image_cols, image_rows)
    p0 = cv2.goodFeaturesToTrack(
        frame_grayscale, mask=mask_full_face, **feature_params)
    # frame_cv2 = np.expand_dims(frame_grayscale, axis=2)
    # frame_cv2 = cv2.cvtColor(frame_cv2, cv2.COLOR_GRAY2BGR)

    # for corner in p0:
    #     x, y = corner.ravel()
    #     cv2.circle(frame_cv2, (int(x), int(y)), 3, 255, -1)

    # cv2.imshow('frame', frame_cv2)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(21, 21),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.005))
    frame_grayscale_new = np.zeros((480, 640)).astype('uint8')

    mask_cheek_n_nose = vtx2mask(
        pixels_ROI_cheek_n_nose, image_cols, image_rows)
    mask_cheek_n_nose_transposed = np.asarray(pixels_ROI_cheek_n_nose).T
    mask_cheek_n_nose_transposed = np.vstack(
        (mask_cheek_n_nose_transposed, np.ones(mask_cheek_n_nose_transposed.shape[1])))
    old_mask_cheek_n_nose_transposed = mask_cheek_n_nose_transposed

    # Initial value
    intensity_signals[0] = np.average(frame_confidence[np.where(mask_cheek_n_nose > 0)])
    depth_signals[0] = np.average(np.sqrt((frame_x[np.where(mask_cheek_n_nose > 0)])**2 + (
        frame_y[np.where(mask_cheek_n_nose > 0)])**2 + (frame_z[np.where(mask_cheek_n_nose > 0)])**2))

    # Create 3 length buffer to calculate RoI means
    # roi_history = np.zeros(tuple([3] + list(old_ROIcoords.shape)))
    # roi_history[0, :, :] = old_ROIcoords
    # roi_history[1, :, :] = old_ROIcoords
    # roi_history[2, :, :] = old_ROIcoords

    for i in range(num_frames-1):
        frame_confidence = confidence_all[i+1, :, :]
        frame_grayscale_new = confidence_all[i+1, :, :]/4
        frame_grayscale_new[np.where(frame_grayscale_new > 255)] = 255
        frame_grayscale_new = np.uint8(frame_grayscale_new)

        # if i == 98:
        #     # Save frame_grayscale_new to a .mat file
        #     scipy.io.savemat('tablet_frame_grayscale_100.mat', {'frame_grayscale': frame_grayscale_new})

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame_grayscale, frame_grayscale_new, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # Now update the previous frame and previous points

        img_track = frame_grayscale_new.copy()
        frame_grayscale = frame_grayscale_new.copy()
        frame_x = x_all[i+1, :, :]
        frame_y = y_all[i+1, :, :]
        frame_z = z_all[i+1, :, :]

        p0 = good_new.reshape(-1, 1, 2)
        transformation_matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new)

        ##################
        D_signal_RR[0, i + 1] = np.average(frame_z[non_zero_pnts_neck[0], non_zero_pnts_neck[1]])
        D_signal_RR[1, i + 1] = np.average(frame_z[non_zero_pnts_chest[0], non_zero_pnts_chest[1]])

        ##################

        # No averaging of ROI
        pixels_cheek_n_nose_updated_transposed = np.dot(
            transformation_matrix, old_mask_cheek_n_nose_transposed)
        old_mask_cheek_n_nose_transposed = np.vstack(
            (pixels_cheek_n_nose_updated_transposed, np.ones(pixels_cheek_n_nose_updated_transposed.shape[1])))

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
        mask_cheek_n_nose_updated = vtx2mask(list(np.reshape(
            pixels_cheek_n_nose_updated_transposed.T, -1)), image_cols, image_rows)
        intensity_signals[i+1] = np.average(
            frame_confidence[np.where(mask_cheek_n_nose_updated > 0)])
        depth_signals[i+1] = np.average(np.sqrt(frame_x[np.where(mask_cheek_n_nose_updated > 0)]**2 + frame_y[np.where(
            mask_cheek_n_nose_updated > 0)]**2 + frame_z[np.where(mask_cheek_n_nose_updated > 0)]**2))

        # show tracking performance

        pixels_cheek_n_nose_updated = np.transpose(
            pixels_cheek_n_nose_updated_transposed)
        pixels_cheek_n_nose_updated = np.int32(pixels_cheek_n_nose_updated)
        img_track = np.stack((img_track,)*3, axis=-1)
        img_track = cv2.polylines(img_track, [pixels_cheek_n_nose_updated], True, color=(0, 0, 255), thickness=1)
        for kk, points in enumerate(good_new[:, 1]):
            img_track = cv2.circle(img_track, (int(good_new[kk, 0]), int(good_new[kk, 1])), radius=2, color=(0, 255, 255), thickness=-1)

        # cv2.imshow('frame', img_track)
        # cv2.waitKey(30)

    cv2.destroyAllWindows()

    # It seems like we might be able to use phase_two.py's output for intensity and depth for
    # the cheek_n_nose ROI and pass them in here in place of intensity_signals and depth_signals

    
    # BEGIN POST-PROCESSING CODE

    # Save pre-smoothened intensity_signals to a .mat file
    # scipy.io.savemat('tablet_intensity_signals.mat', {'intensity_signals': intensity_signals})

    # Save pre-smoothened depth_signals to a .mat file
    # scipy.io.savemat('tablet_depth_signals.mat', {'depth_signals': depth_signals})


    # Smoothen the depth and intensity data
    depth_signals_smooth = scipy.signal.savgol_filter(depth_signals, 9, 2, mode='nearest')

    plt.plot(intensity_signals)
    plt.show()

    intensity_signals_smooth = scipy.signal.savgol_filter(intensity_signals, 5, 2, mode='nearest')

    # Save post-smoothened intensity_signals to a .mat file
    # scipy.io.savemat('tablet_intensity_signals_smooth.mat', {'intensity_signals_smooth': intensity_signals_smooth})

    # Save post-smoothened depth_signals to a .mat file
    # scipy.io.savemat('tablet_depth_signals_smooth.mat', {'depth_signals_smooth': depth_signals_smooth})

    intensity_signals_compensated = compensate_intensity_using_depth(intensity_signals_smooth, depth_signals_smooth, sub_clip_length=2, frames_per_second=30)

    fps = 30
    num_seconds_between_frames = 1.0 / fps
    hr_magnitudes = abs(fft(intensity_signals_compensated))
    # Make the magnitudes one-sided (double the positive magnitudes)
    hr_magnitudes = 2.0 / num_frames * hr_magnitudes[:num_frames // 2]
    # Specify the range of frequencies to look at
    hr_frequencies = np.linspace(0.0, 1.0 / (2.0 * num_seconds_between_frames), num_frames // 2)
    # Convert frequencies from bps to bpm
    hr_frequencies = hr_frequencies * 60
    # Eliminate frequencies outside of the range of interest
    hr_magnitudes[np.where(hr_frequencies <= 40)] = 0
    hr_magnitudes[np.where(hr_frequencies >= 150)] = 0

    # Find all peak frequencies and select the peak with the greatest magnitude
    peaks, properties = scipy.signal.find_peaks(hr_magnitudes)
    max_index = np.argmax(hr_magnitudes[peaks])
    HR_with_depth_comp = hr_frequencies[peaks[max_index]]

    # Repeat the above process but without motion compensation
    hr_magnitudes_no_depth_comp = abs(fft(intensity_signals_smooth))
    hr_magnitudes_no_depth_comp = 2.0 / num_frames * hr_magnitudes_no_depth_comp[:num_frames // 2]
    hr_magnitudes_no_depth_comp[np.where(hr_frequencies <= 40)] = 0
    hr_magnitudes_no_depth_comp[np.where(hr_frequencies >= 150)] = 0

    peaks_no_depth_comp, properties_no_depth_comp = scipy.signal.find_peaks(hr_magnitudes_no_depth_comp)
    max_index_no_depth_comp = np.argmax(hr_magnitudes_no_depth_comp[peaks_no_depth_comp])
    HR_no_depth_comp = hr_frequencies[peaks_no_depth_comp[max_index_no_depth_comp]]

    # error_rate=abs(HR_comp-HR_ref)/abs(HR_ref)*100
    # success_label=error_rate<11
    # success_index=np.where(error_rate<10)
    # success_rate=np.shape(success_index)[1]/len(HR_comp)

    # f1 = plt.figure()
    # ax1 = f1.add_subplot(221)
    # ax1.plot(I_signal_smooth)
    # plt.figure()
    # plt.plot(D_signal_smooth)
    # plt.show()

    # please work - isiah
    # np.savetxt('test.out', depth_signals, delimiter=',')

    plt.figure()
    plt.plot(depth_signals)
    #plt.show()

    # ax2 = f1.add_subplot(222)
    # ax2.plot(D_signal_smooth)
    # ax3 = f1.add_subplot(223)
    # ax3.plot(I_compensated)
    # ax4 = f1.add_subplot(224)
    # ax4.plot(xf, yf)
    # plt.xlim(0, 200)
    # plt.ylim(0,0.5)

    # print("Heart Rate Measured", HR_with_depth_comp)
    # print("Heart Rate Measured (No Motion Comp)", HR_no_depth_comp)
    print(f'Tablet HR: {HR_with_depth_comp}')
    print(f'Tablet HR_ND: {HR_no_depth_comp}')

    # END POST-PROCESSING CODE


    # BEGIN CHEST ROI CODE


    '''
    print(D_signal_RR.shape)
    plt.figure()
    plt.plot(D_signal_RR[0,:])
    plt.plot(D_signal_RR[1,:])
    plt.show()

    plt.figure()
    plt.plot(I_signal)
    plt.show()

    '''
    # transformer = FastICA(n_components=2, max_iter=500, whiten='arbitrary-variance', tol=5e-3)
    # X_transformed = transformer.fit_transform(D_signal_RR.T)

    # plt.figure()
    # plt.plot(X_transformed[:,0])
    # plt.plot(X_transformed[:,1])
    # plt.show()

    # yf_rr1 = abs(fft(X_transformed[:, 0]))
    # yf_rr1 = 2.0 / num_frames * yf_rr1[:num_frames // 2]
    # xf_rr1 = np.linspace(0.0, 1.0 / (2.0 * num_seconds_between_frames), num_frames // 2)
    # xf_rr1 = xf_rr1 * 60
    # yf_rr1[np.where(xf_rr1 <= 5)] = 0
    # yf_rr1[np.where(xf_rr1 >= 30)] = 0

    # peaks, properties = scipy.signal.find_peaks(yf_rr1)
    # max_index_1 = np.argmax(yf_rr1[peaks])

    # yf_rr2 = abs(fft(X_transformed[:, 1]))
    # yf_rr2 = 2.0 / num_frames * yf_rr1[:num_frames // 2]
    # xf_rr2 = np.linspace(0.0, 1.0 / (2.0 * num_seconds_between_frames), num_frames // 2)
    # xf_rr2 = xf_rr2 * 60
    # yf_rr2[np.where(xf_rr2 <= 5)] = 0
    # yf_rr2[np.where(xf_rr2 >= 30)] = 0

    # peaks, properties = scipy.signal.find_peaks(yf_rr2)
    # max_index_2 = np.argmax(yf_rr2[peaks])

    # if yf_rr1[peaks[max_index_1]] > yf_rr2[peaks[max_index_2]]:
    #     RR_ica = xf_rr1[peaks[max_index_1]]
    # else:
    #     RR_ica = xf_rr2[peaks[max_index_2]]

    # print("Respiratory Rate Measured", RR_ica)


    # END CHEST ROI CODE


    # Read temperature
    # read_temp('/dev/ttyUSB0')
    # plt.figure()
    # plt.plot(X_transformed)
    # plt.show()

    # plt.figure()
    # plt.plot(xf_rr, yf_rr)
    # plt.show()





    #### MAIN CODE PROCESSING ####
    I_comp_main = depthComp(np.reshape(intensity_signals_smooth, (1,-1)), np.reshape(depth_signals_smooth, (1,-1)))
    HR_ND_main = getHR(np.reshape(intensity_signals_smooth, (-1)), 600)
    HR_main = getHR(np.reshape(I_comp_main, (-1)), 600)

    print(f'Main HR: {HR_main}')
    print(f'Main HR_ND: {HR_ND_main}')

