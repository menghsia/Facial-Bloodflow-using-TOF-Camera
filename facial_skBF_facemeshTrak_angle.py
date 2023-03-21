import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import scipy
import math
import mat73
from scipy.io import savemat, loadmat
from sklearn import linear_model
import cv2
from typing import ChainMap, Tuple, Union
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import time
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


def ROI_coord_extract(face_landmarks, ROIwhich, image_rows, image_cols):
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
    elif ROIwhich == 'forehead':
        # ROI_vertex = [109,338,336,107]
        ROI_vertex = [109, 338, 9]
    elif ROIwhich == 'left_eye':
        ROI_vertex = [33, 160, 159, 158, 133, 153, 145, 144]
    elif ROIwhich == 'right_eye':
        ROI_vertex = [263, 387, 386, 385, 362, 380, 374, 373]
    else:
        print('No such ROI')
        quit()
    # Landmarks can be found on https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    # Facemesh detection

    # Extract coordinates of all pixels within the ROI polygon
    landmark_px = []

    for i, vtx in enumerate(ROI_vertex):
        landmark_current = _normalized_to_pixel_coordinates(face_landmarks.landmark[vtx].x,
                                                            face_landmarks.landmark[vtx].y, image_cols, image_rows)
        landmark_px.append(landmark_current)
        # print(landmark_px)

    # n-by-2 2d array
    return landmark_px


def Chest_ROI_extract(image, chin_location, plot=False):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    image_3chnl = np.stack((image,) * 3, axis=-1)
    image_3chnl = cv2.convertScaleAbs(image_3chnl)
    shoulder_landmark = [11, 12]
    landmark_px_rr = np.zeros([2, 4, 2])
    image_height, image_width = image.shape
    results = pose.process(image_3chnl)
    body_points = results.pose_landmarks.landmark
    shoulder_point_l = _normalized_to_pixel_coordinates(body_points[11].x, body_points[11].y, image_width, image_height)
    shoulder_point_r = _normalized_to_pixel_coordinates(body_points[12].x, body_points[12].y, image_width, image_height)

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


def mp_preprocess(frameTrk, divisor=4):
    frameTrk = frameTrk.astype(float)
    frameTrk = frameTrk / divisor
    frameTrk[np.where(frameTrk > 255)] = 255
    frameTrk = frameTrk.astype('uint8')
    image_3chnl = np.stack((frameTrk,) * 3, axis=-1)

    return image_3chnl


def norm_vector_ls(Coords):
    """
    This function takes 3-D coordinates as input, output the normal direction of the least square
    fitted plane of those coordinates.


    Coords: N x 3 matrix, containing N coordinates in 3-D, [x,y,z]

    n: normal direction of the surface, [x,y,z]

    """

    # Creat LS problem Ax=z, solve for normal vector n
    N = Coords.shape[0]
    A = np.hstack((Coords[:, 0:1], Coords[:, 2:3], np.ones((N, 1))))
    z = Coords[:, 1:2]
    x = np.linalg.pinv(A) @ z
    n = np.array([x[0], -1, x[1]], dtype=float)

    return n


def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = math.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = math.atan2(z, math.sqrt(XsqPlusYsq))  # theta
    az = math.atan2(y, x)  # phi
    return r, elev, az


def norm_vector_ransac(Coords, thrh):
    """
    get coefficient
    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        PointCloudData
    Returns
    -------
    a, d, d : float
        Coefficients of the plane equation(Z = aX + bY + d).

    n : norm vector of the plane
    """

    xz = np.hstack((Coords[:, 0:1], Coords[:, 2:3]))

    y = Coords[:, 1]
    ransac = linear_model.RANSACRegressor(residual_threshold=thrh, max_trials=500, min_samples=0.5)

    ransac.fit(xz, y)
    a, b = ransac.estimator_.coef_  # coefficients
    d = ransac.estimator_.intercept_  # intercept
    # Y = aX + bZ + d
    n = np.array([a, -1, b])
    # print(a, b, d)

    r, elev, az = cart2sph(a, -1, b)
    return a, b, d, n, elev, az


def extract_geoParam(xSig, ySig, zSig, mask):
    surface_coords1 = np.array([xSig[np.where(mask > 0)], ySig[np.where(mask > 0)], zSig[np.where(mask > 0)]]).T
    a, b, d, n, eleObj, azObj = norm_vector_ransac(surface_coords1, 0.5)
    [CamX, CamY, CamZ] = np.average(surface_coords1, axis=0)
    _, eleCam, azCam = cart2sph(CamX, CamY, CamZ)
    surface_center = np.average(surface_coords1, axis=0)
    distance = np.linalg.norm(surface_center)
    theta = np.arccos(abs(np.dot(n, surface_center)) / (distance * np.linalg.norm(n)))

    return theta, eleObj, azObj, eleCam, azCam

def eye_aspect_ratio(eye):
    # Vertical eye landmarks
    A = dist.euclidean(eye[2], eye[3])
    B = dist.euclidean(eye[4], eye[5])
    C = dist.euclidean(eye[6], eye[7])
    # Horizontal eye landmarks
    D = dist.euclidean(eye[0], eye[1])

    # The EAR Equation
    EAR = (A + B + C) / (3.0 * D)
    return EAR

matpath = './20220601_motion_cold/pat_test_l/'
matname = 'pat_test_l_bfSig'

i = 0
start_frame = 0
img_cols = 640
img_rows = 480

I_signal = np.zeros((5, 1))
D_signal = np.zeros((5, 1))
Ang_signal = np.zeros((5, 1))
eleObj_signal = np.zeros((5, 1))
azObj_signal = np.zeros((5, 1))
eleCam_signal = np.zeros((5, 1))
azCam_signal = np.zeros((5, 1))
x_signal = np.zeros((5, 1))
y_signal = np.zeros((5, 1))
z_signal = np.zeros((5, 1))

filelist = []
for filename in os.listdir(matpath):
    if filename.endswith('.skv.mat'):
        filelist.append(filename)

counter = 0

# Setting up the facemesh parameters
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    time1 = time.time()
    for filename in filelist:

        try:
            mat_data = mat73.loadmat(matpath + filename)
        except:
            mat_data = loadmat(matpath + filename)

        gray_all = mat_data['grayscale']
        x_all = mat_data['x_value']
        y_all = mat_data['y_value']
        z_all = mat_data['z_value']

        counter = counter + 1
        print(counter)

        frame_num = np.size(gray_all, 1)
        I_signal_current = np.zeros(
            (5, frame_num))  # 1: nose;  2: forehead;   3: nose & cheek  4: left cheek   5: right cheek
        D_signal_current = np.zeros((5, frame_num))
        Ang_signal_current = np.zeros((5, frame_num))
        eleObj_signal_current = np.zeros((5, frame_num))
        azObj_signal_current = np.zeros((5, frame_num))
        eleCam_signal_current = np.zeros((5, frame_num))
        azCam_signal_current = np.zeros((5, frame_num))

        x_signal = np.zeros((5, frame_num))
        y_signal = np.zeros((5, frame_num))
        z_signal = np.zeros((5, frame_num))

        time2 = time.time()
        timeelps = time2 - time1
        print('Data loading completed, time elapsed: %s seconds' % timeelps)

        gray_all = np.reshape(gray_all, [img_rows, img_cols, frame_num])
        x_all = np.reshape(x_all, [img_rows, img_cols, frame_num])
        y_all = np.reshape(y_all, [img_rows, img_cols, frame_num])
        z_all = np.reshape(z_all, [img_rows, img_cols, frame_num])

        frame_num = np.size(gray_all, 2)

        for j in range(frame_num):
            frameTrk = mp_preprocess(gray_all[:, :, j])
            frameSig = gray_all[:, :, j]
            xSig = x_all[:, :, j].astype('int32')
            ySig = y_all[:, :, j].astype('int32')
            zSig = z_all[:, :, j].astype('int32')
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frameTrk.flags.writeable = False
            frameTrk = cv2.cvtColor(frameTrk, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frameTrk)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

            # Extract ROI masks using the coordinate that mediapipe provides

            landmark_forehead = ROI_coord_extract(face_landmarks, 'forehead', img_rows, img_cols)
            mask_forehead = vtx2mask(landmark_forehead, img_cols, img_rows)
            landmark_nose = ROI_coord_extract(face_landmarks, 'nose', img_rows, img_cols)
            mask_nose = vtx2mask(landmark_nose, img_cols, img_rows)
            landmark_cn = ROI_coord_extract(face_landmarks, 'cheek_n_nose', img_rows, img_cols)
            mask_cn = vtx2mask(landmark_cn, img_cols, img_rows)
            landmark_lc = ROI_coord_extract(face_landmarks, 'left_cheek', img_rows, img_cols)
            mask_lc = vtx2mask(landmark_lc, img_cols, img_rows)
            landmark_rc = ROI_coord_extract(face_landmarks, 'right_cheek', img_rows, img_cols)
            mask_rc = vtx2mask(landmark_rc, img_cols, img_rows)

            ## Fit plane with point clouds, and calculate the angle between surface and the optical axis

            Ang_signal_current[1, j], eleObj_signal_current[1, j], azObj_signal_current[1, j], eleCam_signal_current[
                1, j], azCam_signal_current[1, j] = \
                extract_geoParam(xSig, ySig, zSig, mask_forehead)
            Ang_signal_current[3, j], eleObj_signal_current[3, j], azObj_signal_current[3, j], eleCam_signal_current[
                3, j], azCam_signal_current[3, j] = \
                extract_geoParam(xSig, ySig, zSig, mask_lc)
            Ang_signal_current[4, j], eleObj_signal_current[4, j], azObj_signal_current[4, j], eleCam_signal_current[
                4, j], azCam_signal_current[4, j] = \
                extract_geoParam(xSig, ySig, zSig, mask_rc)

            ## To see plane fitting performance, use the following code:
            # xz= np.hstack((surface_coords1[:, 0:1],surface_coords1[:, 2:3]))
            # y = xz @ np.array([a,b]) + d
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.scatter3D(surface_coords1[:,0],surface_coords1[:,1],surface_coords1[:,2],cmap='viridis', s = 1)
            # ax.scatter3D(xz[:,0],y,xz[:,1],cmap='viridis', s = 1)
            # ax.plot3D([surface_center[0],surface_center[0]+10*n[0]],
            #       [surface_center[1],surface_center[1]+10*n[1]],
            #       [surface_center[2],surface_center[2]+10*n[2]], 'gray')
            #
            # ax.set_ylim(300,400)
            # ax.set_xlim(-50,50)
            # ax.set_zlim(-33,33)
            # plt.show()



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

            #PERCLOS
            # landmark_leye = ROI_coord_extract(face_landmarks, 'left_eye', img_rows, img_cols)
            # L_ear = eye_aspect_ratio(landmark_leye)
            # landmark_reye = ROI_coord_extract(face_landmarks, 'right_eye', img_rows, img_cols)
            # R_ear = eye_aspect_ratio(landmark_reye)
            #
            # Draw the face mesh annotations on the image and display
            frameTrk.flags.writeable = True
            image = cv2.cvtColor(frameTrk, cv2.COLOR_RGB2BGR)
            pts = np.asarray(landmark_forehead)
            pts = pts.reshape((-1, 1, 2))
            img_showROI = cv2.polylines(image, [pts], True, color=(0, 0, 255), thickness=2)
            pts = np.asarray(landmark_nose)
            pts = pts.reshape((-1, 1, 2))
            img_showROI = cv2.polylines(img_showROI, [pts], True, color=(0, 0, 255), thickness=2)
            # pts = np.asarray(landmark_leye)
            # pts = pts.reshape((-1, 1, 2))
            # img_showROI = cv2.polylines(img_showROI, [pts], True, color=(0, 0, 255), thickness=2)
            # pts = np.asarray(landmark_reye)
            # pts = pts.reshape((-1, 1, 2))
            # img_showROI = cv2.polylines(img_showROI, [pts], True, color=(0, 0, 255), thickness=2)
            #
            cv2.imshow('ROI', img_showROI)
            cv2.waitKey(10)
            # for face_landmarks in results.multi_face_landmarks:
            #     mp_drawing.draw_landmarks(
            #         image=image,
            #         landmark_list=face_landmarks,
            #         connections=mp_face_mesh.FACEMESH_TESSELATION,
            #         landmark_drawing_spec=None,
            #         connection_drawing_spec=mp_drawing_styles
            #         .get_default_face_mesh_tesselation_style())
            #     mp_drawing.draw_landmarks(
            #         image=image,
            #         landmark_list=face_landmarks,
            #         connections=mp_face_mesh.FACEMESH_CONTOURS,
            #         landmark_drawing_spec=None,
            #         connection_drawing_spec=mp_drawing_styles
            #         .get_default_face_mesh_contours_style())
            #     mp_drawing.draw_landmarks(
            #         image=image,
            #         landmark_list=face_landmarks,
            #         connections=mp_face_mesh.FACEMESH_IRISES,
            #         landmark_drawing_spec=None,
            #         connection_drawing_spec=mp_drawing_styles
            #         .get_default_face_mesh_iris_connections_style())
            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            # cv2.waitKey(10)

        I_signal = np.concatenate((I_signal, I_signal_current), axis=1)
        D_signal = np.concatenate((D_signal, D_signal_current), axis=1)
        Ang_signal = np.concatenate((Ang_signal, Ang_signal_current), axis=1)
        eleObj_signal = np.concatenate((eleObj_signal, eleObj_signal_current), axis=1)
        azObj_signal = np.concatenate((azObj_signal, azObj_signal_current), axis=1)
        eleCam_signal = np.concatenate((eleCam_signal, eleCam_signal_current), axis=1)
        azCam_signal = np.concatenate((azCam_signal, azCam_signal_current), axis=1)
I_signal = np.delete(I_signal, 0, 1)
D_signal = np.delete(D_signal, 0, 1)
Ang_signal = np.delete(Ang_signal, 0, 1)
eleObj_signal = np.delete(eleObj_signal, 0, 1)
eleCam_signal = np.delete(eleCam_signal, 0, 1)
azObj_signal = np.delete(azObj_signal, 0, 1)
azCam_signal = np.delete(azCam_signal, 0, 1)

mdic = {"Depth": D_signal, 'I_raw': I_signal, 'Ang_signal': Ang_signal, 'eleObj_signal': eleObj_signal,
        'azObj_signal': azObj_signal, 'eleCam_signal': eleCam_signal, 'azCam_signal': azCam_signal, }
savemat(os.path.join(matpath, matname + '.mat'), mdic)

print('finished')
