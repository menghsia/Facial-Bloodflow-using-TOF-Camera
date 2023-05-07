import numpy as np
from multiprocessing import Pool, RawArray

# Arrays storing intensity and depth signals for all ROIs in this video clip (num_ROIs, num_frames) = (7, 600)
num_ROIs = 7
num_frames = 600
intensity_signal_current = np.zeros((num_ROIs, num_frames))
depth_signal_current = np.zeros((num_ROIs, num_frames))
ear_signal_current = np.zeros(num_frames)

# Create shared memory for the intensity and depth signals
intensity_signal_current_shared = RawArray('d', intensity_signal_current.flatten())
depth_signal_current_shared = RawArray('d', depth_signal_current.flatten())

# Reshape the shared memory back to the original shape of the arrays
intensity_signal_current = np.frombuffer(intensity_signal_current_shared).reshape(num_ROIs, num_frames)
depth_signal_current = np.frombuffer(depth_signal_current_shared).reshape(num_ROIs, num_frames)

# Define the function that will be run in parallel
def process_single_frame(frame):
    # Each array is currently (height*width, num_frames) = (480*640, num_frames) = (307200, num_frames)
    # Reshape to (height, width, num_frames) = (480, 640, num_frames)
    x_all = x_all.reshape([img_rows, img_cols, num_frames])
    y_all = y_all.reshape([img_rows, img_cols, num_frames])
    z_all = z_all.reshape([img_rows, img_cols, num_frames])
    gray_all = gray_all.reshape([img_rows, img_cols, num_frames])

    self._process_single_frame(x_all=x_all, y_all=y_all, z_all=z_all, gray_all=gray_all,
                               img_rows=img_rows, img_cols=img_cols, frame=frame,
                               face_mesh=face_mesh, intensity_signal_current=intensity_signal_current,
                               depth_signal_current=depth_signal_current, ear_signal_current=ear_signal_current,
                               mp_drawing=mp_drawing, mp_drawing_styles=mp_drawing_styles,
                               mp_face_mesh=mp_face_mesh, visualize_ROI=visualize_ROI,
                               visualize_FaceMesh=visualize_FaceMesh)

# Define the pool of workers
with Pool(processes=16) as pool:
    # Map the function over the range of frames to process each frame in parallel
    pool.map(process_single_frame, range(num_frames))
