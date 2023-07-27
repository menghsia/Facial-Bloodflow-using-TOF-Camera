import math
import numpy as np
from typing import Tuple, List


class DepthCompensator:
    def run(self, intensity: np.ndarray, depth: np.ndarray, a_range: Tuple[float, float] = (1.0, 1.0), b_range: Tuple[float, float] = (0.0, 6.0), a_step: float = 0.01, b_step: float = 0.01, window_length: int = 2, fps: int = 30) -> np.ndarray:
        """
        1. Split the input array of intensity signals into subarrays of window_length seconds
        2. Apply depth compensation to each subarray
        3. Concatenate the depth-compensated intensity arrays.
    
        Args:
            intensity: 1D array of intensity signals.
            depth: 1D array of depth signals.
            a_range: Range of a values to try
            b_range: Range of b values to try
            a_step: Step size for a values
            b_step: Step size for b values
            window_length: Length of each subarray in seconds.
            fps: Frames per second of the video.
    
        Returns:
            1D array of depth-compensated intensity signals concatenated together.
        """
        # Use window_length and fps to calculate the number of frames in each subarray
        frames_per_window = window_length * fps
    
        # Split the intensity and depth arrays into subarrays of length frames_per_window
        intensity_windows = self._split_into_subarrays(intensity, frames_per_window)
        depth_windows = self._split_into_subarrays(depth, frames_per_window)
    
        num_subarrays = len(intensity_windows)
    
        # Create an array of zeros with the same length as intensity
        intensity_compensated = np.zeros_like(intensity)
    
        for subarray_idx in range(num_subarrays):
            # Calculate the start and end indices to set in intensity_compensated
            start_idx = subarray_idx * frames_per_window
            end_idx = start_idx + len(intensity_windows[subarray_idx])
            
            # Apply depth compensation to each subarray
            intensity_compensated[start_idx:end_idx] = self._compensate_intensity_using_depth(intensity_windows[subarray_idx], depth_windows[subarray_idx], a_range, b_range, a_step, b_step)
    
        return intensity_compensated
    

    def _convert_to_z_scores(self, array: np.ndarray) -> np.ndarray:
        """
        Converts an array to z-scores.
    
        The output data will have zero mean and unit variance.
    
        Args:
            array: 1D array of data
    
        Returns:
            z_scores: 1D array of z-scores
        """
        mean = np.mean(array)
        standard_deviation = np.std(array)
        z_scores = (array - mean) / standard_deviation
    
        return z_scores
    

    def _compensate_intensity_using_depth(self, intensity: np.ndarray, depth: np.ndarray, a_range: Tuple[float, float], b_range: Tuple[float, float], a_step: float, b_step: float) -> np.ndarray:
        """ 
        Takes in a 1D array of intensity signals and a 1D array of depth signals and returns a 1D array of depth-compensated intensity signals.
        The output array is the same length as the input arrays.
        The output array is normalized to have mean 0 and standard deviation 1 (z-score normalization).
    
        Uses the formula intensity_compensated = intensity_raw / (a * depth_raw^(-b)) where a and b are constants.
        a, b are chosen to minimize the correlation between intensity_compensated and depth_raw.
        We use pearson correlation coefficient to measure correlation and minimize the absolute value of the correlation coefficient.
    
        Args:
            intensities: 1D array of intensity signals
            depths: 1D array of depth signals
            a_range: Range of a values to try
            b_range: Range of b values to try
            a_step: Step size for a values
            b_step: Step size for b values
        
        Returns:
            intensity_compensated: 1D array of depth-compensated intensity signals
        """
        if a_range[1] < a_range[0]:
            raise ValueError("a_range[1] must be greater than or equal to a_range[0]")
        elif a_range[0] == a_range[1]:
            a_values = np.array([a_range[0]])
        else:
            a_values = np.arange(a_range[0], a_range[1], a_step)
        
        if b_range[1] < b_range[0]:
            raise ValueError("b_range[1] must be greater than or equal to b_range[0]")
        elif b_range[0] == b_range[1]:
            b_values = np.array([b_range[0]])
        else:
            b_values = np.arange(b_range[0], b_range[1], b_step)

        # # DEBUGGING: Used to compare against previous implementation
        # a_values=[1.0]
        # b_values=np.arange(0.2, 5.1, 0.1)
    
        best_intensity_compensated = np.zeros(intensity.shape)
        # Save lowest_correlation as maximum possible value for the float type
        lowest_corr_coeff_absolute = np.finfo(float).max
        # best_a = np.finfo(float).max
        # best_b = np.finfo(float).max
    
        for a_value in a_values:
            for b_value in b_values:
                intensity_compensated = intensity * (a_value * (depth**b_value))
    
                correlation_matrix = np.corrcoef(intensity_compensated, depth)
                corr_coeff_absolute = abs(correlation_matrix[1, 0])
    
                if corr_coeff_absolute < lowest_corr_coeff_absolute:
                    best_intensity_compensated = intensity_compensated
                    lowest_corr_coeff_absolute = corr_coeff_absolute
                    # best_a = a_value
                    # best_b = b_value
        
        # print(f"Best a: {best_a}")
        # print(f"Best b: {best_b}")
        # print(f"Lowest correlation coefficient absolute value: {lowest_corr_coeff_absolute}")
    
        best_intensity_compensated = self._convert_to_z_scores(best_intensity_compensated)
    
        return best_intensity_compensated
    
    
    def _split_into_subarrays(self, array: np.ndarray, subarray_length: int) -> List[np.ndarray]:
        """
        Splits the input array into distinct subarrays of a specified length each.

        If the input array does not divide evenly into these subarrays, the function will split the input array into as many complete subarrays of the specified length as possible,
        and then the last subarray will be the remaining length, which would be shorter than the specified length.

        Args:
            array: 1D array to be split into subarrays.
            subarray_length: Maxmimum length of each subarray.

        Returns:
            List of 1D subarrays.
        """
        length = len(array)
    
        if length <= subarray_length:
            return [array]
    
        num_subarrays = length // subarray_length
        remaining_length = length % subarray_length
    
        subarrays = np.split(array[:length - remaining_length], num_subarrays)
        
        if remaining_length > 0:
            subarrays.append(array[-remaining_length:])
    
        return subarrays


def compensate_intensity_using_depth_old(intensities, depths, b_values=np.arange(0.2, 5.1, 0.1), a_values=[1.0], sub_clip_length=2, frames_per_second=30):
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


if __name__ == "__main__":
    print("Testing depth_compensation.py...")

    # Test 1: Test that the function works with a simple example

    # Use 100 random floats between 0 and 12 for intensity and depth
    intensity = np.array([3.930, 0.186, 4.066, 3.550, 0.480, 8.340, 2.755, 7.387, 8.279, 2.316,
                          4.450, 11.998, 5.326, 7.113, 11.108, 3.898, 8.616, 6.404, 6.796, 0.415,
                          11.678, 11.353, 3.950, 1.417, 7.903, 3.332, 1.097, 4.658, 4.852, 0.627,
                          4.637, 0.568, 5.997, 6.525, 8.207, 11.360, 3.818, 7.389, 7.411, 0.574,
                          5.194, 5.554, 2.932, 10.020, 2.678, 9.699, 1.018, 9.344, 8.311, 2.258,
                          8.753, 2.703, 3.509, 10.425, 11.438, 2.607, 5.210, 6.657, 4.197, 4.404,
                          5.417, 3.387, 4.287, 6.752, 11.609, 11.862, 8.174, 0.801, 10.569, 1.073,
                          7.381, 6.925, 3.581, 6.617, 6.156, 7.631, 1.184, 10.938, 3.296, 11.055,
                          10.528, 7.813, 1.549, 5.227, 7.370, 5.382, 2.760, 10.141, 8.695, 9.020,
                          8.268, 2.208, 4.558, 7.770, 10.624, 6.091, 1.390, 6.813, 7.540, 0.916])
    
    depth = np.array([5.433, 6.821, 8.935, 6.902, 9.769, 7.834, 9.948, 1.719, 10.882, 11.594,
                      11.737, 2.247, 6.251, 7.270, 1.239, 0.766, 3.338, 5.252, 4.000, 2.758,
                      1.510, 6.069, 7.650, 4.868, 1.708, 8.836, 3.596, 0.779, 3.158, 4.836,
                      3.105, 2.959, 1.274, 7.309, 4.172, 5.440, 5.766, 0.760, 2.306, 0.561,
                      4.306, 1.474, 3.167, 8.053, 9.795, 3.483, 5.774, 10.980, 6.926, 10.900,
                      0.541, 5.905, 6.481, 11.029, 8.809, 3.549, 5.080, 10.488, 4.809, 8.478,
                      4.407, 6.337, 5.734, 0.508, 5.715, 0.369, 11.840, 9.031, 0.743, 9.480,
                      0.832, 6.310, 5.047, 5.211, 4.339, 9.114, 9.689, 0.834, 11.637, 0.316,
                      3.487, 1.172, 11.254, 3.597, 5.145, 0.485, 6.402, 10.599, 3.480, 3.643,
                      0.552, 0.932, 2.431, 10.583, 0.177, 4.720, 5.949, 5.185, 5.875, 8.844])
    
    old_intensity_compensated = compensate_intensity_using_depth_old(intensity, depth)
    # print(f"Old intensity_compensated: {old_intensity_compensated}")
    
    depth_compensator = DepthCompensator()

    # print(f"Intensity: {intensity}")
    intensity_compensated = depth_compensator.run(intensity, depth, a_range=(1.0, 1.0), b_range=(0.0, 6.0), a_step=0.1, b_step=0.1, window_length=2, fps=30)
    # print(f"New intensity_compensated: {intensity_compensated}")

    # Assert that the old and new intensity compensated arrays are the same
    assert np.allclose(old_intensity_compensated, intensity_compensated), "Old and new intensity compensated arrays are not the same"


    # Test 2: Test that the function works with an example that is closer to the real data length and range
    
    # Generate 633 random floats between 0 and 13000 for intensity and depth
    intensity = np.random.uniform(low=0.0, high=13000.0, size=633)
    depth = np.random.uniform(low=0.0, high=13000.0, size=633)

    old_intensity_compensated = compensate_intensity_using_depth_old(intensity, depth)
    intensity_compensated = depth_compensator.run(intensity, depth, a_range=(1.0, 1.0), b_range=(0.0, 6.0), a_step=0.1, b_step=0.1, window_length=2, fps=30)

    assert np.allclose(old_intensity_compensated, intensity_compensated), "Old and new intensity compensated arrays are not the same"

    
    # Test 3: Arrays shorter than fps

    intensity = np.random.uniform(low=0.0, high=13000.0, size=29)
    depth = np.random.uniform(low=0.0, high=13000.0, size=29)

    old_intensity_compensated = compensate_intensity_using_depth_old(intensity, depth)
    intensity_compensated = depth_compensator.run(intensity, depth, a_range=(1.0, 1.0), b_range=(0.0, 6.0), a_step=0.1, b_step=0.1, window_length=2, fps=30)

    assert np.allclose(old_intensity_compensated, intensity_compensated), "Old and new intensity compensated arrays are not the same"


    # Test 4: Arrays exactly as long as fps

    intensity = np.random.uniform(low=0.0, high=13000.0, size=30)
    depth = np.random.uniform(low=0.0, high=13000.0, size=30)

    old_intensity_compensated = compensate_intensity_using_depth_old(intensity, depth)
    intensity_compensated = depth_compensator.run(intensity, depth, a_range=(1.0, 1.0), b_range=(0.0, 6.0), a_step=0.1, b_step=0.1, window_length=2, fps=30)

    assert np.allclose(old_intensity_compensated, intensity_compensated), "Old and new intensity compensated arrays are not the same"

    print("All tests passed!")