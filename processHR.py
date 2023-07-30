import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time
from scipy.fftpack import fft
import math

from depth_compensation import DepthCompensator


class ProcessHR():
    """
    ProcessHR is a class that uses depths and intensities to output a calculated heartrate.

    Args:
        input_file (str): filename from the correct directory where while is located

    Attributes:
        input_file (str): input filename in which raw depths and intensities are stored
        time (float): runtime of run() from start to finish
    """

    def __init__(self, input_file):
        """
        Initializes class variables

        Args:
            input_file (str): filename from the correct directory where while is located
        """  

        self.input_file = input_file
        self.time = 0
    
    def run(self):
        """
        Runs supporting functions to retreive relavant values and output plots.
        Supporting functions:
            Plots Raw and Compensated Forehead Intensities 
            Plots Relative Blood Concentration Change for Nose, Forehead and Cheek ROIs
            Plots 3 Heart Rate Frequency Spectrum Graphs w and w/o motion compensation for three 20 second clips
            Return Average Heart Rates in terminal w and w/o motion comp. for the three 20 second clips

        Args:
            None 

        """

        # tb: time array (doubles)
        # bc: blood concentration (au) array
        # HRsig: Heart Rate Signal array
        # Depth: Raw depth array
        start_HRtime = time.time()
        HRsig, HRsigRaw, I_comp, Depth, I_raw= self.processRawData()

        # Plot smoothed blood concentration at times(s)
        # plt.figure()

        # plt.plot(tb, self.smooth(bc[1], 51), color = 'blue')
        # plt.plot(tb, self.smooth(bc[0], 51), color = 'red')
        # plt.plot(tb, self.smooth(((bc[2]+bc[3])/2), 51), color = 'orange')

        # plt.xlabel('Time (s)')
        # plt.legend(['Nose', 'Forehead', 'Cheek Average'])
        # plt.ylabel('Relative Blood Concentration Change (a.u.)')
        

        ## getHR() NEEDS FIXING ##
        # Get HR Data
        HR_comp = self.getHR(HRsig, 600)
        HR_ND = self.getHR(HRsigRaw, 600)

        print(f'Main HR: {HR_comp}')
        print(f'Main HR_ND: {HR_ND}')


        I_comp_tab = self.tablet_depthComp(I_raw[2,:], Depth[2,:])
        HR_comp_tab = self.tablet_getHR(I_comp_tab, 600)
        HR_ND_tab = self.tablet_getHR(HRsigRaw, 600)

        print()
        print(f'Tablet HR: {HR_comp}')
        print(f'Tablet HR_ND: {HR_ND}')

        # Calculate Heart Rate (Motion Score)
        #self.motionComp(HRsig, Depth)
        end_HRtime = time.time()
        self.time = end_HRtime - start_HRtime
        #plt.show()
        return 

    def processRawData(self, dataTitle = None):
        """
        processRawData:
        Extracts raw depths and intensities from .mat file
        Plots compensated and raw forehead intensities
        Stores Heart Rate Signal info 

        Args:
            dataTitle (str): title of data that is being processed (OPTIONAL)

        Returns:
            tb (1D array of ints): Time (seconds)
            bc (Stacked arrays):  Smoothed blood concentrations (au) for nose, forehead, left cheek, and right cheek ROIs
            HRsig (1D array): Cheek and Nose ROI compensated intensity used for heart rate signal
            HRsigRaw (1D array): Cheek and Nose ROI raw intensity
            I_comp (2D array): Compensated intensities
            Depth (2D array):  Raw depths 

        """

        # Load in raw depth and intensities
        data = scipy.io.loadmat(self.input_file)
        # Depth: 2D array of depths (7x1800 for a 1 minute clip)
        # I_raw: 2D array of raw intensities (7x1800 for a 1 minute clip)
        Depth = data['Depth'] 
        I_raw = data['I_raw']

        # Remove extraneous zeros
        for i in range(Depth.shape[1]-1, -1, -1):
            if Depth[0, i] == 0:
                Depth = np.delete(Depth, i, axis=1)
                I_raw = np.delete(I_raw, i, axis=1)
            else:
                break

        Depth = np.delete(Depth, 6, axis=0)
        I_raw = np.delete(I_raw, 6, axis=0)

        # Smooth each ROI in the 2D arrays of depths and intensities
        for i in range(6):
            Depth[i,:] = scipy.signal.savgol_filter(Depth[i,:], 9, 2, mode='nearest', axis=0)
            I_raw[i,:] = scipy.signal.savgol_filter(I_raw[i,:], 5, 2, mode='nearest', axis=0)

        # Depth = scipy.signal.savgol_filter(Depth, 9, 2, mode='nearest', axis=0)
        # I_raw = scipy.signal.savgol_filter(I_raw, 5, 2, mode='nearest', axis=0)

        # Compensate for movement
        # I_comp: 2D array of compensated intensities
        I_comp = self.depthComp(I_raw, Depth, 2, 30)

        depth_compensator = DepthCompensator()
        moose_I_comp = depth_compensator.run(I_raw, Depth, window_length=2, fps=30)

        plt.plot(I_raw[2,:])
        plt.show()
        
        # Process waveforms into the different regions
        Fs = 30 # Frames/Second
        T = 1 / Fs

        # Cheek and nose ROI is set aside for Heart Rate Signal calculation
        HRsig = I_comp[2, :] 
        HRsigRaw = I_raw[2, :]

        # Smoothed blood concentrations for nose, forehead, left cheek, and right cheek ROIs
        # bc_nose, bc_forehead, bc_lc, bc_rc: 1D array
        # bc_nose = self.smooth(-np.log(I_comp[0, :]), 19)
        # bc_forehead = self.smooth(-np.log(I_comp[1, :]), 19)
        # bc_lc = self.smooth(-np.log(I_comp[3, :]), 19)
        # bc_rc = self.smooth(-np.log(I_comp[4, :]), 19)

        # # Stacked bc's for plotting time and blood concentrations
        # bc = np.vstack((bc_forehead, bc_nose, bc_lc, bc_rc))
        # tb = np.arange(0, I_raw.shape[1]) * T


        # Plots Raw and Compensated cheek and nose intensities
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].plot(I_raw[2, :])
        axs[0].set_ylabel('Raw Intensity')


        axs[1].plot(I_comp[2, :])
        axs[1].set_ylabel('Compensated Intensity')


        axs[0].set_xticks([])
        if dataTitle is not None:
            axs[0].set_title('Cheek and Nose Signal Intensity: ' + dataTitle)
        else:
            axs[0].set_title('Cheek and Nose Signal Intensity')

        #plt.show()

        return HRsig, HRsigRaw, I_comp, Depth, I_raw
    
    def depthComp(self, I_raw, Depth, timeWindow, Fs):
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
                for bi in np.arange(0.2, 5.01, 0.01):
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

    def motionComp(self, HRsig, depth):
        """
        motionComp caculates and outputs a plot of heart rate found w and w/o motion comp.

        Args:
            HRsig: 1D array of Cheek and Nose compensated intensity
            depth: 2D array of raw depth

        Returns:
            specW (1D array of ints): Heart rate (BPM) with motion comp. 
            HR (int): Heart rate with motion compensation for the last 20 second clip
            specND (1D array of ints): Heart rate (BPM) with motion comp.
            HR_ND (int): Heart rate without motion compensation for the last 20 second clip
            
        """

        specW = np.zeros(151)
        specND = np.zeros(151)
        i = 0

        # Loops through 10 second increments of 20 second clip to FFT HRsig spectrum
        while i <= 300:
            # FFT for spectrum, convert to one sided
            spec = np.fft.fft(HRsig[i:(i+300)])
            P2 = np.abs(spec/300)
            P1 = P2[0:150+1]
            P1[1:-1] = 2*P1[1:-1]

            # Find top two peaks
            peak_indices, _ = find_peaks(P1)
            peaks = sorted(P1[peak_indices])
            peaks = peaks[::-1]
            A1, A2 = peaks[0], peaks[1]

            # Calculate motion score
            score = (np.var(depth[i:i+300]) - np.mean(depth[i:i+300])) / (A1 / A2)
            score = np.abs(score)

            # Calculate final spectrum
            specW += P1 / score
            specND += P1
            i += 30

        # Calculate final heart rate
        f = 30 * np.arange(300/2+1) / 300 * 60
        f_Filtered_range = np.logical_or(f < 40, f > 150)
        specW[f_Filtered_range] = 0

        peak_locs, _ = find_peaks(specW)
        peaks = specW[peak_locs]
        peaks = peaks.tolist()
        maxIndex = peaks.index(max(peaks))
        HR = f[peak_locs[maxIndex]]

        # Calculate final heart rate (No motion comp)
        specND[f_Filtered_range] = 0

        peak_locs, _ = find_peaks(specND)
        peaks = specND[peak_locs]
        peaks = peaks.tolist()
        maxIndex = peaks.index(max(peaks))
        HR_ND = f[peak_locs[maxIndex]]

        # Plot and display results
        fig, ax1 = plt.subplots()
        plt.xlabel('Heart Rate (BPM)')
        plt.title('Heart Rate Frequency Spectrum')

        ax1.set_ylabel('Spectrum (W/ MC)')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Spectrum (W/O MC)')

        ax1.plot(f, specW, label='W/ MC', color='blue')
        ax2.plot(f, specND, label='W/O MC', color='orange')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        plt.xlim([40,150])
        ##plt.show()

        print('Heart rate (With Motion Comp):', HR)
        #print('Heart rate (W/O Motion Comp):', HR_ND)

        return specW, HR, specND, HR_ND

    def getHR(self, HRsig, L, trial=None): 
        ###  NEEDS FIXING ###

        # Prepare Parameters
        Fs = 30
        T = 1/Fs

        # Get HR
        spectrum = abs(fft(HRsig))
        spectrum = 2.0 / L * spectrum[:L // 2]
        f = np.linspace(0.0, 1.0 / (2.0 * T), L // 2) * 60
        f_Filtered_range = np.logical_or(f < 40, f > 150)
        spectrum[f_Filtered_range] = 0

        # HR peak locate
        pks, properties = find_peaks(spectrum.squeeze())
        maxindex = np.argmax(spectrum[pks])
        HR = f[pks[maxindex]]

        plt.figure()
        plt.plot(f, spectrum)
        plt.xlim((40, 150))
        
        return HR
    
    def smooth(self, a, span):
        '''
        smooth is a function to smooth data. This smooth function equivalent to Matlab's

        Args:
            a: NumPy 1-D array containing the data to be smoothed
            span: smoothing window size needs, which must be odd number
        
        Returns:
            1D array of smoothed data of a

        '''
        
        out0 = np.convolve(a,np.ones(span,dtype=int),'valid')/span    
        r = np.arange(1,span-1,2)
        start = np.cumsum(a[:span-1])[::2]/r
        stop = (np.cumsum(a[:-span:-1])[::2]/r)[::-1]
        return np.concatenate((  start , out0, stop  ))
    

    ## FUNCTIONS FOR TABLET CODE PROCESSING ##

    def tablet_depthComp(self, intensities, depths, b_values=np.arange(0.2, 5.1, 0.1), a_values=[1.0], sub_clip_length=2, frames_per_second=30):
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
    

    def tablet_getHR(self, intensity_signals_compensated,num_frames):

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

        return HR_with_depth_comp