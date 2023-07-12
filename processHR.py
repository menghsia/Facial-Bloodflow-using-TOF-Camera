import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time


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
        HRsig, HRsigRaw, I_comp, Depth = self.processRawData()

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

        print(HR_comp)
        print(HR_ND)

        # Calculate Heart Rate (Motion Score)
        #self.motionComp(HRsig, Depth)
        end_HRtime = time.time()
        self.time = end_HRtime - start_HRtime
        plt.show()
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

        # Compensate for movement
        # I_comp: 2D array of compensated intensities
        I_comp = self.depthComp(I_raw, Depth, 2, 30)
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

        return HRsig, HRsigRaw, I_comp, Depth
    
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
                cor = 1

                # For each clip, iterate through different b values with a = 1
                for bi in np.arange(0.2, 5.1, 0.1):
                    bI_comp = I_raw[ROI, ((i - 1) * (timeWindow * Fs)) : ((i * (timeWindow * Fs)) - 1)] / ((Depth[ROI, ((i - 1) * (timeWindow * Fs)) : ((i * (timeWindow * Fs)) - 1)]) ** (-bi))
                    # Find correlation between bI_comp and Depth
                    corr_v = np.corrcoef(bI_comp, Depth[ROI, ((i - 1) * (timeWindow * Fs)) : ((i * (timeWindow * Fs)) - 1)])
                    # Take absolute value of correlation coefficients
                    corr_ = abs(corr_v[1, 0])

                    # If the new correlation coeff is less than the old one, reset cor value and best I_comp
                    if corr_ < cor:
                        cor = corr_
                        best = bI_comp

                # Normalize data using z-scores
                I_comp_ROI[((i - 1) * (timeWindow * Fs)) : ((i * (timeWindow * Fs)) - 1)] = (best - np.mean(best))/np.std(best)
                i += 1

            # For the remainder of the clip if it is 
            cor = 1
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
        spectrum = abs(np.fft.fft(HRsig))
        spectrum = 2.0 / L * spectrum[:L // 2]
        f = np.linspace(0.0, 1.0 / (2.0 * T), L // 2) * 60
        f_Filtered_range = np.logical_or(f < 40, f > 150)
        spectrum[f_Filtered_range] = 0

        # HR peak locate
        pks, properties = find_peaks(spectrum.squeeze())
        maxindex = np.argmax(spectrum[pks])
        HR = f[pks[maxindex]]

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
