import scipy.io
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import time


class ProcessHR():
    """
    ProcessHR is a class that uses depths and intensities to output a calculated heartrate.

    Args:
        input_file (str): filename from the correct directory where while is located

    Attributes:
        input_file (str): input filename in which raw depths and intensities are stored
    """

    def __init__(self, input_file):
        """
        Initializes class variables

        Args:
            input_file (str): filename from the correct directory where while is located
        """  

        self.input_file = input_file
    
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
        tb, bc, HRsig, HRsigRaw, I_comp, Depth = self.processRawData()

        # Plot smoothed blood concentration at times(s)
        plt.figure()

        plt.plot(tb, self.smooth(bc[1], 51), color = 'blue')
        plt.plot(tb, self.smooth(bc[0], 51), color = 'red')
        plt.plot(tb, self.smooth(((bc[2]+bc[3])/2), 51), color = 'orange')

        plt.xlabel('Time (s)')
        plt.legend(['Nose', 'Forehead', 'Cheek Average'])
        plt.ylabel('Relative Blood Concentration Change (a.u.)')
        

        ''' ## getHR() NEEDS FIXING ##
        # Get HR Data
        t_HR_comp, HR_comp = getHR(HRsig, 900)

        # Plot HR Data
        plt.figure()
        plt.plot(t_HR_comp, HR_comp, label='comp')
        plt.title('Heart Rate')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Heart Rate (bpm)')

        t_HR_raw, HR_raw = getHR(HRsigRaw, 900)
        plt.plot(t_HR_raw, HR_raw, '--', label='raw')

        plt.legend()
        plt.show()
        '''

        # Calculate Heart Rate (Motion Score)
        self.motionComp(HRsig[:600], Depth[2, :600])
        self.motionComp(HRsig[600:1200], Depth[2, 600:1200])
        self.motionComp(HRsig[1200:1800], Depth[2, 1200:1800])
        plt.show()
        return 

    def processRawData(self, dataTitle=None):
        
        # FUNCTION: Extracts raw data from .mat file to 
        #           plot raw and compensated intensities at a ROI

        # Load in raw depth and intensities
        data = scipy.io.loadmat(self.input_file)
        Depth = data['Depth']
        I_raw = data['I_raw']

        # Remove extraneous zeros
        for i in range(Depth.shape[1]-1, -1, -1):
            if Depth[0, i] == 0:
                Depth = np.delete(Depth, i, axis=1)
                I_raw = np.delete(I_raw, i, axis=1)
            else:
                break

        # Compensate for movement
        I_comp = self.depthComp(I_raw, Depth, 60, 30)

        # Process waveforms into the different regions
        Fs = 30 # Frames/Second
        T = 1 / Fs

        HRsig = I_comp[2, :] 
        HRsigRaw = I_raw[2, :]

        bc_nose = self.smooth(-np.log(I_comp[0, :]), 19)
        bc_forehead = self.smooth(-np.log(I_comp[1, :]), 19)
        bc_lc = self.smooth(-np.log(I_comp[3, :]), 19)
        bc_rc = self.smooth(-np.log(I_comp[4, :]), 19)

        bc = np.vstack((bc_forehead, bc_nose, bc_lc, bc_rc))
        tb = np.arange(0, I_raw.shape[1]) * T


        # Plot Raw and Compensated Data
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].plot(I_raw[0, :])
        axs[0].set_ylabel('Raw Intensity')


        axs[1].plot(I_comp[0, :])
        axs[1].set_ylabel('Compensated Intensity')


        axs[0].set_xticks([])
        if dataTitle is not None:
            axs[0].set_title('Forehead Signal Intensity: ' + dataTitle)
        else:
            axs[0].set_title('Forehead Signal Intensity')

        #plt.show()

        return tb, bc, HRsig, HRsigRaw, I_comp, Depth
    
    def depthComp(self, I_raw, Depth, timeWindow, Fs):
        # FUNCTION: Returns Compensated Intensity 

        # Make matrix for final output
        comp = np.ones_like(I_raw)

        best = 1
        best_comp = 1

        # Iterate through the different ROIs
        for j in range(I_raw.shape[0]):
            # Make variables for I_comp (to be appended) and i to iterate through
            compj = np.ones(I_raw.shape[1])
            i = 1

            # Iterate through every clip...so every 60 frames
            while (i * (timeWindow * Fs)) < len(I_raw[j, :]):
                cor = 1

                # For each clip, iterate through different b values with a = 1
                for bi in np.arange(0.2, 5, 0.01):
                    bI_comp = I_raw[j, ((i - 1) * (timeWindow * Fs)) : (i * (timeWindow * Fs))] / (Depth[j, ((i - 1) * (timeWindow * Fs)) : (i * (timeWindow * Fs))] ** (-bi))
                    # Find correlation between bI_comp and Depth
                    corr_v = np.corrcoef(bI_comp, Depth[j, ((i - 1) * (timeWindow * Fs)) : (i * (timeWindow * Fs))])
                    # Take absolute value of correlation coefficients
                    corr_ = np.abs(corr_v[0, 1])

                    # If the new correlation coeff is less than the old one, reset cor value and I_comp
                    if corr_ < cor:
                        cor = corr_
                        best = bI_comp

                # Normalize data
                compj[((i - 1) * (timeWindow * Fs)) : (i * (timeWindow * Fs))] = best / np.mean(best)
                i += 1

            # For the remainder
            cor = 1
            for bii in np.arange(0.1, 5, 0.01):
                bI_comp = I_raw[j, (((i - 1) * (timeWindow * Fs))) :] / (Depth[j, (((i - 1) * (timeWindow * Fs))) :] ** (-bii))
                # Find correlation between bI_comp and Depth
                corr_v = np.corrcoef(bI_comp, Depth[j, (((i - 1) * (timeWindow * Fs)) ) :])
                # Take absolute value of correlation coefficients
                corr_ = np.abs(corr_v[0, 1])

                # If the new correlation coeff is less than the old one, reset cor value and I_comp
                if corr_ < cor:
                    cor = corr_
                    best_comp = bI_comp

            # Normalize data
            compj[(((i - 1) * (timeWindow * Fs))) :] = best_comp / np.mean(best_comp)
            # Append to final output matrix
            comp[j, :] = compj


        return comp

    def motionComp(self, HRsig, depth):
        # FUNCTION: Finds heart rate using motion score 
        #           ie. with regard to motion compensation
        # Outputs a plot of heart rate found w and w/o motion comp.

        specW = np.zeros(151)
        specND = np.zeros(151)
        i = 0

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
        print('Heart rate (W/O Motion Comp):', HR_ND)

        return specW, HR, specND, HR_ND

    def getHR(self, HRsig, L, trial=None): 
        ###  NEEDS FIXING ###

        # Prepare Parameters
        Fs = 30
        step = 30
        step_t = step / Fs
        L_t = L / Fs
        HR = []

        # Get HR
        j = 0
        counter = 1
        while j + L - 1 < HRsig.shape[0]:
            spectrum = np.fft.fft(HRsig[j:(j+L)])
            P2 = np.abs(spectrum / L)
            onesided = P2[1:(L//2) + 1]
            onesided[1:-1] = 2*onesided[1:-1]
            f = Fs * np.arange((L//2) + 1) / L * 60
            f_Filtered_range = np.logical_or(f < 40, f > 150)
            onesided[f_Filtered_range] = 0

            # HR peak locate
            pks, loc = find_peaks(onesided.squeeze())
            maxindex = np.argmax(pks)
            HR_current = f[loc[maxindex]]
            HR.append(HR_current)

            j = j + step
            counter = counter + 1

        t_HR = np.arange(L_t / 2, len(HR) * step_t + L_t / 2, step_t)
        return t_HR, HR
    
    def smooth(self, a, span):
        # FUNCTION: smooth function equivalent to matlab 
        # a: NumPy 1-D array containing the data to be smoothed
        # span: smoothing window size needs, which must be odd number
        out0 = np.convolve(a,np.ones(span,dtype=int),'valid')/span    
        r = np.arange(1,span-1,2)
        start = np.cumsum(a[:span-1])[::2]/r
        stop = (np.cumsum(a[:-span:-1])[::2]/r)[::-1]
        return np.concatenate((  start , out0, stop  ))
