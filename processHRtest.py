import scipy.io
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks


##MOTION COMP
## RETURN heart rate scalar
## HRsig(compensated intensity for cheek and nose ROI), depth, time window
## depthComp returns 2D array
## I_comp
## use stft
def __init__(input_dir):
    # Directory where input files are located (likely ./skvs/mat/)
    input_dir = input_dir


def motionComp(HRsig, depth):
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
    f = 30 * (np.arange(300/2+1) / 300) * 60
    f_Filtered_range = np.logical_or(f < 40, f > 150)
    specW[f_Filtered_range] = 0


    peak_locs, _ = find_peaks(specW)
    peaks = np.array(peak_locs, specW[peak_locs]).T
    maxindex = np.argmax(peaks)
    HR = f[loc[maxindex]]


    # Calculate final heart rate (No motion comp)
    specND[f_Filtered_range] = 0


    peaks, loc = findpeaks(specND)
    maxindex = np.argmax(peaks)
    HR_ND = f[loc[maxindex]]


    # Plot and display results
    plt.figure()
    plt.xlabel('Heart Rate (BPM)')
    plt.title('Heart Rate Frequency Spectrum')
    plt.xlim([40, 150])


    plt.plot(f, specW, label='W/ MC')
    plt.plot(f, specND, label='W/O MC')
    plt.legend()


    plt.show()


    print('Heart rate (With Motion Comp):', HR)
    print('Heart rate (W/O Motion Comp):', HR_ND)


    return specW, HR, specND, HR_ND


def depthComp(I_raw, Depth, timeWindow, Fs):
    # Make matrix for final output
    comp = np.ones_like(I_raw)


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


    '''
    def getHR( HRsig, L):
        Fs = 30
        step = 30
        step_t = step / Fs
        L_t = L / Fs
        HR = []
        j = 1
        counter = 1


        while j + L - 1 < HRsig.shape[0]:
            spectrum = np.fft.fft(HRsig[j:j + L - 1])
            P2 = abs(spectrum / L)
            onesided = P2[:(L // 2) + 1]
            onesided[1:-1] = 2 * onesided[1:-1]
            f = Fs * (np.arange(0, L // 2 + 1) / L) * 60
            f_Filtered_range = (f < 40) | (f > 150)
            onesided[f_Filtered_range] = 0


            pks, loc = findpeaks(onesided)
            maxindex = np.argmax(pks)
            HR_current = f[loc[maxindex]]
            HR.append(HR_current)


            j = j + step
            counter = counter + 1


        t_HR = np.arange(L_t / 2, ((len(HR) - 1) * step_t + L_t / 2), step_t)
        return t_HR, HR
    '''
   
def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def findpeaks(x):
    dx = np.diff(x)
    peak_locs = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    peak_vals = x[peak_locs]
    return peak_vals, peak_locs


def processRawData( filename, dataTitle=None):
    data = scipy.io.loadmat(filename)
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
    I_comp = depthComp(I_raw, Depth, 60, 30)


    # Process waveforms into the different regions
    Fs = 30
    T = 1 / Fs


    HRsig = I_comp[2, :]
    HRsigRaw = I_raw[2, :]

    '''
    bc_nose = savgol_filter(-np.log(I_comp[0, :]), 20, 3)
    bc_forehead = savgol_filter(-np.log(I_comp[1, :]), 20, 3)
    bc_lc = savgol_filter(-np.log(I_comp[3, :]), 20, 3)
    bc_rc = savgol_filter(-np.log(I_comp[4, :]), 20, 3)
    '''
    bc_nose = smooth(-np.log(I_comp[0, :]), 19)
    bc_forehead = smooth(-np.log(I_comp[1, :]), 19)
    bc_lc = smooth(-np.log(I_comp[3, :]), 19)
    bc_rc = smooth(-np.log(I_comp[4, :]), 19)

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


    plt.show()


    return tb, bc, HRsig, HRsigRaw, I_comp, Depth
       


tb, bc, HRsig, HRsigRaw, I_comp, Depth = processRawData('lauren_5-23.mat')




# Plot smoothed blood concentration
plt.figure()
'''
plt.plot(tb, savgol_filter(bc[1], 50, 3), color = 'blue')
plt.plot(tb, savgol_filter(bc[0], 50, 3), color = 'red')
plt.plot(tb, savgol_filter((bc[2]+bc[3]/2), 50, 3), color = 'orange')
'''
#hi = bc[2]+bc[3]/2

plt.plot(tb, smooth(bc[1], 51), color = 'blue')
plt.plot(tb, smooth(bc[0], 51), color = 'red')
plt.plot(tb, smooth(((bc[2]+bc[3])/2), 51), color = 'orange')

plt.xlabel('Time (s)')
plt.legend(['Nose', 'Forehead', 'Cheek Average'])
plt.ylabel('Relative Blood Concentration Change (a.u.)')
plt.show()


# Calculate Heart Rate (Motion Score)
motionComp(HRsig[:600], Depth[2, :600])
motionComp(HRsig[600:1200], Depth[2, 600:1200])
motionComp(HRsig[1200:1800], Depth[2, 1200:1800])



