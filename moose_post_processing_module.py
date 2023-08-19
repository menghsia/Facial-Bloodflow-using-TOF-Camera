import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, welch, cwt, morlet, butter, filtfilt, find_peaks
import pywt

class HeartRateAnalyzer:
    def interpolate_peak(self, freqs, magnitudes, peak_index):
        if peak_index == 0 or peak_index == len(freqs) - 1:
            return freqs[peak_index]

        # Quadratic interpolation
        alpha = magnitudes[peak_index - 1]
        beta = magnitudes[peak_index]
        gamma = magnitudes[peak_index + 1]

        # Calculate the interpolated frequency bin correction
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        
        # Return the corrected frequency
        return freqs[peak_index] + p * (freqs[1] - freqs[0])

    def find_dominant_frequency(self, signal, fs):
        f, t, Zxx = stft(signal, fs=fs, nperseg=fs*4, noverlap=fs*3, nfft=fs*10)
        magnitudes = np.abs(Zxx)
        dominant_freqs = f[np.argmax(magnitudes, axis=0)]
        return np.mean(dominant_freqs)

    def find_dominant_frequency_welch(self, signal, fs):
        f, Pxx = welch(signal, fs=fs, nperseg=fs*4, noverlap=fs*3, nfft=fs*10)
        return f[np.argmax(Pxx)]

    def find_dominant_frequency_cwt(self, signal, fs):
        widths = np.arange(1, int(fs*4))
        cwt_matrix = cwt(signal, morlet, widths)

        # Sum over time and find the width that has the max energy
        energy_sum = np.sum(np.abs(cwt_matrix)**2, axis=1)
        dominant_width = widths[np.argmax(energy_sum)]
        
        # Convert width to frequency
        dominant_frequency = fs / dominant_width
        
        return dominant_frequency
    
    def wavelet_denoising(self, signal):
        # Decompose the signal into wavelet coefficients
        coeffs = pywt.wavedec(signal, 'db1', level=4)

        # Set a threshold to remove noise. Universal threshold is a common choice.
        threshold = np.median(np.abs(coeffs[-1])) / 0.6745

        # Threshold the coefficients
        coeffs_thresholded = [pywt.threshold(coeff, threshold, mode='soft') for coeff in coeffs]

        # Reconstruct the signal from the thresholded coefficients
        denoised_signal = pywt.waverec(coeffs_thresholded, 'db1')
        
        return denoised_signal

    def plot_signal_and_spectrum(self, intensity_compensated, signal, fft_freqs, fft_vals, bpm_values):
        # Single Figure with Three Subplots
        plt.figure(figsize=(12, 10))

        # Time Domain Signal
        plt.subplot(3, 1, 1)
        plt.plot(intensity_compensated, signal)
        plt.title('Intensity compensated (Time Domain Signal)')
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')

        # Frequency Spectrum
        plt.subplot(3, 1, 2)
        plt.plot(fft_freqs, np.abs(fft_vals))
        plt.title('Frequency Spectrum (Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim([0, 4.2])  # Limiting x-axis to show only frequencies up to 250bpm/60s=4.2 Hz for clarity

        # Frequency Spectrum (displaying BPM)
        plt.subplot(3, 1, 3)
        plt.plot(bpm_values, np.abs(fft_vals))
        plt.title('Frequency Spectrum (Heart Rate)')
        plt.xlabel('Heart Rate (BPM)')
        plt.ylabel('Magnitude')
        plt.xlim([0, 250])  # Limiting x-axis to show only BPM values up to 250 bpm for clarity

        plt.tight_layout()
        plt.show()

    def bandpass_filter(self, signal, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    def find_dominant_frequency_weighted(self, fft_freqs, fft_vals, num_peaks=3):
        # Finding peaks in the magnitude spectrum
        peaks, _ = find_peaks(np.abs(fft_vals), distance=1)
        
        # Sorting the peaks based on their magnitudes (descending)
        sorted_peaks = sorted(peaks, key=lambda x: np.abs(fft_vals[x]), reverse=True)
        
        # Taking only top 'num_peaks' peaks
        prominent_peaks = sorted_peaks[:num_peaks]
        
        # Calculating the weighted frequency
        total_weight = sum([np.abs(fft_vals[p]) for p in prominent_peaks])
        weighted_freqs = sum([fft_freqs[p] * np.abs(fft_vals[p]) for p in prominent_peaks])
        
        dominant_frequency_weighted = weighted_freqs / total_weight
        
        return dominant_frequency_weighted
    
    def adaptive_thresholding(self, signal, window_length=150, factor=1.5):
        """
        Apply adaptive thresholding to the signal.
        """
        # Length of the signal
        N = len(signal)

        # Output signal after thresholding
        thresholded_signal = np.zeros(N)
        
        # Calculate threshold for each window and apply it
        for start in range(0, N, window_length):
            end = min(start + window_length, N)
            window = signal[start:end]
            
            # Median-based threshold
            med = np.median(window)
            # Compute the threshold for this window
            threshold = med + factor * np.std(window)
            
            thresholded_signal[start:end] = np.where(window > threshold, window, 0)
            
        return thresholded_signal

    def calculate_HR(self, intensity_compensated, plot=False, sampling_rate=30):
        # Only consider BPMs in the range of 30-250 BPM by applying a bandpass filter
        # Convert the valid BPM range to frequency (Hz)
        min_valid_freq = 30 / 60  # 30 BPM in Hz
        max_valid_freq = 250 / 60  # 250 BPM in Hz
        intensity_compensated = self.bandpass_filter(intensity_compensated, min_valid_freq, max_valid_freq, sampling_rate, order=1)

        # Apply wavelet denoising
        intensity_compensated = self.wavelet_denoising(intensity_compensated)

        # # Apply adaptive thresholding
        # intensity_compensated = self.adaptive_thresholding(intensity_compensated)

        # Detrend the data
        signal = intensity_compensated - np.poly1d(np.polyfit(np.linspace(0, len(intensity_compensated), len(intensity_compensated)), intensity_compensated, 1))(np.linspace(0, len(intensity_compensated), len(intensity_compensated)))
        
        # Apply a window function
        window = np.hamming(signal.size)
        signal_windowed = signal * window

        # Compute the FFT
        fft_vals = np.fft.rfft(signal_windowed)
        fft_freqs = np.fft.rfftfreq(signal.size, d=1/30)  # Assuming constant sample interval

        # Zero-padding to improve FFT resolution
        # zero_padded_signal = np.pad(signal_windowed, (0, len(signal_windowed)), 'constant')

        # # Compute the FFT
        # fft_vals = np.fft.rfft(zero_padded_signal)
        # fft_freqs = np.fft.rfftfreq(len(zero_padded_signal), d=1/30)  # Assuming constant sample interval

        # 1% error
        # # Find the dominant frequency
        # dominant_frequency = fft_freqs[np.argmax(np.abs(fft_vals))]

        # 0.07% error
        # Find the dominant frequency bin
        peak_index = np.argmax(np.abs(fft_vals))
        # Interpolate to find a more accurate frequency
        dominant_frequency = self.interpolate_peak(fft_freqs, np.abs(fft_vals), peak_index)

        # # Use power spectrum for peak detection
        # power_spectrum = np.abs(fft_vals)**2
        # peak_index = np.argmax(power_spectrum)

        # # Interpolate to find a more accurate frequency
        # dominant_frequency = self.interpolate_peak(fft_freqs, power_spectrum, peak_index)

        # To compute the weighted dominant frequency
        # dominant_frequency = self.find_dominant_frequency_weighted(fft_freqs, fft_vals)
        
        # 1.71% error
        # dominant_frequency = self.find_dominant_frequency(signal, sampling_rate)

        # 2% error
        # dominant_frequency = self.find_dominant_frequency_welch(signal, sampling_rate)

        # 81.63% error
        # # Find dominant frequency using CWT
        # dominant_frequency = self.find_dominant_frequency_cwt(signal, sampling_rate)
        
        calculated_bpm = dominant_frequency * 60 # Convert frequencies to BPMs

        # Compute BPM for each frequency value
        bpm_values = fft_freqs * 60  # Convert each frequency value to bpm

        if plot:
            self.plot_signal_and_spectrum(np.linspace(0, len(intensity_compensated), len(intensity_compensated)), signal, fft_freqs, np.abs(fft_vals), bpm_values)
        
        return calculated_bpm
    
    def calculate_HR_chunks(self, intensity_compensated, chunk_duration, plot=False, sampling_rate=30):
        num_samples = len(intensity_compensated)
        chunk_samples = int(chunk_duration * sampling_rate)
        
        bpm_results = []
        for i in range(0, num_samples, chunk_samples):
            chunk = intensity_compensated[i:i + chunk_samples]
            if len(chunk) == chunk_samples:  # Exclude any leftover chunk
                bpm = self.calculate_HR(chunk, plot=plot, sampling_rate=sampling_rate)
                bpm_results.append(bpm)
        
        # Return the average BPM over all chunks
        return np.mean(bpm_results)


def test_HR(actual_bpm, noise_modifier, plot=False):
    # Parameters for signal generation
    sampling_rate = 30  # Sampling rate: 30 Hz (30 fps) (fs)
    num_seconds = 20  # Duration: 20 seconds
    num_samples = num_seconds * sampling_rate  # Total number of samples
    intensity_compensated = np.linspace(0, num_seconds, num_samples)  # Time array

    # Generate the sample signal outside the calculate_HR function
    freq = actual_bpm / 60  # Frequency equivalent to 100 bpm
    clean_signal = np.sin(2 * np.pi * freq * intensity_compensated)
    noise = noise_modifier * np.random.normal(size=intensity_compensated.size)
    intensity_compensated = clean_signal + noise

    # Calculate the BPM
    analyzer = HeartRateAnalyzer()
    calculated_bpm = analyzer.calculate_HR(intensity_compensated, plot=plot, sampling_rate=sampling_rate)
    # calculated_bpm = analyzer.calculate_HR_chunks(intensity_compensated, chunk_duration=5, plot=plot, sampling_rate=sampling_rate)

    # Calculate % error BPM
    percent_error_bpm = np.abs(actual_bpm - calculated_bpm) / actual_bpm * 100

    print(f"Equivalent Heart Rate: {calculated_bpm:.2f} bpm (Actual: {actual_bpm:.2f} bpm)")
    print(f"Percent Error BPM: {percent_error_bpm:.2f}%")


if __name__ == '__main__':
    np.random.seed(42)  # Set the random seed for reproducibility

    # 0.08% error
    print("Test 1")
    test_HR(actual_bpm=100, noise_modifier=0.2, plot=False)

    # 0.01% error
    print("Test 2")
    test_HR(actual_bpm=100, noise_modifier=0.3, plot=False)

    # 0.07% error
    print("Test 3")
    test_HR(actual_bpm=100, noise_modifier=0.5, plot=False)

    # 0.00% error
    print("Test 4")
    test_HR(actual_bpm=100, noise_modifier=4, plot=False)

    # 0.99% error
    print("Test 5")
    test_HR(actual_bpm=100, noise_modifier=5, plot=False)

    # 14.16% error
    print("Test 6")
    test_HR(actual_bpm=100, noise_modifier=6, plot=False)

    # 16.43% error
    print("Test 7")
    test_HR(actual_bpm=100, noise_modifier=10, plot=False)

    # 30.22% error
    print("Test 8")
    test_HR(actual_bpm=74, noise_modifier=8, plot=False)

    # 33.20% error
    print("Test 9")
    test_HR(actual_bpm=82, noise_modifier=8, plot=False)

    # 18.15% error
    print("Test 10")
    test_HR(actual_bpm=100, noise_modifier=8, plot=False)
