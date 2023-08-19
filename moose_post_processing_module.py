import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, welch, cwt, morlet, butter, filtfilt
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

    def calculate_HR(self, intensity_compensated, plot=False, sampling_rate=30):
        # Only consider BPMs in the range of 30-250 BPM by applying a bandpass filter
        # Convert the valid BPM range to frequency (Hz)
        min_valid_freq = 30 / 60  # 30 BPM in Hz
        max_valid_freq = 250 / 60  # 250 BPM in Hz
        intensity_compensated = self.bandpass_filter(intensity_compensated, min_valid_freq, max_valid_freq, sampling_rate)

        # Apply wavelet denoising
        intensity_compensated = self.wavelet_denoising(intensity_compensated)

        # Detrend the data
        signal = intensity_compensated - np.poly1d(np.polyfit(np.linspace(0, len(intensity_compensated), len(intensity_compensated)), intensity_compensated, 1))(np.linspace(0, len(intensity_compensated), len(intensity_compensated)))
        
        # Apply a window function
        window = np.hamming(signal.size)
        signal_windowed = signal * window

        # Compute the FFT
        fft_vals = np.fft.rfft(signal_windowed)
        fft_freqs = np.fft.rfftfreq(signal.size, d=1/30)  # Assuming constant sample interval

        # 1% error
        # # Find the dominant frequency
        # dominant_frequency = fft_freqs[np.argmax(np.abs(fft_vals))]

        # 0.07% error
        # Find the dominant frequency bin
        peak_index = np.argmax(np.abs(fft_vals))
        # Interpolate to find a more accurate frequency
        dominant_frequency = self.interpolate_peak(fft_freqs, np.abs(fft_vals), peak_index)
        
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
        
        return calculated_bpm, dominant_frequency


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
    calculated_bpm, dominant_frequency = analyzer.calculate_HR(intensity_compensated, plot=plot, sampling_rate=sampling_rate)

    # Calculate % error BPM
    percent_error_bpm = np.abs(actual_bpm - calculated_bpm) / actual_bpm * 100

    print(f"Dominant Frequency: {dominant_frequency:.2f} Hz (Actual: {freq:.2f} Hz)")
    print(f"Equivalent Heart Rate: {calculated_bpm:.2f} bpm (Actual: {actual_bpm:.2f} bpm)")
    print(f"Percent Error BPM: {percent_error_bpm:.2f}%")


if __name__ == '__main__':
    np.random.seed(42)  # Set the random seed for reproducibility

    # 0.07% error
    print("Test 1")
    test_HR(actual_bpm=100, noise_modifier=0.2, plot=False)

    # 0.02% error
    print("Test 2")
    test_HR(actual_bpm=100, noise_modifier=0.3, plot=False)

    # 0.07% error
    print("Test 3")
    test_HR(actual_bpm=100, noise_modifier=0.5, plot=False)

    # 0.06% error
    print("Test 4")
    test_HR(actual_bpm=100, noise_modifier=4, plot=False)

    # 0.98% error
    print("Test 5")
    test_HR(actual_bpm=100, noise_modifier=5, plot=False)

    # 14.16% error
    print("Test 6")
    test_HR(actual_bpm=100, noise_modifier=6, plot=False)

    # 16.41% error
    print("Test 7")
    test_HR(actual_bpm=100, noise_modifier=10, plot=True)
