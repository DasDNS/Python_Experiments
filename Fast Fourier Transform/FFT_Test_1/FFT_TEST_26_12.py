import numpy as np
import scipy.io.wavfile as wav

# Printf function to describe what's happening
def printf(message):
    print(f"[INFO]: {message}")

# Load WAV file
def load_wav_file(file_path):
    try:
        printf(f"Loading WAV file: {file_path}")
        rate, data = wav.read(file_path)
        bit_depth = data.dtype.itemsize * 8  # Calculate bit depth
        num_channels = 1 if len(data.shape) == 1 else data.shape[1]
        printf(f"Loaded WAV file successfully.")
        printf(f"Sample rate: {rate} Hz")
        printf(f"Bit depth: {bit_depth}-bit")
        printf(f"Number of channels: {num_channels}")
        printf(f"Number of samples: {len(data)}")
        return rate, data
    except Exception as e:
        printf(f"Failed to load WAV file. Error: {e}")
        return None, None

# Perform FFT and calculate frequency magnitude
def perform_fft(rate, data):
    printf("Performing FFT to convert audio to frequency domain")
    # If stereo, select one channel
    if len(data.shape) > 1:
        printf("Stereo audio detected. Using the first channel.")
        data = data[:, 0]

    # Normalize data
    printf("Normalizing audio data.")
    data = data / np.max(np.abs(data))

    # Perform FFT
    fft_result = np.fft.fft(data)
    magnitude = np.abs(fft_result)  # Magnitude of FFT

    # Frequency bins
    freqs = np.fft.fftfreq(len(data), d=1/rate)
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]

    printf("FFT calculation completed.")
    return positive_freqs, positive_magnitude

# Save magnitude values to text file
def save_to_text_file(file_path, freqs, magnitude):
    try:
        printf(f"Saving frequency magnitude values to text file: {file_path}")
        with open(file_path, 'w') as file:
            for f, m in zip(freqs, magnitude):
                file.write(f"{f},{m}\n")
        printf("Frequency magnitude values saved successfully.")
    except Exception as e:
        printf(f"Failed to save to text file. Error: {e}")

# Main function
def main():
    input_wav = "bee_sound_youtube_downsampled_ICS43434_35000Hz.WAV"  # Path to input WAV file
    output_txt = "bee_sound_youtube_downsampled_ICS43434_35000Hz.txt"  # Path to save the output

    # Load audio file
    rate, data = load_wav_file(input_wav)
    if data is None:
        return

    # Perform FFT
    freqs, magnitude = perform_fft(rate, data)

    # Save output to file
    save_to_text_file(output_txt, freqs, magnitude)

if __name__ == "__main__":
    main()
