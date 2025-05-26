import wave
import numpy as np

# File paths
WAV_FILE = 'queen_present.wav'  # Input WAV file
OUTPUT_FILE = 'fft_magnitude.txt'  # Output text file with FFT magnitudes
FFT_SIZE = 1024  # FFT size, should match STM32 code

def process_wav_fft():
    try:
        # Open the WAV file
        print(f"Opening WAV file: {WAV_FILE}...")
        with wave.open(WAV_FILE, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            samp_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            print(f"WAV file details: Channels={n_channels}, Sample Width={samp_width}, Frame Rate={frame_rate}, Frames={n_frames}")

            # Ensure the file is PCM 16-bit mono
            if n_channels != 1 or samp_width != 2:
                raise ValueError("The .wav file must be 16-bit mono PCM format.")

            # Read frames from the WAV file
            wav_data = wav_file.readframes(n_frames)
            print("WAV file loaded.")

            # Convert WAV data to 16-bit PCM format (int16)
            pcm_data = np.frombuffer(wav_data, dtype=np.int16)

            # Perform FFT directly on the PCM data (without normalization)
            fft_output = np.fft.rfft(pcm_data, n=FFT_SIZE)

            # Compute magnitude of FFT output (no scaling)
            fft_magnitude = np.abs(fft_output)

            # Save FFT magnitude to a text file
            print(f"Saving FFT data to {OUTPUT_FILE}...")
            with open(OUTPUT_FILE, 'w') as f:
                for value in fft_magnitude:
                    f.write(f"{value}\n")
            print(f"FFT data saved to {OUTPUT_FILE}.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    process_wav_fft()
