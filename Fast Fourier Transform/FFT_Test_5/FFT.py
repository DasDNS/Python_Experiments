import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Set your folder path
folder_path = '.'

# Output folder for images
output_path = os.path.join(folder_path, "fft_images")
os.makedirs(output_path, exist_ok=True)

# Loop through all .wav files
for file in os.listdir(folder_path):
    if file.endswith(".wav"):
        file_path = os.path.join(folder_path, file)
        print(f"Processing: {file}")

        # Load audio
        y, sr = librosa.load(file_path, sr=None)

        # FFT
        fft_vals = np.abs(np.fft.rfft(y))
        fft_vals = 1000 * fft_vals / np.max(fft_vals)  # Normalize to 0–1000

        fft_freqs = np.fft.rfftfreq(len(y), 1/sr)

        # Plot FFT
        plt.figure(figsize=(10, 4))
        plt.plot(fft_freqs, fft_vals)
        plt.title(f'FFT Spectrum - {file}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 1000)  # Adjust as needed for bee sounds
        plt.grid()
        
        # Save plot
        image_filename = os.path.splitext(file)[0] + "_fft.png"
        image_path = os.path.join(output_path, image_filename)
        plt.savefig(image_path)
        plt.close()

print("✅ All FFT plots saved in:", output_path)