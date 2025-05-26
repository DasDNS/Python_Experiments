import numpy as np
import scipy.io.wavfile as wav
import csv
import matplotlib.pyplot as plt  # For plotting

def compute_fft_and_save_to_csv_and_plot(wav_file, output_csv, output_image, queen_freq_range=(200, 550)):
    # Read the WAV file
    rate, data = wav.read(wav_file)
    
    # If the audio is stereo (2 channels), convert to mono by averaging the channels
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Apply FFT
    fft_data = np.fft.rfft(data)  # Fast Fourier Transform
    freqs = np.fft.rfftfreq(len(data), 1 / rate)  # Frequencies corresponding to the FFT

    # Open the CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(['Frequency', 'Magnitude'])
        
        # Write the frequency and magnitude values
        for f, mag in zip(freqs, np.abs(fft_data)):
            writer.writerow([f, mag])
    
    print(f"FFT results saved to {output_csv}")

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, np.abs(fft_data), label="FFT Magnitude")
    plt.title('Frequency vs Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Highlight the queen frequency range
    plt.axvspan(queen_freq_range[0], queen_freq_range[1], color='red', alpha=0.3, label=f'Queen Range {queen_freq_range[0]}-{queen_freq_range[1]} Hz')

    # Add legend
    plt.legend()

    # Save the plot as an image file
    plt.savefig(output_image)
    print(f"Graph saved as {output_image}")

# Example Usage
input_wav = '2025-01-30_14-51-43.wav'  # Replace with your input WAV file path
output_csv = 'fft_results.csv'  # Output CSV file
output_image = 'fft_plot.png'  # Output image file (graph)
compute_fft_and_save_to_csv_and_plot(input_wav, output_csv, output_image)
