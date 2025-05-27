# FFT Plot Generator for WAV Files

This Python script processes all `.wav` audio files in the current directory and generates FFT (Fast Fourier Transform) spectrum plots. The plots are saved as PNG images in a subfolder named `fft_images`.

## Features

- Automatically detects and processes `.wav` files in the working directory.
- Computes the FFT using `numpy` and `librosa`.
- Normalizes the FFT amplitude for better visualization.
- Saves each FFT spectrum as a `.png` image.
- Uses `matplotlib` for plotting.
- Plots are focused on the frequency range 0–1000 Hz, ideal for bee or low-frequency analysis.

## Requirements

Install the required Python packages using pip:

```bash
pip install numpy matplotlib librosa

How to Use
Place your .wav audio files in the same folder as the script.

Run the script:

bash
Copy
Edit
python fft_plot_generator.py
The script will:

Create a folder called fft_images (if not already present).

Generate FFT plots for each .wav file.

Save the images inside fft_images.

Output
For each .wav file, a corresponding FFT image is saved as:

Copy
Edit
filename_fft.png
Example:

bash
Copy
Edit
bee_sound.wav → fft_images/bee_sound_fft.png
Customization
You can modify the script to:

Change the frequency display range (currently 0–1000 Hz).

Adjust amplitude scaling.

Support stereo files or other sample rates by tweaking librosa.load() parameters.
