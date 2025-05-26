# WAV FFT Magnitude Extractor

This Python script reads a 16-bit mono PCM WAV audio file, applies a Hann window, computes the Fast Fourier Transform (FFT) on a fixed-size segment, and saves the magnitude of the FFT results to a text file.

---

## Features

- Supports **16-bit mono PCM WAV** files only.
- Applies a **Hann window** to reduce spectral leakage.
- Computes FFT of size **1024 samples** (configurable).
- Saves FFT magnitude values to a plain text file.
- Prints detailed info and progress in the console.

---

## Requirements

- Python 3.x
- NumPy
- (Optional) `matplotlib` for plotting if you want to extend the script

Install dependencies via pip:

```bash
pip install numpy

Usage
Place your WAV file (16-bit mono PCM) in the same directory or provide the full path.

Update the WAV_FILE variable in the script with your file name.

Run the script:

bash
Copy
Edit
python process_wav_fft.py
The FFT magnitudes will be saved in the file specified by OUTPUT_FILE (default: fft_magnitude.txt).

Configuration
WAV_FILE — Path to your input WAV file.

OUTPUT_FILE — Path to save the FFT magnitude results.

FFT_SIZE — Number of samples used for the FFT (default: 1024).

Notes
The script reads only the first FFT_SIZE samples of the audio file. If the audio is shorter than FFT_SIZE, consider padding it.

This script is designed to help analyze audio signals such as bee hive sounds and detect frequency patterns.

Example Output
kotlin
Copy
Edit
WAV file details: Channels=1, Sample Width=2, Frame Rate=44100, Frames=44100
WAV file loaded.
Saving FFT data to fft_magnitude.txt...
FFT data saved to fft_magnitude.txt.
