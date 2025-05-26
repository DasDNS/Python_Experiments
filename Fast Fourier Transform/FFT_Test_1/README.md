# Audio FFT Magnitude Extractor

This Python script reads a WAV audio file, performs a Fast Fourier Transform (FFT) to convert the audio signal into the frequency domain, and saves the frequency magnitude values to a text file.

---

## Features

- Supports both mono and stereo WAV files (automatically uses the first channel if stereo).
- Automatically detects and prints sample rate, bit depth, and number of channels.
- Normalizes audio data before FFT.
- Outputs frequency-magnitude pairs for positive frequencies.
- Clear info logging throughout the process.

---

## Requirements

- Python 3.x
- NumPy
- SciPy

Install dependencies with:

```bash
pip install numpy scipy

Usage
Place your WAV file in the same directory or specify the full path.

Modify the input_wav variable in the script with your WAV file path.

Run the script:

bash
Copy
Edit
python fft_audio_processor.py
The frequency and magnitude data will be saved as a CSV-like .txt file, specified by output_txt in the script.

How It Works
Loads WAV file and extracts audio data.

If stereo, selects the first channel.

Normalizes audio data to the range [-1, 1].

Performs FFT to convert time-domain audio data to frequency domain.

Saves the positive frequency bins and their magnitudes to a text file.

Example Output Snippet
python-repl
Copy
Edit
Frequency (Hz), Magnitude
0.0, 12345.67
34.33, 4567.89
...
Notes
The script processes the entire audio file at once; large files may require significant memory.

Frequencies are in Hertz (Hz).

Magnitude corresponds to the FFT amplitude at each frequency bin.
