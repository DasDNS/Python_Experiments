# 🎧 WAV FFT Analyzer (16-bit PCM)

This Python script reads a **16-bit WAV file**, performs a **Fast Fourier Transform (FFT)** on the audio data, and saves the frequency domain representation to a text file for further analysis or visualization.

---

## 📁 Project Structure
.
├── 16bit_audio_input.wav # Input audio file (16-bit WAV)
├── fft_output.txt # Output text file (frequency + FFT values)
├── wav_fft_analyzer.py # This script
└── README.md # Project documentation


---

## 📊 Features

- Accepts **mono or stereo 16-bit WAV** files.
- Computes the FFT using NumPy.
- Outputs:
  - Frequency (Hz)
  - Real part of FFT value
  - Imaginary part of FFT value
- Only positive frequency components are saved (due to symmetry in FFT).

---

## 📦 Requirements

- Python 3.x
- `numpy` (for numerical operations)

---

## ▶️ Usage

1. Replace `16bit_audio_input.wav` with your own **16-bit WAV file**.
2. Run the script:

```bash
python wav_fft_analyzer.py

0.0	123.0	0.0
43.0	110.2	-35.7
...

Each line includes:

Frequency in Hz

Real part of FFT value

Imaginary part of FFT value

📌 Notes
The script checks for 16-bit audio only.

FFT length equals the number of samples.

Stereo files are handled as interleaved; no channel separation.

⚙️ Applications
Audio signal analysis

Frequency domain filtering

Bioacoustic or environmental monitoring

Machine Learning preprocessing
