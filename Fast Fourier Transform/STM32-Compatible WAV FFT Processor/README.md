# 🎧 STM32-Compatible WAV FFT Processor

This script processes a 16-bit mono PCM `.wav` file using **real FFT** (`rfft`) and outputs the **magnitude spectrum** to a `.txt` file, specifically formatted for embedded systems (e.g., **STM32**) that perform FFT analysis using fixed-size buffers.

---

## 📂 Project Structure
.
├── queen_present.wav # Input WAV file
├── fft_magnitude.txt # Output magnitudes for use with STM32
├── process_wav_fft.py # This script
└── README.md # Documentation


---

## 🎯 Purpose

This tool is designed for **preprocessing audio samples** that will be analyzed on embedded systems, particularly **STM32 MCUs** using CMSIS-DSP libraries or equivalent FFT logic.

---

## ⚙️ Features

- ✅ Supports **16-bit mono** PCM WAV files
- ✅ Applies **1024-point FFT** (adjustable)
- ✅ Outputs **raw magnitudes only** (ideal for ML or embedded comparison)
- ✅ Output matches STM32-style buffer logic (e.g., for inference or visualization)

---

## 🧰 Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `wave` (built-in)

Install NumPy if not available:

```bash
pip install numpy

🚀 How to Use
Place your 16-bit mono WAV file in the directory.

Rename it to queen_present.wav (or edit WAV_FILE in the script).

Run the script:

bash
Copy
Edit
python process_wav_fft.py
It will create a file named fft_magnitude.txt with one magnitude per line.

🧪 Example Output
txt
Copy
Edit
1137.0
483.2
32.8
...
Each value corresponds to the magnitude of a frequency bin in the FFT result (computed from rfft).

🐝 Use Case
This script is ideal for:

Benchmarking FFT output on STM32 against Python results

Feeding real audio data into ML models on microcontrollers

Analyzing the presence of queen bee signals in a known frequency band (e.g., 200–550 Hz)

🔧 Customization
Change the FFT size to match your embedded code:

python
Copy
Edit
FFT_SIZE = 1024  # Options: 512, 1024, 2048, etc.
Change the file paths if needed:

python
Copy
Edit
WAV_FILE = 'your_file.wav'
OUTPUT_FILE = 'fft_output.txt'

