# 🐝 Queen Frequency Detection & FFT Visualizer

This Python script performs a **Fast Fourier Transform (FFT)** on a given WAV audio file, detects energy in the **queen bee frequency range**, and outputs:

- 📄 A CSV with `Frequency` vs `Magnitude`
- 📈 A PNG plot with the queen frequency range **highlighted in red**

---

## 📂 Project Files
.
├── 2025-01-30_14-51-43.wav # Input WAV file
├── fft_results.csv # Output CSV file with frequency data
├── fft_plot.png # Output plot highlighting queen range
├── fft_with_queen_range.py # This script
└── README.md # Documentation


---

## 🎯 Purpose

This tool helps visualize and analyze frequency activity in recorded hive audio, particularly targeting the **queen bee signal range** (default: 200–550 Hz).

---

## 🧰 Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `csv` (built-in)

Install any missing libraries via pip:

```bash
pip install numpy scipy matplotlib

🚀 How to Use
Replace 2025-01-30_14-51-43.wav with your target WAV file.

Run the script:

bash
Copy
Edit
python fft_with_queen_range.py
The script outputs:

fft_results.csv: Frequency vs Magnitude values

fft_plot.png: Visual FFT plot with queen range highlighted

📌 Features
✅ Supports stereo and mono WAV input

✅ Applies real FFT (rfft) for optimized performance

✅ Saves clean frequency plots with grid and labels

✅ Highlights queen frequency range (customizable)

🐝 Queen Detection Logic
By default, the queen frequency range is set to:

python
Copy
Edit
queen_freq_range = (200, 550)  # Hz
You can modify this in the function call to match your specific detection criteria.

📊 Output Example
CSV Preview:

python-repl
Copy
Edit
Frequency,Magnitude
0.0,2.43
43.1,4.28
...
Plot Preview:

A line plot showing all frequencies, with the 200–550 Hz region shaded in red to indicate the potential queen signature zone.

📈 Applications
Bee colony health and queen presence detection

Bioacoustic analysis

Research in precision apiculture

ML dataset preparation (frequency-domain features)


