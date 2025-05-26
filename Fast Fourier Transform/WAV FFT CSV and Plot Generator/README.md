# 🔊 WAV FFT CSV and Plot Generator

This Python script reads a **WAV audio file**, computes its **frequency spectrum using FFT**, and outputs both:
- A CSV file containing **Frequency vs Magnitude**
- A plot image for **visual analysis** of the frequency content

---

## 📁 Project Structure
├── mosi_recording.wav # Input WAV file
├── fft_results.csv # Output CSV file
├── fft_plot.png # Output plot (saved image)
├── fft_to_csv_and_plot.py # This script
└── README.md # Documentation


---

## 📊 Features

- Converts stereo audio to mono if needed.
- Computes real FFT (rFFT) for performance.
- Saves results to:
  - A CSV file (`Frequency`, `Magnitude`)
  - A PNG image file (frequency vs magnitude plot)
- Helpful for frequency analysis of signals (voice, environment, machines, etc.)

---

## 📦 Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `csv` (built-in)

You can install missing packages via pip:

```bash
pip install numpy scipy matplotlib

▶️ Usage
Replace mosi_recording.wav with your desired input WAV file.

Run the script:

bash
Copy
Edit
python fft_to_csv_and_plot.py
It will generate:

fft_results.csv — Frequency vs Magnitude table

fft_plot.png — Frequency spectrum plot

📈 Output Example
CSV:

python-repl
Copy
Edit
Frequency,Magnitude
0.0,1.42
43.0,1.98
...
Plot:
A line graph showing how magnitude varies with frequency, saved as fft_plot.png.

📌 Notes
Handles mono and stereo input.

Uses np.fft.rfft for efficiency (real-valued FFT).

Plot size: 10 × 6 inches with labeled axes and grid.

🧠 Applications
Audio/speech signal analysis

Environmental sound monitoring

Vibration or fault detection

Preprocessing for ML models
