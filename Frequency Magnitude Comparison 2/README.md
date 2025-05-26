# 🎵 Frequency-Magnitude Comparison with Common and Almost Equal Frequencies

This Python script compares two frequency-magnitude datasets and highlights not only the exact common frequencies but also those that are *almost equal* (within a defined threshold). It visualizes the results using `matplotlib` to show overlaps, similarities, and differences between the two signals.

---

## 📁 Files Overview

.
├── Queen_sound_youtube_downsampled_ICS43434_35000Hz.txt # ICS43434 MEMS microphone output
├── Queen_sound_youtube_downsampled.txt # Reference (e.g., YouTube audio)
├── frequency_almost_equal.py # This Python script
└── README.md # Project documentation


---

## 🔍 What It Does

- Loads two frequency-magnitude datasets (`.txt` files)
- Identifies:
  - Exact common frequencies
  - Almost equal frequencies (within a threshold of 0.25 Hz)
- Calculates and prints:
  - Total frequencies in each file
  - Number of common and unique frequencies
  - Number of almost equal frequencies
  - Total matching frequencies
- Interpolates data for comparison
- Visualizes:
  - Raw frequency-magnitude curves
  - Common frequencies (green)
  - Almost equal frequencies (yellow)
  - Overlapping magnitude regions (gray)

---

## 🧮 Example Output
Total frequencies in ICSfile: 1600
Total frequencies in File 2: 1443
Number of common frequencies: 1245
Number of unique frequencies in ICSfile: 355
Number of unique frequencies in File 2: 198
Number of almost equal frequencies: 96
Total unique frequencies: 1798
Matching frequencies (common + almost equal): 1341


---

## 📊 Graph Features

- **Blue line**: ICS43434 dataset
- **Red dashed line**: Reference dataset
- **Green dots**: Common frequencies
- **Yellow dots**: Almost equal frequencies
- **Gray area**: Overlap where magnitudes are closely matched (`< 0.1` difference)

---

## ⚙️ Requirements

Make sure to install these Python packages:

```bash
pip install numpy matplotlib scipy

▶️ How to Run
Place your frequency data files in the same directory.

Make sure they're comma-separated with two columns: frequency,magnitude.

Update the file names if needed in the script:

python
Copy
Edit
file1 = 'Queen_sound_youtube_downsampled_ICS43434_35000Hz.txt'
file2 = 'Queen_sound_youtube_downsampled.txt'
Run the script:

bash
Copy
Edit
python frequency_almost_equal.py
🧠 Use Cases
Compare real sensor data with reference samples

Validate MEMS microphone recordings

Analyze spectral similarity between different sources


