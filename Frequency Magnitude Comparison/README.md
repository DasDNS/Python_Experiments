# 🔍 Frequency Magnitude Comparison and Visualization

This Python script compares two datasets containing frequency vs. magnitude data — one from an ICS43434 microphone recording and another from a different source (e.g., YouTube downsampled audio). It visualizes the common and unique frequency components and highlights the regions where the magnitudes are similar.

---

## 📁 Project Files
.
├── Queen_sound_youtube_downsampled_ICS43434_35000Hz.txt # Frequency-magnitude data from ICS43434
├── Queen_sound_youtube_downsampled.txt # Frequency-magnitude data from reference source
├── frequency_comparison.py # Python script for analysis and plotting
└── README.md # Project documentation


---

## 📜 Description

The script:
- Loads and parses two `.txt` files containing frequency and magnitude pairs.
- Identifies and counts:
  - Common frequencies
  - Unique frequencies in each dataset
- Interpolates data for better comparison.
- Plots both datasets with:
  - Common frequencies marked
  - Overlap areas highlighted (where magnitudes are closely matched)

---

## 📊 Output Example

When the script runs, it prints:

Number of common frequencies: 1245
Number of unique frequencies in ICSfile: 355
Number of unique frequencies in file2: 198


It also displays a graph showing:

- Solid blue line for ICS43434 dataset
- Dashed red line for File 2
- Green dots for common frequencies
- Gray shaded area where magnitudes are within a similarity threshold

---

## 🧰 Requirements

- Python 3.x
- Required Python libraries:

```bash
pip install numpy matplotlib scipy

▶️ How to Run
Make sure your frequency data is stored as two-column text files (frequency, magnitude), separated by commas.

Adjust the file names in the script if needed:

python
Copy
Edit
file1 = 'Queen_sound_youtube_downsampled_ICS43434_35000Hz.txt'
file2 = 'Queen_sound_youtube_downsampled.txt'
Run the script:

bash
Copy
Edit
python frequency_comparison.py
📌 Notes
Adjust the threshold in this line to control what is considered "overlap":

python
Copy
Edit
overlap = np.abs(mag1[...] - mag2_interp) < 0.1
This comparison is useful for analyzing signal similarity, detecting matching frequency components, or validating sensor data (e.g., MEMS vs. standard mic).
