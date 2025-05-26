# 📤 WAV to Text File Converter (16-bit PCM)

This Python script reads a **16-bit PCM WAV audio file** and exports its raw PCM values to a plain text file, where each line represents one sample value.

---

## 📁 Project Structure
.
├── 16bit_audio_input.wav # Input WAV file (must be 16-bit PCM)
├── output_audio_file.txt # Output text file containing PCM values
├── wav_to_txt_converter.py # This Python script
└── README.md # Documentation


---

## 🛠️ Features

- Reads mono or multi-channel 16-bit WAV files.
- Extracts and decodes raw audio sample values.
- Saves each sample as an integer in a `.txt` file.
- Designed for audio analysis, visualization, or preprocessing for ML.

---

## ✅ Requirements

This script uses only Python's **standard library**:

- `wave`
- `struct`

No external dependencies are required.

---

## ▶️ Usage

1. Place your **16-bit PCM WAV** file in the same directory and name it `16bit_audio_input.wav`.
2. Run the script:

```bash
python wav_to_txt_converter.py

The output will be saved as output_audio_file.txt, with one PCM value per line.

📌 Notes
Only 16-bit WAV files are supported. The script checks for this and raises a ValueError if not.

If the WAV file is stereo (2 channels), the samples will be interleaved.

🔧 Example Output
A sample of the text file might look like:

python-repl
Copy
Edit
0
123
-234
...
Each number corresponds to one 16-bit audio sample.


