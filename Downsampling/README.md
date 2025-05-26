# 🎧 WAV File Bit Depth Converter

This Python script converts a **32-bit PCM WAV audio file** to a **16-bit PCM WAV file**. This is useful when dealing with audio processing systems or libraries that only support 16-bit files.

---

## 📁 Files Overview
.
├── Queen_not_present.wav # Original 32-bit WAV file
├── Queen_not_present_16.wav # Output 16-bit WAV file
├── convert_to_16bit.py # This Python script
└── README.md # Documentation


---

## 🔍 What It Does

- Reads a 32-bit `.wav` file.
- Confirms the data type is 32-bit integer (`int32`).
- Scales down the values and converts them to 16-bit integer (`int16`).
- Saves the result as a new 16-bit WAV file.

---

## ⚙️ Requirements

Install the required Python libraries:

```bash
pip install numpy scipy

▶️ How to Use
Ensure the Queen_not_present.wav file is in the same directory.

Run the script:

bash
Copy
Edit
python convert_to_16bit.py
If successful, a file named Queen_not_present_16.wav will be created.

📌 Notes
This script assumes the input file is in 32-bit signed integer PCM format.

If the input file is not 32-bit, it raises a ValueError.

The conversion scales the data by dividing by 2^16 to fit within the int16 range.

🧠 Why Use This?
Some audio libraries, DSP tools, or embedded systems may not support 32-bit PCM.

Reducing bit depth can save disk space (at the cost of dynamic range).
