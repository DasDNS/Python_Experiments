# 🎙️ Serial Audio Data Recorder (ESP32-Compatible)

This Python script reads raw 16-bit PCM audio data from a serial port (e.g., from an ESP32), logs it into a `.txt` file, and optionally converts it into a `.wav` file for playback and analysis.

## 📜 Description

- Connects to a serial port to receive audio data streamed in real time.
- Expects data in 16-bit signed PCM format (`int16`) as bytes.
- Saves the raw sample data to a text file (`.txt`) for analysis.
- Converts the data into a playable WAV file using the calculated sample rate.

> Designed to match the buffer size and sample rate used on an ESP32.

---

## 📁 Project Structure
.
├── audio_data_26_11.txt # Text file with saved PCM data
├── audio_data_26_11.wav # Converted WAV file for playback
├── serial_audio_recorder.py # Main Python script
└── README.md # This file


---

## ⚙️ Configuration

Adjust these constants in the script to match your setup:

```python
SERIAL_PORT = "COM4"         # Your device's serial port
BAUD_RATE = 115200           # Must match the device's baud rate
SAMPLE_RATE = 16384          # Same rate as used by the ESP32
BUFFER_SIZE = 512            # Must match ESP32's buffer chunk size

▶️ How to Use
Connect your device that is streaming PCM audio data via UART (e.g., ESP32).

Run the script:

bash
Copy
Edit
python serial_audio_recorder.py
Stop recording any time with Ctrl+C.

Upon completion:

A .txt file (audio_data_26_11.txt) will contain raw PCM values.

A .wav file (audio_data_26_11.wav) will be generated for playback using any media player.

🛠️ Requirements
Python 3.x

Required libraries:

pyserial

numpy

Install them using:

bash
Copy
Edit
pip install pyserial numpy
🧠 Notes
The script automatically computes the effective sample rate from the actual number of samples received.

It skips incomplete or partial samples to avoid audio corruption.

Useful for embedded audio testing, remote mic streaming, or serial data acquisition.
