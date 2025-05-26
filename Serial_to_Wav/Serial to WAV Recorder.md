# Serial to WAV Recorder

This Python script reads 16-bit PCM audio data sent over a serial connection (e.g., from a microcontroller like an ESP32) and saves it as a `.wav` file. The data is collected in real-time, buffered in chunks, and saved in a separate thread to ensure smooth operation.

## 📜 Description

- Opens a serial connection to a specified port (e.g., `COM6`) at `115200` baud.
- Reads raw audio data for a specified duration (`5 seconds`) at a sampling rate of `16,000 Hz`.
- Saves the incoming audio data to a WAV file (`output_sine_wave.wav`) in mono format (16-bit PCM).
- Uses multithreading to read and write simultaneously.

## 🧾 File Structure
.
├── serial_to_wav.py # Python script
├── output_sine_wave.wav # Generated WAV file
└── README.md # This documentation


## ⚙️ Configuration

You can adjust these values in the script:

```python
SERIAL_PORT = 'COM6'         # Serial port connected to the device
BAUD_RATE = 115200           # Baud rate for serial communication
WAV_FILE = 'output_sine_wave.wav'  # Output WAV file name
SAMPLE_RATE = 16000          # Audio sampling rate
DURATION = 5                 # Duration of recording (in seconds)
CHUNK_SIZE = 1024            # Number of samples per chunk

⚠️ Important: Make sure your connected device is sending raw 16-bit little-endian PCM data continuously for the duration of the recording.

▶️ Usage
Connect your audio-generating device to the serial port.

Modify the script to match your correct SERIAL_PORT.

Run the script:

bash
Copy
Edit
python3 serial_to_wav.py
The resulting output_sine_wave.wav file can be played using any audio player.

🛠️ Requirements
Python 3.x

pyserial library (pip install pyserial)
