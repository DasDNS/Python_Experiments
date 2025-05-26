# PCM Serial Data Recorder

This Python script captures 16-bit PCM audio data sent over a serial connection and saves it to a plain text file. Each line in the text file contains a single PCM sample value. This is useful for analyzing audio samples or converting them into WAV files later.

## 📜 Description

- Connects to a serial port (e.g., `COM4`) at a specified baud rate (`115200`).
- Reads a specified number of PCM samples (e.g., `1000` samples).
- Each PCM value is assumed to be 16-bit little-endian (`<h` format).
- Writes the data into a `.txt` file line-by-line.

## 🧾 File Structure
.
├── pcm_serial_recorder.py # Python script
├── output.txt # Output text file with PCM samples
└── README.md # This documentation


## ⚙️ Configuration

Update these variables in the script according to your setup:

```python
serial_port = 'COM4'          # Serial port where your device is connected
baud_rate = 115200            # Serial communication baud rate
num_samples = 1000            # Number of samples to capture
output_text_file = 'output.txt'  # Destination file for saving PCM values

▶️ Usage
Connect your audio source (e.g., microcontroller) to the serial port.

Ensure the device sends raw 16-bit PCM samples continuously.

Run the script:

bash
Copy
Edit
python3 pcm_serial_recorder.py
After reading the specified number of samples, the data will be saved to output.txt.

🛑 You can stop recording early with Ctrl+C. Partial data will still be saved if interrupted.

🛠️ Requirements
Python 3.x

pyserial (pip install pyserial)

