# Serial WAV Audio Sender

This Python script reads a WAV audio file, sends the audio samples over a serial port (UART) to a connected device, and captures any incoming serial data from the device for a fixed duration, saving it to a text file.

---

## Features

- Reads 16-bit PCM WAV files (mono or stereo)
- Normalizes and converts audio data to 16-bit signed integers
- Sends audio data in chunks over serial port at specified baud rate
- Receives and logs serial data sent back from the connected device
- Saves received serial data to `output.txt`
- Configurable serial port and baud rate

---

## Requirements

- Python 3.x
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pyserial](https://pyserial.readthedocs.io/en/latest/)

Install dependencies via pip:

```bash
pip install numpy scipy pyserial

Usage
Connect your device (e.g., microcontroller) to the PC via a serial port.

Adjust the following parameters in the script or update the __main__ section:

python
Copy
Edit
audio_file = "queen_present.wav"  # Path to your WAV file
serial_port = "COM6"              # Your serial port (e.g., "COM6" on Windows or "/dev/ttyUSB0" on Linux)
baud_rate = 115200                # Baud rate (adjust as needed)
capture_duration = 10             # Duration in seconds to capture UART responses
Run the script:

bash
Copy
Edit
python send_audio_serial.py
The script will:

Read and normalize the WAV file audio data

Send the audio samples over the serial port

Capture incoming serial data from the device for the specified duration

Save captured data to output.txt

Notes
The script assumes the WAV file is 16-bit PCM format.

Ensure your serial port settings (port, baud rate) match those of your connected device.

The capture duration limits how long the script listens for incoming UART data.

Modify chunk size or delay between writes if you experience communication issues.

Troubleshooting
Serial port not found or permission denied: Verify the port name and permissions (on Linux, you might need to add your user to the dialout group or use sudo).

Data not sending/receiving properly: Check baud rate and cable connections.

UnicodeDecodeError during UART read: Incoming data might not be UTF-8; script skips undecodable lines.


