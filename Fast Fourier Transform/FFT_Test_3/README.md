# Serial WAV Audio Sender

This Python script reads a WAV audio file, normalizes its data, and sends the 16-bit PCM audio samples over a serial port (UART) to a connected device.

---

## Features

- Reads WAV audio files using `scipy`
- Normalizes and converts audio data to 16-bit signed integers
- Sends audio data sample-by-sample over a specified serial port and baud rate

---

## Requirements

- Python 3.x
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pyserial](https://pyserial.readthedocs.io/en/latest/)

Install dependencies with:

```bash
pip install numpy scipy pyserial

Usage
Connect your device to your computer via serial (UART).

Modify the following variables in the script or update them directly in the __main__ block:

python
Copy
Edit
audio_file = "queen_present.wav"  # Path to your WAV file
serial_port = "COM6"              # Your serial port (e.g., "COM6" on Windows or "/dev/ttyUSB0" on Linux)
baud_rate = 115200                # Serial baud rate (default is 115200)
Run the script:

bash
Copy
Edit
python send_audio_file.py
Notes
The WAV file should be in a format readable by scipy.io.wavfile (typically 16-bit PCM).

Ensure that the serial port and baud rate match those configured on your receiving device.

The script sends raw 16-bit samples continuously; implement any necessary framing or protocol on the receiving side.

Troubleshooting
Serial port errors: Confirm your device is connected and the port name is correct.

Data transmission issues: Check baud rate compatibility and cable integrity.
