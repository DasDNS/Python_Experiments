# WAV File Transfer and FFT Data Receiver via UART

This Python script sends a 16-bit mono PCM WAV audio file over a UART serial connection to a connected STM32 device, waits to receive FFT magnitude data processed by the STM32, and saves the received FFT magnitudes to a text file.

---

## Features

- Reads and validates 16-bit mono PCM WAV files
- Sends raw audio data via UART to an STM32 microcontroller
- Receives FFT magnitude data (float32) from STM32 via UART
- Saves received FFT data to a text file
- Prints transfer and receive timing information

---

## Requirements

- Python 3.x
- [numpy](https://numpy.org/)
- [pyserial](https://pyserial.readthedocs.io/en/latest/)
- [wave](https://docs.python.org/3/library/wave.html) (built-in)

Install Python dependencies with:

```bash
pip install numpy pyserial

Usage
Connect your STM32 device to your PC via UART serial.

Configure the UART port and baud rate in the script:

python
Copy
Edit
UART_PORT = 'COM3'         # Change this to your serial port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
UART_BAUDRATE = 115200     # Adjust baud rate if necessary
Place the input WAV file (16-bit mono PCM) in the same directory or specify its path:

python
Copy
Edit
WAV_FILE = 'queen_present_4.wav'  # Your input WAV file path
Run the script:

bash
Copy
Edit
python send_wav_receive_fft.py
The script will:

Send the WAV data over UART

Wait for the STM32 to process and send back FFT magnitude data

Save the FFT magnitudes to output_fft.txt by default

Notes
The script expects the STM32 firmware to respond with FFT magnitude data in float32 format, sending FFT_SIZE (default 1024) magnitudes, each 4 bytes.

Ensure your STM32 UART communication protocol matches this behavior.

The input WAV file must be mono and 16-bit PCM.

Transfer timing is printed for diagnostics.

Troubleshooting
Serial port errors: Verify the correct port and that no other program is using it.

Timeouts or incomplete data: Increase UART_TIMEOUT or check STM32 transmission.

Incorrect WAV format: Use audio software (Audacity, etc.) to convert to mono 16-bit PCM.
