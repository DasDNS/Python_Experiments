# USB CDC Serial Data Receiver

This Python script is designed to receive binary data over a USB CDC serial connection from a microcontroller (e.g., STM32) and save it to a local file. It also provides basic real-time logging of received data size and transmission speed.

## 📦 Features

- Listens on a specified USB CDC serial port (e.g., `/dev/ttyACM1`).
- Receives and writes binary data to a file.
- Logs:
  - Total bytes received
  - Time taken for transmission
  - Measured data rate in bytes per second

## 🛠 Requirements

- Python 3.x
- `pyserial` library

Install using pip:

```bash
pip install pyserial

📄 Script Overview
File: receive_usb_data.py
python
Copy
Edit
PORT = "/dev/ttyACM1"  # Modify this for your USB CDC device
BAUD_RATE = 115200     # Required by pyserial (ignored for USB CDC)
OUTPUT_FILE = "received_data.txt"
PORT: Serial port used for communication (e.g., /dev/ttyACM1 or COM3).

BAUD_RATE: Still needed by PySerial, but not used for USB CDC.

OUTPUT_FILE: The file where all incoming data will be stored (in binary mode).

🚀 Usage
Connect your STM32 (or other USB CDC-enabled device).

Confirm the correct serial port using:

bash
Copy
Edit
ls /dev/ttyACM*  # On Linux
Run the script:

bash
Copy
Edit
python receive_usb_data.py
Data transmission starts automatically. Once the stream ends (timeout), the script:

Closes the file

Prints summary statistics

📝 Example Output
yaml
Copy
Edit
Listening on /dev/ttyACM1 ...
Received 4096 bytes, Total: 4096 bytes
Received 2048 bytes, Total: 6144 bytes
No more data received, exiting...
Transmission Completed. Total bytes received: 6144
Time elapsed: 0.45 seconds
Measured Speed: 13653.78 B/s
🧠 Notes
For STM32, ensure the USB CDC interface is correctly configured.

The script uses a timeout-based method to detect the end of transmission.

Works best for short-to-medium data bursts (e.g., file transfers, sensor dumps).
