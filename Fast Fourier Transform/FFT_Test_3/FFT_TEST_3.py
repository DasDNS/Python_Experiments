import serial
import numpy as np
from scipy.io import wavfile

def send_audio_file(file_path, serial_port, baud_rate=115200):
    # Read WAV file
    sample_rate, data = wavfile.read(file_path)
    # Normalize data
    data = data / np.max(np.abs(data))
    data = (data * 32767).astype(np.int16)  # Convert to 16-bit PCM format

    # Open serial port
    ser = serial.Serial(serial_port, baud_rate)

    # Send data
    for sample in data:
        ser.write(sample.tobytes())

    # Close serial port
    ser.close()

if __name__ == "__main__":
    audio_file = "queen_present.wav"  # Path to your WAV file
    serial_port = "COM6"  # Adjust this to your serial port
    send_audio_file(audio_file, serial_port)
