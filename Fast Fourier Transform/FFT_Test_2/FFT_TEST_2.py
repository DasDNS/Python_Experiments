import serial
import numpy as np
from scipy.io import wavfile

def send_audio_file(file_path, serial_port, baud_rate=115200):
    # Read WAV file
    print(f"Reading WAV file: {file_path}")
    sample_rate, data = wavfile.read(file_path)
    print(f"Sample rate: {sample_rate}, Data shape: {data.shape}")
    # Normalize data
    data = data / np.max(np.abs(data))
    data = (data * 32767).astype(np.int16)  # Convert to 16-bit PCM format

    # Open serial port
    print(f"Opening serial port: {serial_port}")
    ser = serial.Serial(serial_port, baud_rate)

    # Send data
    print("Sending audio data...")
    for sample in data:
        ser.write(sample.tobytes())
    print("Audio data sent successfully!")

    # Capture and save UART data
    print("Capturing UART data...")
    with open('output.txt', 'w') as file:
        while True:
            if ser.in_waiting > 0:
                received_data = ser.readline().decode('utf-8').strip()
                print(received_data)  # Print for debugging purposes
                file.write(received_data + '\n')
                file.flush()
                print("Saving received data to text file...")
    
    # Close serial port
    ser.close()
    print("Serial port closed.")

if __name__ == "__main__":
    audio_file = "queen_present.wav"  # Path to your WAV file
    serial_port = "COM6"  # Adjust this to your serial port
    send_audio_file(audio_file, serial_port)
