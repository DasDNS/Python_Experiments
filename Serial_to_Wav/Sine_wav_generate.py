import serial
import wave
import threading
import time

# Configuration
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
WAV_FILE = 'output_sine_wave.wav'
SAMPLE_RATE = 16000
DURATION = 5
CHUNK_SIZE = 1024
SAMPLES = SAMPLE_RATE * DURATION

# Serial connection setup
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
ser.flushInput()

# Function to read from serial and write to WAV file in a separate thread
def read_and_write():
    with wave.open(WAV_FILE, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)

        for _ in range(SAMPLES // CHUNK_SIZE):
            data = ser.read(2 * CHUNK_SIZE)  # Read chunk
            if len(data) == 2 * CHUNK_SIZE:  # Ensure full chunk is read
                wav_file.writeframes(data)

        print(f"Sine wave saved to {WAV_FILE}")

# Start the reading and writing process in a separate thread
thread = threading.Thread(target=read_and_write)
thread.start()

# Main thread can be used for other tasks or waiting for completion
thread.join()

# Closing the serial port after process completes
if ser.is_open:
    ser.close()
    print("Serial port closed.")
