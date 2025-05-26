import serial
import wave
import numpy as np
import time

# UART settings
UART_PORT = 'COM3'  # Change to your port
UART_BAUDRATE = 115200
UART_TIMEOUT = 1

# File paths
WAV_FILE = 'queen_present_4.wav'  # Input WAV file
OUTPUT_FILE = 'output_fft.txt'  # Output text file with FFT magnitudes
FFT_SIZE = 1024  # Match the FFT size used in STM32 code

def send_wav_receive_fft():
    try:
        # Open UART connection
        print(f"Opening UART connection on {UART_PORT} with baudrate {UART_BAUDRATE}...")
        with serial.Serial(UART_PORT, UART_BAUDRATE, timeout=UART_TIMEOUT) as ser:
            print(f"Opening WAV file: {WAV_FILE}...")
            with wave.open(WAV_FILE, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                samp_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()

                print(f"WAV file details: Channels={n_channels}, Sample Width={samp_width}, Frame Rate={frame_rate}, Frames={n_frames}")

                # Ensure the file is PCM 16-bit mono
                if n_channels != 1 or samp_width != 2:
                    raise ValueError("The .wav file must be 16-bit mono PCM format.")

                # Read frames and send them via UART
                wav_data = wav_file.readframes(n_frames)
                print("Starting WAV file transfer...")

                start_transfer_time = time.time()  # Start timing the transfer
                ser.write(wav_data)
                end_transfer_time = time.time()  # End timing the transfer

                print("WAV file transfer completed.")

                # Wait for STM32 to finish processing and return FFT magnitudes
                print("Waiting for FFT data from STM32...")

                start_receive_time = time.time()  # Start timing the receive
                fft_data = ser.read(FFT_SIZE * 4)  # Read FFT_SIZE values (4 bytes each for float32)
                end_receive_time = time.time()  # End timing the receive

                print(f"Received {len(fft_data)} bytes of FFT magnitude data.")

                # Convert received data into numpy array of 32-bit floats
                fft_magnitudes = np.frombuffer(fft_data, dtype=np.float32)

                # Save FFT magnitudes to a text file
                print(f"Saving FFT data to {OUTPUT_FILE}...")
                with open(OUTPUT_FILE, 'w') as f:
                    for value in fft_magnitudes:
                        f.write(f"{value}\n")
                print(f"FFT data saved to {OUTPUT_FILE}.")

                # Print timing information
                print("\n--- Timing Information ---")
                print(f"Time to send WAV file: {end_transfer_time - start_transfer_time:.2f} seconds")
                print(f"Time to receive FFT data: {end_receive_time - start_receive_time:.2f} seconds")
                print(f"Total process time: {end_receive_time - start_transfer_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    send_wav_receive_fft()
