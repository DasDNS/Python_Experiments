import serial
import struct

def read_pcm_from_serial(serial_port, baud_rate, num_samples):
    with serial.Serial(serial_port, baud_rate) as ser:
        pcm_values = []
        try:
            while len(pcm_values) < num_samples:
                raw_data = ser.read(2)  # 2 bytes per sample
                value = struct.unpack('<h', raw_data)[0]
                pcm_values.append(value)
        except KeyboardInterrupt:
            print("Recording stopped by user.")
        return pcm_values

def save_to_text_file(pcm_values, output_file):
    with open(output_file, 'w') as f:
        for value in pcm_values:
            f.write(f"{value}\n")

# Usage
serial_port = 'COM4'
baud_rate = 115200
num_samples = 1000  # Number of PCM samples to read
output_text_file = 'output.txt'

print("Recording... Press Ctrl+C to stop.")
try:
    pcm_values = read_pcm_from_serial(serial_port, baud_rate, num_samples)
    save_to_text_file(pcm_values, output_text_file)
    print(f"PCM values saved to {output_text_file}")
except KeyboardInterrupt:
    print("Recording interrupted!")

