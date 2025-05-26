import wave
import struct

def read_text_file(input_file):
    with open(input_file, 'r') as f:
        pcm_values = [int(line.strip()) for line in f]
    return pcm_values

def save_to_wav_file(pcm_values, output_file, num_channels, sample_rate):
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)

        # Convert PCM values to byte data
        raw_data = struct.pack('<' + 'h' * len(pcm_values), *pcm_values)
        wav_file.writeframes(raw_data)

if __name__ == "__main__":
    input_text_file = 'Output_Test.txt'
    output_wav_file = 'output_audio_test.wav'
    num_channels = 1  # Mono audio
    sample_rate = 44100  # 44.1 kHz

    pcm_values = read_text_file(input_text_file)
    save_to_wav_file(pcm_values, output_wav_file, num_channels, sample_rate)

    print(f"WAV file saved to {output_wav_file}")
