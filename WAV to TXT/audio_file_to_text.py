import wave
import struct

def read_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Ensure it is 16-bit PCM
        if wav_file.getsampwidth() != 2:
            raise ValueError("Only 16-bit WAV files are supported.")

        num_channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        # Read audio data
        raw_data = wav_file.readframes(num_frames)
        # Convert raw byte data to 16-bit PCM values
        pcm_values = struct.unpack('<' + 'h' * (num_frames * num_channels), raw_data)

        return pcm_values, num_channels, sample_rate

def save_to_text_file(pcm_values, output_file):
    with open(output_file, 'w') as f:
        for value in pcm_values:
            f.write(f"{value}\n")

if __name__ == "__main__":
    input_wav_file = '16bit_audio_input.wav'
    output_text_file = 'output_audio_file.txt'

    pcm_values, num_channels, sample_rate = read_wav_file(input_wav_file)
    save_to_text_file(pcm_values, output_text_file)

    print(f"PCM values saved to {output_text_file}")
