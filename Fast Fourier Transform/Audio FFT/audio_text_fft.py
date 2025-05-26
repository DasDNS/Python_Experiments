import wave
import numpy as np

def read_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("Only 16-bit WAV files are supported.")
        
        num_channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        # Read audio data
        raw_data = wav_file.readframes(num_frames)
        # Convert raw byte data to 16-bit PCM values
        pcm_values = np.frombuffer(raw_data, dtype=np.int16)
        return pcm_values, sample_rate, num_channels

def compute_fft(pcm_values, sample_rate):
    # Compute FFT
    fft_values = np.fft.fft(pcm_values)
    freqs = np.fft.fftfreq(len(fft_values), 1/sample_rate)
    
    # Only return the positive half of the frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft_values = fft_values[:len(fft_values)//2]
    
    return positive_freqs, positive_fft_values

def save_to_text_file(freqs, fft_values, output_file):
    with open(output_file, 'w') as f:
        for freq, value in zip(freqs, fft_values):
            f.write(f"{freq}\t{value.real}\t{value.imag}\n")

if __name__ == "__main__":
    input_wav_file = '16bit_audio_input.wav'
    output_text_file = 'fft_output.txt'

    pcm_values, sample_rate, num_channels = read_wav_file(input_wav_file)
    freqs, fft_values = compute_fft(pcm_values, sample_rate)
    save_to_text_file(freqs, fft_values, output_text_file)

    print(f"FFT results saved to {output_text_file}")
