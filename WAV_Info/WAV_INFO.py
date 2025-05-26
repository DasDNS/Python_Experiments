import wave
import sys

def get_wav_info(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # Get file parameters
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            bit_depth = sample_width * 8
            duration = n_frames / frame_rate
            bit_rate = frame_rate * n_channels * bit_depth

            print(f"WAV File: {file_path}")
            print(f"Number of Channels: {n_channels}")
            print(f"Sample Width (Bytes): {sample_width}")
            print(f"Bit Depth: {bit_depth}")
            print(f"Frame Rate (Sampling Rate): {frame_rate} Hz")
            print(f"Number of Frames: {n_frames}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Bit Rate: {bit_rate / 1000:.2f} kbps")
    except wave.Error as e:
        print(f"Error reading WAV file: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Change the file path to the location of your WAV file
    wav_file_path = "downloaded_audio.wav"
    
    if len(sys.argv) > 1:
        wav_file_path = sys.argv[1]
    
    get_wav_info(wav_file_path)