# PCM Text to WAV File Converter

This Python script converts a plain text file (`Output_Test.txt`) containing 16-bit PCM audio sample values into a standard WAV audio file (`output_audio_test.wav`).

## 📜 Description

The script reads integer PCM values from a text file where each line represents a single audio sample. It then writes these samples to a `.wav` file using Python's built-in `wave` and `struct` modules.

This is useful when raw PCM data is collected or generated outside standard audio software (e.g., from microcontrollers or custom DSP pipelines) and needs to be converted to a playable audio format.

## 📁 File Structure
.
├── Output_Test.txt # Input text file containing 16-bit PCM audio data
├── output_audio_test.wav # Generated WAV audio file
├── pcm_text_to_wav.py # Python script
└── README.md # This documentation


## ▶️ How to Use

1. **Prepare the Input File**

   Ensure `Output_Test.txt` contains raw PCM values (one per line), e.g.:

0
132
-257
32767
-32768

pgsql
Copy
Edit


These should be 16-bit signed integers (from -32768 to 32767).

2. **Run the Script**

```bash
python3 pcm_text_to_wav.py

Output

A valid WAV audio file named output_audio_test.wav will be created in the current directory.

⚙️ Configuration
You can modify these variables in the script:

input_text_file: Input file name (default: Output_Test.txt)

output_wav_file: Output file name (default: output_audio_test.wav)

num_channels: Number of audio channels (1 for mono, 2 for stereo)

sample_rate: Sampling rate in Hz (default: 44100 for CD quality)

🛠️ Requirements
Python 3.x

No third-party libraries required
