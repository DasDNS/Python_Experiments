# WAV File Information Extractor

This Python script extracts and displays technical details about a `.wav` audio file such as sample rate, bit depth, number of channels, duration, and more.

## Features

- Reads standard PCM `.wav` audio files
- Displays:
  - Number of channels (mono, stereo, etc.)
  - Sample width (in bytes)
  - Bit depth
  - Sampling rate (frame rate)
  - Total number of frames
  - Duration (in seconds)
  - Bit rate (in kbps)

## Requirements

- Python 3.x

This script uses only built-in libraries:
- `wave`
- `sys`

## Usage

### 1. Run with default file path:

By default, it looks for a file named `downloaded_audio.wav` in the same directory.

```bash
python wav_info.py

Example Output
yaml
Copy
Edit
WAV File: downloaded_audio.wav
Number of Channels: 1
Sample Width (Bytes): 2
Bit Depth: 16
Frame Rate (Sampling Rate): 8000 Hz
Number of Frames: 16000
Duration: 2.00 seconds
Bit Rate: 128.00 kbps
