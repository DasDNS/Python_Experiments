import numpy as np
import scipy.io.wavfile as wav

# Read the 32-bit WAV file
rate, data = wav.read("Queen_not_present.wav")

# Check if the data is in 32-bit range
if data.dtype == np.int32:
    # Convert 32-bit integer to 16-bit integer
    data_16bit = (data / 2**16).astype(np.int16)  # Scale down and cast to int16
else:
    raise ValueError("The input WAV file is not a 32-bit audio file.")

# Save the file as 16-bit WAV
wav.write("Queen_not_present_16.wav", rate, data_16bit)

print("Conversion complete: Saved as 'Queen_not_present_16.wav'")
