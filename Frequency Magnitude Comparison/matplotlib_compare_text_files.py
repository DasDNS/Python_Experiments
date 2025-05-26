import numpy as np
import matplotlib.pyplot as plt

# Load the data from both files
file1 = 'Queen_sound_youtube_downsampled_ICS43434_35000Hz.txt'
file2 = 'Queen_sound_youtube_downsampled.txt'

# Load both files
data1 = np.loadtxt(file1, delimiter=',')
data2 = np.loadtxt(file2, delimiter=',')

# Extract frequencies and magnitudes
freq1, mag1 = data1[:, 0], data1[:, 1]
freq2, mag2 = data2[:, 0], data2[:, 1]

# Find common frequencies between both datasets
common_freqs = np.intersect1d(freq1, freq2)

# Count the number of common frequencies and the number of unique frequencies in each file
num_common = len(common_freqs)
num_file1_unique = len(freq1) - num_common
num_file2_unique = len(freq2) - num_common

# Output the counts
print(f'Number of common frequencies: {num_common}')
print(f'Number of unique frequencies in ICSfile: {num_file1_unique}')
print(f'Number of unique frequencies in file2: {num_file2_unique}')

# Interpolate mag2 to match common frequencies from file1
from scipy.interpolate import interp1d
interp_func = interp1d(freq2, mag2, kind='linear', fill_value='extrapolate')
mag2_interp = interp_func(common_freqs)

# Plot both datasets with transparency
plt.figure(figsize=(10, 6))
plt.plot(freq1, mag1, label='ICSfile', linestyle='-', color='b', alpha=0.7)
plt.plot(freq2, mag2, label='File 2', linestyle='--', color='r', alpha=0.7)

# Plot the common frequencies and highlight the overlap
plt.scatter(common_freqs, mag1[np.isin(freq1, common_freqs)], color='green', label='Common Frequencies', zorder=5)

# Highlight overlap region (where mag1 and mag2_interp are close)
overlap = np.abs(mag1[np.isin(freq1, common_freqs)] - mag2_interp) < 0.1  # Adjust threshold if needed
plt.fill_between(common_freqs, mag1[np.isin(freq1, common_freqs)], mag2_interp, where=overlap, color='gray', alpha=0.3, label='Overlap')

# Labels and title
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.title('Comparison of Frequency vs Magnitude with Overlap and Unique Frequencies')

# Show the plot
plt.show()
