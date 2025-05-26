# Get the number of almost equal frequencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the data from both files
file1 = 'Queen_sound_youtube_downsampled_ICS43434_35000Hz.txt'
file2 = 'Queen_sound_youtube_downsampled.txt'

# Load both files
data1 = np.loadtxt(file1, delimiter=',')
data2 = np.loadtxt(file2, delimiter=',')

# Extract frequencies and magnitudes
freq1, mag1 = data1[:, 0], data1[:, 1]
freq2, mag2 = data2[:, 0], data2[:, 1]

# Sort freq2 for efficient searching
freq2_sorted = np.sort(freq2)

# Define the threshold for "almost equal" frequencies
threshold = 0.25  # Adjust as needed

# Find "almost equal" frequencies using a binary search approach
almost_equal_freqs = []
for f1 in freq1:
    # Find the closest frequency in freq2
    idx = np.searchsorted(freq2_sorted, f1)
    if idx < len(freq2_sorted) and abs(freq2_sorted[idx] - f1) < threshold:
        almost_equal_freqs.append(f1)
    elif idx > 0 and abs(freq2_sorted[idx - 1] - f1) < threshold:
        almost_equal_freqs.append(f1)

almost_equal_freqs = np.array(almost_equal_freqs)
num_almost_equal = len(almost_equal_freqs)

# Find common frequencies between both datasets
common_freqs = np.intersect1d(freq1, freq2)

# Calculate total unique frequencies
total_unique_freqs = len(np.union1d(freq1, freq2))

# Combine common and almost equal frequencies
matching_freqs = np.union1d(common_freqs, almost_equal_freqs)
num_matching_freqs = len(matching_freqs)+len(common_freqs)

# Calculate similarity percentage
#similarity_percentage = (num_matching_freqs / total_unique_freqs) * 100

# Output the counts
print(f'Total frequencies in ICSfile: {len(freq1)}')
print(f'Total frequencies in File 2: {len(freq2)}')
print(f'Number of common frequencies: {len(common_freqs)}')
print(f'Number of unique frequencies in ICSfile: {len(freq1) - len(common_freqs)}')
print(f'Number of unique frequencies in File 2: {len(freq2) - len(common_freqs)}')
print(f'Number of almost equal frequencies: {num_almost_equal}')
print(f'Total unique frequencies: {total_unique_freqs}')
print(f'Matching frequencies (common + almost equal): {num_matching_freqs}')
#print(f'Similarity between the two files based on frequencies: {similarity_percentage:.2f}%')

# Interpolate mag2 to match common frequencies from file1
interp_func = interp1d(freq2, mag2, kind='linear', fill_value='extrapolate')
mag2_interp = interp_func(common_freqs)

# Get magnitudes for common frequencies from file1
mag1_common = mag1[np.isin(freq1, common_freqs)]

# Define overlap condition (adjust threshold as needed)
overlap = np.abs(mag1_common - mag2_interp) < 0.1

# Plot both datasets
plt.figure(figsize=(12, 8))
plt.plot(freq1, mag1, label='ICSfile', linestyle='-', color='blue', alpha=0.7)
plt.plot(freq2, mag2, label='File 2', linestyle='--', color='red', alpha=0.7)

# Highlight common frequencies
plt.scatter(common_freqs, mag1[np.isin(freq1, common_freqs)], color='green', label='Common Frequencies', zorder=5)

# Highlight almost equal frequencies
plt.scatter(almost_equal_freqs, np.interp(almost_equal_freqs, freq1, mag1), color='yellow', label='Almost Equal Frequencies', zorder=6)

# Highlight overlapping regions
plt.fill_between(
    common_freqs,
    mag1_common,
    mag2_interp,
    where=overlap,
    color='gray',
    alpha=0.5,
    label='Overlap'
)

# Add labels and title
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.title('Frequency-Magnitude Comparison with Common and Almost Equal Frequencies')

# Show the plot
plt.show()
