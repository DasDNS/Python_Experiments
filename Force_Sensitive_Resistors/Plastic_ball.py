import matplotlib.pyplot as plt
import numpy as np

# Labels for the fingers/locations
labels = [
    "Little",
    "Ring",
    "Middle",
    "Index",
    "Thumb",
    "Little_End",
    "Ring_End",
    "Palm",
    "Thumb_End"
]

# Raw snapshot data (code order)
raw = [0.11, 0.14, 0.12, 0.11, 24.27, 0.12, 0.05, 0.10, 72.69, 0.16]

# Correct mapping based on pins (Ignore PA6)
mapped = {
    "Little": raw[9],        # PA0
    "Ring": raw[6],          # PA3
    "Middle": raw[0],        # PB1
    "Index": raw[1],         # PB0
    "Thumb": raw[2],         # PA7
    "Little_End": raw[8],    # PA1
    "Ring_End": raw[7],      # PA2
    "Palm": raw[4],          # PA5
    "Thumb_End": raw[5],     # PA4
}

values = list(mapped.values())

# ---------------- BAR CHART ----------------
plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.title("Spherical Object 1 - Plastic Ball (Bar Chart)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------- RADAR CHART ----------------
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values_cycle = values + values[:1]
angles_cycle = angles + angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles_cycle, values_cycle)
ax.fill(angles_cycle, values_cycle, alpha=0.1)
ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.set_title("Spherical Object 1 - Plastic Ball (Radar Chart)")
plt.show()
