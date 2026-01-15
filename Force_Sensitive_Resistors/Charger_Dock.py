import matplotlib.pyplot as plt
import numpy as np

# Finger / location labels (final order)
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

# Raw snapshot data in *code order*:
# PB1, PB0, PA7, PA6(ignore), PA5, PA4, PA3, PA2, PA1, PA0
raw = [0.12, 5.04, 0.09, 0.17, 0.16, 0.08, 0.22, 0.06, 0.06, 1.96]

# --------- Correct Mapping (Ignoring PA6) ---------
mapped = {
    "Little": raw[9],       # PA0
    "Ring": raw[6],         # PA3
    "Middle": raw[0],       # PB1
    "Index": raw[1],        # PB0
    "Thumb": raw[2],        # PA7
    "Little_End": raw[8],   # PA1
    "Ring_End": raw[7],     # PA2
    "Palm": raw[4],         # PA5
    "Thumb_End": raw[5],    # PA4
}

values = list(mapped.values())

# ---------------- BAR CHART ----------------
plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.title("Cuboidal Object 4 - Charger Dock (Bar Chart)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------- RADAR CHART ----------------
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values_cycle = values + values[:1]  # Loop back to start
angles_cycle = angles + angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles_cycle, values_cycle)
ax.fill(angles_cycle, values_cycle, alpha=0.1)
ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.set_title("Cuboidal Object 4 - Charger Dock (Radar Chart)")
plt.show()
