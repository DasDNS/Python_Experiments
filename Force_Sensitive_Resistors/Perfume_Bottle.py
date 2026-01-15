import matplotlib.pyplot as plt
import numpy as np

# =====================================
# Correctly mapped data (PA6 removed)
# =====================================

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

values = [
    101.94,   # Little (PA0)
    101.80,   # Ring (PA3)
    102.12,   # Middle (PB1)
    102.01,   # Index (PB0)
    102.11,   # Thumb (PA7)
    101.81,   # Little_End (PA1)
    101.49,   # Ring_End (PA2)
    994.90,   # Palm (PA5)
    962.59    # Thumb_End (PA4)
]

# =====================================
# BAR CHART
# =====================================
plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.title("Cuboidal Shape – Perfume Bottle (FSR Readings)")
plt.ylabel("FSR Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================================
# RADAR CHART
# =====================================

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values_radar = values + [values[0]]
angles_radar = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.plot(angles_radar, values_radar)
ax.fill(angles_radar, values_radar, alpha=0.25)

ax.set_xticks(angles)
ax.set_xticklabels(labels)

plt.title("Cuboidal Shape – Perfume Bottle (Radar Plot)")
plt.tight_layout()
plt.show()
