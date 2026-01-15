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
    0.46,     # Little (PA0)
    0.84,     # Ring (PA3)
    0.65,     # Middle (PB1)
    1.76,     # Index (PB0)
    804.05,   # Thumb (PA7)
    25.42,    # Little_End (PA1)
    9.41,     # Ring_End (PA2)
    0.69,     # Palm (PA5)
    0.52      # Thumb_End (PA4)
]

# =====================================
# BAR CHART
# =====================================
plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.title("Cuboidal Shape – Toothpaste (FSR Readings)")
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

plt.title("Cuboidal Shape – Toothpaste (Radar Plot)")
plt.tight_layout()
plt.show()
