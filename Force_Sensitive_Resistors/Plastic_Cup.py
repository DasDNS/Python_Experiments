import matplotlib.pyplot as plt
import numpy as np

# ===============================
# Finger labels (fixed axis order)
# ===============================
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

# ===============================
# FSR values (PA6 removed + remapped)
# Plastic Cup (Conical Object)
# ===============================
values = [0.22, 0.35, 0.26, 0.27, 0.21, 0.19, 0.25, 0.16, 0.22]

# =====================================
# BAR CHART
# =====================================
plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.title("FSR Distribution – Conical Object (Plastic Cup)")
plt.ylabel("FSR Reading")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================================
# RADAR CHART
# =====================================
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
values_radar = values + values[:1]  # close loop
angles_radar = np.concatenate((angles, [angles[0]]))

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles_radar, values_radar)
ax.fill(angles_radar, values_radar, alpha=0.25)

ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.set_title("Radar Chart – Conical Object (Plastic Cup)")

plt.show()
