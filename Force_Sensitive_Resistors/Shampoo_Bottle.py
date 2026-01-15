import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Correct Finger Order
# ==============================
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

# ==============================
# Shampoo Bottle FSR Values (PA6 ignored)
# Correctly re-mapped values
# ==============================
values = [0.83, 0.83, 0.93, 0.77, 0.83, 0.81, 2.96, 0.91, 0.35]

# =====================================
# BAR CHART
# =====================================
plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.title("FSR Distribution – Cylindrical Object (Shampoo Bottle)")
plt.ylabel("FSR Reading")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================================
# RADAR CHART
# =====================================
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
values_radar = values + values[:1]  # close the circle
angles_radar = np.concatenate((angles, [angles[0]]))

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles_radar, values_radar)
ax.fill(angles_radar, values_radar, alpha=0.25)

ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.set_title("Radar Chart – Cylindrical Object (Shampoo Bottle)")

plt.show()
