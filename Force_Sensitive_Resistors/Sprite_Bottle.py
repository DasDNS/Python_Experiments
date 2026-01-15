import matplotlib.pyplot as plt
import numpy as np

# Final ordered labels
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

# Mapped values for cylindrical (Sprite bottle)
fsr_values = [
    7.98,     # Little       (PA0)
    16.30,    # Ring         (PA3)
    8.16,     # Middle       (PB1)
    8.00,     # Index        (PB0)
    7.94,     # Thumb        (PA7)
    7.89,     # Little_End   (PA1)
    7.96,     # Ring_End     (PA2)
    991.11,   # Palm         (PA5)
    8.38      # Thumb_End    (PA4)
]

# -------------------------------------------
# BAR CHART
# -------------------------------------------
plt.figure(figsize=(10, 5))
plt.bar(labels, fsr_values)
plt.xlabel("Mapped Finger / Location")
plt.ylabel("FSR Reading")
plt.title("Cylindrical Object (Sprite Bottle) - Bar Chart")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------------
# RADAR CHART
# -------------------------------------------

values = fsr_values + [fsr_values[0]]   # close loop
angles = np.linspace(0, 2 * np.pi, len(values))

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title("Cylindrical Object (Sprite Bottle) - Radar Chart", y=1.1)
plt.show()
