import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths
# =========================

us_path = r"US_data/US_volunteer_dataset/ground_truth_data/US/001_P1_1_left_US.nii"
label_path = r"US_data/US_volunteer_dataset/ground_truth_data/US_thyroid_label/001_P1_1_left.nii"

# =========================
# Load
# =========================

us_img = nib.load(us_path)
lbl_img = nib.load(label_path)

us = us_img.get_fdata().astype(np.float32)
lbl = lbl_img.get_fdata().astype(np.float32)

# =========================
# Info
# =========================

print("US shape:", us.shape)
print("LABEL shape:", lbl.shape)

print("\nUS spacing:", us_img.header.get_zooms())
print("LABEL spacing:", lbl_img.header.get_zooms())

print("\nUS affine:\n", us_img.affine)
print("\nLABEL affine:\n", lbl_img.affine)

print("\nLabel unique:", np.unique(lbl))

# =========================
# Normalize US
# =========================

p1, p99 = np.percentile(us, [1, 99])

us_n = np.clip(
    (us - p1) / (p99 - p1 + 1e-8),
    0,
    1
)

# =========================
# Choose slice
# =========================

z = us.shape[2] // 2

# =========================
# Visualization
# =========================

fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(us_n[:, :, z].T, cmap="gray", origin="lower")
ax[0].set_title("US")

ax[1].imshow(lbl[:, :, z].T, cmap="gray", origin="lower")
ax[1].set_title("LABEL")

ax[2].imshow(us_n[:, :, z].T, cmap="gray", origin="lower")
ax[2].imshow(
    lbl[:, :, z].T,
    cmap="jet",
    alpha=0.4,
    origin="lower"
)
ax[2].set_title("Overlay")

# boundary only
boundary = np.logical_xor(
    lbl[:, :, z],
    np.pad(lbl[:, :, z][1:, :], ((0,1),(0,0)))
)

ax[3].imshow(us_n[:, :, z].T, cmap="gray", origin="lower")
ax[3].contour(
    lbl[:, :, z].T,
    colors="red",
    linewidths=1
)
ax[3].set_title("Contour")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()