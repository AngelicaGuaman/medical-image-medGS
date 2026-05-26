import nibabel as nib
import numpy as np

label_path = r"US_data/US_volunteer_dataset/ground_truth_data/US_thyroid_label/001_P1_1_left.nii"

lbl = nib.load(label_path).get_fdata()

vals, counts = np.unique(lbl, return_counts=True)

print("\nClasses:\n")

for v, c in zip(vals, counts):
    print(f"Class {v}: {c} voxels")