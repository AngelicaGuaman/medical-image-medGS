import nibabel as nib
import numpy as np
import imageio.v3 as iio
from pathlib import Path
import json

# =========================================================
# DIRECTORIOS
# =========================================================

us_dir = Path("../US_data/US_volunteer_dataset/ground_truth_data/US")

label_dir = Path(
    "../US_data/US_volunteer_dataset/ground_truth_data/US_thyroid_label"
)

output_root = Path("../medgs_data")

# =========================================================
# FUNCIONES
# =========================================================

def process_volume(us_path, label_path, out_dir):

    metadata = []
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"

    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Processing:")
    print(f"US     : {us_path.name}")
    print(f"LABEL  : {label_path.name}")

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    us_img = nib.load(str(us_path))
    lbl_img = nib.load(str(label_path))

    us = us_img.get_fdata().astype(np.float32)
    lbl = lbl_img.get_fdata().astype(np.uint8)

    if us.shape != lbl.shape:
        print(f"❌ Shape mismatch: US {us.shape} vs LABEL {lbl.shape}")
        return

    if not np.allclose(us_img.affine, lbl_img.affine):
        print("⚠️ Warning: affine mismatch")

    spacing = tuple(float(x) for x in us_img.header.get_zooms()[:3])

    print("Shape:", us.shape)
    print("Spacing:", spacing)
    print("Label classes:", np.unique(lbl))

    # -----------------------------------------------------
    # NORMALIZE
    # -----------------------------------------------------

    p1, p99 = np.percentile(us, [1, 99])

    us = np.clip((us - p1) / (p99 - p1 + 1e-8), 0, 1)

    us = (us * 255).astype(np.uint8)

    # -----------------------------------------------------
    # VALID SLICES
    # -----------------------------------------------------

    valid_z = []

    for z in range(lbl.shape[2]):

        if np.sum(lbl[:, :, z] == 1) > 50:
            valid_z.append(z)

    if len(valid_z) == 0:
        print("⚠️ No valid slices")
        return

    print(
        f"✔️ Valid slices: {valid_z[0]} -> {valid_z[-1]}"
    )

    # -----------------------------------------------------
    # EXPORT
    # -----------------------------------------------------

    for i, z in enumerate(valid_z):

        img = us[:, :, z]

        mask = (
            (lbl[:, :, z] == 1)
            .astype(np.uint8) * 255
        )

        iio.imwrite(img_dir / f"{i:04d}.png", img)

        iio.imwrite(mask_dir / f"{i:04d}.png", mask)

        image_name = f"{i:04d}.png"
        mask_name = f"{i:04d}.png"

        metadata.append({
            "index": int(i),
            "original_z": int(z),
            "t": float(i / (len(valid_z) - 1)) if len(valid_z) > 1 else 0.0,
            "image": f"images/{image_name}",
            "mask": f"masks/{mask_name}",
        })

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


    case_info = {
        "sample_name": out_dir.name,
        "us_file": str(us_path),
        "label_file": str(label_path),
        "task": "thyroid_binary",
        "class_mapping": {
            "0": "background",
            "1": "thyroid",
        },
        "original_shape": list(map(int, us.shape)),
        "spacing_mm": list(spacing),
        "valid_z_start": int(valid_z[0]),
        "valid_z_end": int(valid_z[-1]),
        "num_slices": int(len(valid_z)),
        "normalization": {
            "type": "percentile",
            "p1": float(p1),
            "p99": float(p99),
        },
        "affine": us_img.affine.tolist(),
    }


    with open(out_dir / "case_info.json", "w", encoding="utf-8") as f:
        json.dump(case_info, f, indent=2)   

    print(f"✔️ Exported {len(valid_z)} slices")


# =========================================================
# RECORRER TODOS LOS US
# =========================================================

us_files = sorted(us_dir.glob("*.nii"))

print(f"\n🔍 Found {len(us_files)} US volumes")

for us_path in us_files:

    # nombre:
    # 001_P1_1_left_US.nii

    stem = us_path.stem
    # 001_P1_1_left_US

    sample_name = stem.replace("_US", "")
    # 001_P1_1_left

    label_path = label_dir / f"{sample_name}.nii"

    # comprobar existencia
    if not label_path.exists():

        print(f"\n❌ Missing label for:")
        print(us_path.name)

        continue

    out_dir = output_root / sample_name

    process_volume(
        us_path=us_path,
        label_path=label_path,
        out_dir=out_dir
    )

print("\n🎉 DONE")