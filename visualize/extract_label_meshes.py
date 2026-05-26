import nibabel as nib
import numpy as np
from skimage import measure
import trimesh
from pathlib import Path

label_dir = Path(
    "US_data/US_volunteer_dataset/ground_truth_data/US_thyroid_label"
)

label_files = sorted(
    list(label_dir.glob("*.nii")) +
    list(label_dir.glob("*.nii.gz"))
)

def strip_nii_extension(path: Path) -> str:
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem

def process_label_volume(label_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(label_path)
    lbl = img.get_fdata().astype(np.uint8)
    spacing = img.header.get_zooms()[:3]

    print("Shape:", lbl.shape)
    print("Spacing:", spacing)
    print("Classes:", np.unique(lbl))

    for cls in np.unique(lbl):
        if cls == 0:
            continue

        binary = lbl == cls
        voxels = int(binary.sum())

        if voxels < 100:
            print(f"Skipping class {cls}, too small")
            continue

        verts, faces, normals, values = measure.marching_cubes(
            binary.astype(np.float32),
            level=0.5,
            spacing=spacing
        )

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()

        out_path = out_dir / f"class_{int(cls)}.ply"
        mesh.export(out_path)

        print(f"Saved {out_path} | voxels={voxels} | vertices={len(mesh.vertices)} | faces={len(mesh.faces)}")


print(f"\n🔍 Found {len(label_files)} label volumes")
out_dir = Path("meshes")
out_dir.mkdir(exist_ok=True)

for label_path in label_files:

    stem = strip_nii_extension(label_path)

    sample_name = stem
    process_label_volume(label_path, out_dir / sample_name)


