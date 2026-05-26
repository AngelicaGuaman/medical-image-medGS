import json
from pathlib import Path
import imageio.v3 as iio
import matplotlib.pyplot as plt
import random

case_dir = Path("../medgs_data/001_P1_1_left")

with open(case_dir / "metadata.json", "r") as f:
    metadata = json.load(f)

sample = random.choice(metadata)

img = iio.imread(case_dir / sample["image"])
mask = iio.imread(case_dir / sample["mask"])

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray")
plt.imshow(mask, cmap="Reds", alpha=0.4)
plt.title(f"Slice {sample['index']} | t={sample['t']:.3f}")
plt.axis("off")
plt.show()