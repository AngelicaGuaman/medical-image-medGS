import imageio.v3 as iio
import matplotlib.pyplot as plt

mask = iio.imread(
    "medgs_data/001_P1_1_left_multiclass/masks/0050.png"
)

plt.figure(figsize=(6,6))

plt.imshow(
    mask,
    cmap="tab10",
    vmin=0,
    vmax=3
)

plt.colorbar()

plt.title("Semantic Mask")

plt.show()