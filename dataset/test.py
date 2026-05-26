from torch.utils.data import DataLoader
from thyroid_dataset import ThyroidMedGSDataset

ds = ThyroidMedGSDataset("../medgs_data/001_P1_1_left")

loader = DataLoader(
    ds,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

batch = next(iter(loader))

print(batch["image"].shape)
print(batch["mask"].shape)
print(batch["t"].shape)