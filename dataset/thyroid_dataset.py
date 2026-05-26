import json
from pathlib import Path

import imageio.v3 as iio
import torch
from torch.utils.data import Dataset


class ThyroidMedGSDataset(Dataset):
    def __init__(self, case_dir):
        self.case_dir = Path(case_dir)

        with open(self.case_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        image = iio.imread(self.case_dir / item["image"])
        mask = iio.imread(self.case_dir / item["mask"])

        image = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(mask).float() / 255.0

        image = image.unsqueeze(0)  # [1,H,W]
        mask = mask.unsqueeze(0)    # [1,H,W]

        t = torch.tensor([item["t"]], dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "t": t,
            "index": item["index"],
            "original_z": item["original_z"],
        }