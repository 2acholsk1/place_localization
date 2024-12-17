import torch
import albumentations as A
from pathlib import Path
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, images_paths: list[Path], transorms: A.Compose):
        self._images_paths = images_paths
        self._transforms = transorms

    def __len__(self) -> int:
        return len(self._images_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__getitem__(index)