import torch
import numpy as np
import albumentations as A
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class EvaluationDataset(Dataset):
    def __init__(self,
                 places_dirs: list[Path],
                 num_of_imgs_per_place: int,
                 transforms: A.Compose):
        super().__init__()
        
        self._places_images: list[list[Path]] = [
            sorted([img_path for img_path in place_dir.iterdir() if img_path.is_file()])
            for place_dir in places_dirs
        ]
        
        self._num_of_imgs_per_place = num_of_imgs_per_place
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._places_images) * self._num_of_imgs_per_place

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        place_idx = idx // self._num_of_imgs_per_place
        img_idx = idx % self._num_of_imgs_per_place
        img_path = self._places_images[place_idx][img_idx]
        
        img = np.asarray(Image.open(img_path))
        img = self._transforms(image=img)['image']
        
        return img, torch.tensor(place_idx)