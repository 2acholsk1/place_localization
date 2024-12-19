import torch
import albumentations as A
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self,
                 places_dirs: list[Path],
                 num_of_places_per_batch: int,
                 num_of_imgs_per_place: int,
                 num_of_batches_per_epoch: int,
                 transorms: A.Compose,
                 place_transforms: A.Compose):
        super().__init__()

        self._places_images: list[list[Path]] = [
            sorted([image_path for image_path in place_dir.iterdir() if image_path.is_file()])
            for place_dir in places_dirs
        ]

        self._num_of_places = num_of_places_per_batch
        self._num_of_imgs_per_place = num_of_imgs_per_place
        self._num_of_batches_per_epoch = num_of_batches_per_epoch
        self._transforms = transorms
        self._

    def __len__(self) -> int:
        return self._num_of_batches_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        selected_places_ids = torch.randperm(len(self._places_images))[:self._num_of_places]
        selected_img = []
        selected_img_place_ids = []
        for place_idx in selected_places_ids:
            place_imgs = self._places_images[place_idx]
            selected_img_ids = torch.randperm(len(place_imgs))[:self._num_of_imgs_per_place]

        images = {}
        min_height, min_width = np.inf, np.inf
        for i, img_idx in enumerate(selected_img_ids):
            img_path = place_imgs[img_idx]
            key 