import torch
import albumentations as A
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

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
        self._place_transforms = place_transforms

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
            key = 'img' if i == 0 else f'img{i}'
            img_data = np.asarray(Image.open(img_path))
            min_height = min(min_height, img_data.shape[0])
            min_width = min(min_width, img_data.shape[1])
            images[key] = img_data
        
        transformed = self._place_transforms(**{
            key: img[:min_height, :min_width] for key, img in images.items()
        })
        
        for img in transformed.values():
            img = self._transforms(image=img)['img']
            selected_img.append(img)
            selected_img_place_ids.append(place_idx)
        
        selected_img, selected_img_place_ids = self._shuffle(selected_img, selected_img_place_ids)
        
        return torch.stack(selected_img), torch.tensor(selected_img_place_ids)
        
        

    @staticmethod
    def _shuffle(images: list[torch.Tensor], place_ids: list[int]) -> tuple[list[torch.Tensor], list[int]]:
        indices = torch.randperm(len(images))

        return [images[index] for index in indices], [place_ids[index] for index in indices]