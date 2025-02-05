import lightning.pytorch as pl
import albumentations as A
import albumentations.pytorch.transforms
import timm.data
from pathlib import Path
from torch.utils.data import DataLoader

from place_localization.datasets.triplet import TripletDataset
from place_localization.datasets.evaluation import EvaluationDataset

class TripletDatamodule(pl.LightningDataModule):
    def __init__(self,
                 data_path: Path,
                 num_of_places_per_batch: int,
                 num_of_imgs_per_place: int,
                 num_of_batch_per_epoch: int,
                 augment: bool,
                 val_batch_size: int,
                 num_of_workers: int):
        super().__init__()

        self._data_path = data_path
        self._num_of_places_per_batch = num_of_places_per_batch
        self._num_imgs_per_place = num_of_imgs_per_place
        self._num_of_batch_per_epoch = num_of_batch_per_epoch
        self._val_batch_size = val_batch_size
        self._num_of_workers = num_of_workers

        self._transforms = A.Compose([
            A.CenterCrop(512, 512),
            A.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        
        self._augmentations = A.Compose([
            albumentations.Rotate(limit=10, p=1.0),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=1.0),
            albumentations.CenterCrop(512, 512),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ]) if augment else self._transforms

        self._place_augmentations = A.Compose([
            A.Affine(translate_percent=0.2, rotate=360, fit_output=True),
        ], additional_targets={f'image{i}': 'image' for i in range(1, num_of_imgs_per_place)})
        
        self.train_dataset = None
        self.val_dataset = None
        self.easy_test_dataset = None
        self.medium_test_dataset = None
        self.hard_test_dataset = None

    def setup(self, stage):
        train_places_dirs = self.get_places_dirs(Path(self._data_path) / 'train')
        val_places_dirs = self.get_places_dirs(Path(self._data_path) / 'val')
        easy_test_places_dirs = self.get_places_dirs(Path(self._data_path) / 'easy_test')
        medium_test_places_dirs = self.get_places_dirs(Path(self._data_path) / 'medium_test')
        hard_test_places_dirs = self.get_places_dirs(Path(self._data_path) / 'hard_test')
        
        print(f'Number of train places: {len(train_places_dirs)}')
        print(f'Number of val places: {len(val_places_dirs)}')
        print(f'Number of test places: {len(easy_test_places_dirs)}')

        self.train_dataset = TripletDataset(
            train_places_dirs,
            self._num_of_places_per_batch,
            self._num_imgs_per_place,
            self._num_of_batch_per_epoch,
            self._transforms,
            self._place_augmentations
        )

        self.val_dataset = EvaluationDataset(
            val_places_dirs,
            self._num_imgs_per_place,
            self._transforms
        )
        
        self.easy_test_dataset = EvaluationDataset(
            easy_test_places_dirs,
            self._num_imgs_per_place,
            self._transforms,
        )

        self.medium_test_dataset = EvaluationDataset(
            medium_test_places_dirs,
            self._num_imgs_per_place,
            self._transforms,
        )

        self.hard_test_dataset = EvaluationDataset(
            hard_test_places_dirs,
            self._num_imgs_per_place,
            self._transforms,
        )

    def get_places_dirs(self, data_dir: Path) -> list[Path]:
        return sorted(
            [place_dir for place_dir in data_dir.iterdir()
             if place_dir.is_dir() and len(list(place_dir.iterdir())) >= self._num_imgs_per_place]
        )

    def get_num_of_places(self, subset: str) -> int:
        assert subset in ['train', 'val', 'test']
        return len(self.get_places_dirs(Path(self._data_path) / subset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=1, num_workers=self._num_of_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._val_batch_size, num_workers=self._num_of_workers
        )

    def test_dataloader(self):
        return (
            DataLoader(
                self.easy_test_dataset, batch_size=self._val_batch_size, num_workers=self._num_of_workers,
            ),
            DataLoader(
                self.medium_test_dataset, batch_size=self._val_batch_size, num_workers=self._num_of_workers,
            ),
            DataLoader(
                self.hard_test_dataset, batch_size=self._val_batch_size, num_workers=self._num_of_workers,
            )
        )
