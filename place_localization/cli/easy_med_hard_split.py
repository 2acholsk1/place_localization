import subprocess
from pathlib import Path

import albumentations
import click
import numpy as np
from PIL import Image
from joblib import delayed

from place_localization.utils.joblib_tqdm import ProgressParallel


@click.command()
@click.option('--split-data-path', type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True)
@click.option('--output-dir', type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option('--quality', default=95, help='JPEG quality')
def easy_med_hard_split(split_data_path: Path, output_dir: Path, quality: int):
    # print('Copying train')
    # subprocess.run(['cp', '-r', '--reflink=auto', split_data_path / 'train',  output_dir / 'train'])

    print('Copying test')
    subprocess.run(['cp', '-r', '--reflink=auto', split_data_path / 'test',  output_dir / 'easy_test'])

    medium_augmentations = albumentations.Compose([
        albumentations.CenterCrop(1897, 1897),
        albumentations.OneOf([
            albumentations.Rotate(limit=(-5, -1), p=1.0),
            albumentations.Rotate(limit=(1, 5), p=1.0),
        ], p=1.0),
        albumentations.OneOf([
            albumentations.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, -0.01), p=1.0),
            albumentations.Affine(scale=(0.95, 1.05), translate_percent=(0.01, 0.05), p=1.0),
        ], p=1.0),
        albumentations.CenterCrop(512, 512),
    ])
    hard_augmentations = albumentations.Compose([
        albumentations.CenterCrop(1897, 1897),
        albumentations.OneOf([
            albumentations.Rotate(limit=(-10, -5), p=1.0),
            albumentations.Rotate(limit=(5, 10), p=1.0),
        ], p=1.0),
        albumentations.OneOf([
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, -0.05), p=1.0),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.05), p=1.0),
        ], p=1.0),
        albumentations.CenterCrop(512, 512),
    ])

    # print('Processing val')
    # val_places_dirs = sorted((split_data_path / 'val').iterdir())
    # ProgressParallel(n_jobs=-2, total=len(val_places_dirs))(
    #     delayed(process_val_dir)(val_place_dir, output_dir, hard_augmentations, quality)
    #     for val_place_dir in val_places_dirs
    # )

    # print('Processing test')
    # test_places_dirs = sorted((split_data_path / 'test').iterdir())
    # ProgressParallel(n_jobs=-2, total=len(test_places_dirs))(
    #     delayed(process_test_dir)(test_place_dir, output_dir, medium_augmentations, hard_augmentations, quality)
    #     for test_place_dir in test_places_dirs
    # )


def process_test_dir(place_dir: Path, output_dir: Path,
                     medium_augmentations: albumentations.Compose, hard_augmentations: albumentations.Compose,
                     quality: int):
    current_medium_output_dir = output_dir / 'medium_test' / place_dir.name
    current_medium_output_dir.mkdir(parents=True, exist_ok=True)
    current_hard_output_dir = output_dir / 'hard_test' / place_dir.name
    current_hard_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(place_dir.iterdir())[:5]:
        original_image = np.asarray(Image.open(image_path))
        medium_image = medium_augmentations(image=original_image)['image']
        hard_image = hard_augmentations(image=original_image)['image']
        Image.fromarray(medium_image).save(current_medium_output_dir / image_path.name, quality=quality)
        Image.fromarray(hard_image).save(current_hard_output_dir / image_path.name, quality=quality)


def process_val_dir(place_dir: Path, output_dir: Path,
                    hard_augmentations: albumentations.Compose,
                    quality: int):
    current_output_dir = output_dir / 'val' / place_dir.name
    current_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(place_dir.iterdir())[:5]:
        image = np.asarray(Image.open(image_path))
        image = hard_augmentations(image=image)['image']
        Image.fromarray(image).save(current_output_dir / image_path.name, quality=quality)


if __name__ == '__main__':
    easy_med_hard_split()