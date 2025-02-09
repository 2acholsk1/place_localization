import subprocess
from itertools import product
from pathlib import Path

import click
from PIL import Image
from joblib import delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from place_localization.utils.joblib_tqdm import ProgressParallel


@click.command()
@click.option('--data-path', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('--output-dir', type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option('--train-val-test-ratio', type=click.Tuple([float, float, float]),
              default=(0.7, 0.1, 0.2))
def train_val_test_split(data_path: Path, output_dir: Path, train_val_test_ratio: tuple[float, float, float]):
    places_dirs = sorted([place_dir for place_dir in data_path.iterdir() if place_dir.is_dir()])
    train_places_dirs, val_places_dirs = train_test_split(
        places_dirs, train_size=train_val_test_ratio[0], random_state=42
    )
    val_places_dirs, test_places_dirs = train_test_split(
        val_places_dirs, train_size=train_val_test_ratio[1] / (train_val_test_ratio[1] + train_val_test_ratio[2]),
        random_state=42
    )

    print(f'Train: {len(train_places_dirs)}')
    train_output_dir = output_dir / 'train'
    train_output_dir.mkdir(parents=True, exist_ok=True)
    for train_place_dir in tqdm(train_places_dirs):
        subprocess.run(['cp', '-r', '--reflink=auto', train_place_dir, train_output_dir])

    print(f'Val: {len(val_places_dirs)}')
    val_output_dir = output_dir / 'val'
    for val_place_dir in tqdm(val_places_dirs):
        subprocess.run(['cp', '-r', '--reflink=auto', val_place_dir, val_output_dir])

    # ProgressParallel(n_jobs=-2, total=len(val_places_dirs))(
    #     delayed(process_dir)(val_place_dir, val_output_dir) for val_place_dir in val_places_dirs
    # )

    print(f'Test: {len(test_places_dirs)}')
    test_output_dir = output_dir / 'test'
    for test_place_dir in tqdm(test_places_dirs):
        subprocess.run(['cp', '-r', '--reflink=auto', test_place_dir, test_output_dir])

    # ProgressParallel(n_jobs=-2, total=len(test_places_dirs))(
    #     delayed(process_dir)(test_place_dir, test_output_dir) for test_place_dir in test_places_dirs
    # )


def process_dir(place_dir: Path, output_dir: Path):
    images_paths = sorted([image_path for image_path in place_dir.iterdir() if image_path.is_file()])
    for val_image_path in images_paths:
        tile(val_image_path, output_dir, tile_size=1278)


def tile(image_path: Path, output_dir: Path, tile_size: int):
    place_id = image_path.parent.name
    image = Image.open(image_path)
    w, h = image.size

    grid = product(range(0, h - h % tile_size, tile_size), range(0, w - w % tile_size, tile_size))
    for i, j in grid:
        current_place_id = f'{place_id}_{i}_{j}'
        current_output_dir = output_dir / current_place_id
        current_output_dir.mkdir(parents=True, exist_ok=True)

        box = (j, i, j + tile_size, i + tile_size)
        image.crop(box).save(current_output_dir / image_path.name)


if __name__ == '__main__':
    train_val_test_split()
