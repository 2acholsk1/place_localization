"""Created by Dominik Pieczynski"""
import random
import shutil
import zlib
from pathlib import Path

import click
import numpy as np
import tifffile
from PIL import Image
from joblib import delayed
from tifffile import TiffFileError

from place_localization.utils.joblib_tqdm import ProgressParallel


@click.command()
@click.argument('input-folder', type=click.Path(exists=True, path_type=Path))
@click.argument('output-folder', type=click.Path(path_type=Path))
@click.option('--quality', default=95, help='JPEG quality')
@click.option('--number-of-jobs', default=-2, help='Number of jobs')
def tiff_to_jpeg(input_folder: Path, output_folder: Path, quality: int, number_of_jobs: int):
    input_dirs = [current_input_dir for current_input_dir in input_folder.iterdir()]

    print(f'Converting {len(input_dirs)} places')
    ProgressParallel(n_jobs=number_of_jobs, batch_size=1, total=len(input_dirs))(
        delayed(convert)(current_input_dir, output_folder / current_input_dir.name, quality)
        for current_input_dir in input_dirs
    )

    # Remove output directories that have less than 5 images
    for current_output_dir in output_folder.iterdir():
        for image_path in current_output_dir.iterdir():
            try:
                image = Image.open(image_path)
                if image.size[0] < 768 or image.size[1] < 768:
                    print(f'Removing {image_path} because it is too small')
                    image_path.unlink()
            except Exception as e:
                print(f'Removing {image_path} because it is corrupted: {e}')

        if len(list(current_output_dir.iterdir())) < 5:
            print(f'Removing {current_output_dir} because it has less than 5 images')
            shutil.rmtree(current_output_dir)


# def convert(input_dir: Path, output_dir: Path, quality: int):
#     input_files = [file_path for file_path in input_dir.iterdir() if file_path.is_file()]
#     if len(input_files) < 5:
#         return

#     for file_path in random.sample(input_files, k=5):
#         output_path = output_dir / f'{file_path.stem}.jpg'
#         if output_path.exists():
#             continue

#         try:
#             image = tifffile.imread(str(file_path))
#         except (ValueError, TiffFileError, zlib.error):
#             file_path.unlink()
#             return

#         if not np.any(image):
#             return

#         # Convert to channels-last uint8
#         image = (image / image.max() * 255).astype('uint8')
#         image = image.transpose(1, 2, 0)

#         # Convert to JPEG
#         output_dir.mkdir(parents=True, exist_ok=True)
#         Image.fromarray(image).save(str(output_path), quality=quality)

def convert(input_dir: Path, output_dir: Path, quality: int):
    input_files = [file_path for file_path in input_dir.iterdir() if file_path.is_file()]
    if len(input_files) < 5:
        return

    for file_path in input_files:
        output_path = output_dir / f'{file_path.stem}.jpg'
        if output_path.exists():
            continue

        try:
            image = tifffile.imread(str(file_path))
        except (ValueError, TiffFileError, zlib.error):
            file_path.unlink()
            continue

        if not np.any(image):
            continue

        image = (image / image.max() * 255).astype('uint8')
        image = image.transpose(1, 2, 0)

        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(str(output_path), quality=quality)

if __name__ == '__main__':
    tiff_to_jpeg()