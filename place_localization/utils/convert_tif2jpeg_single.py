import random
import shutil
import zlib
from pathlib import Path

import click
import numpy as np
import tifffile
from PIL import Image
from tifffile import TiffFileError

@click.command()
@click.argument('input-file', type=click.Path(exists=True, path_type=Path))
@click.argument('output-file', type=click.Path(path_type=Path))
@click.option('--quality', default=95, help='JPEG quality')
def tiff_to_jpeg(input_file: Path, output_file: Path, quality: int):
    print(f'Converting {input_file} to {output_file}')
    convert(input_file, output_file, quality)


def convert(input_file: Path, output_file: Path, quality: int):
    try:
        image = tifffile.imread(str(input_file))
    except (ValueError, TiffFileError, zlib.error) as e:
        print(f'Error reading {input_file}: {e}')
        return

    if not np.any(image):
        print(f'{input_file} is empty or invalid.')
        return

    image = (image / image.max() * 255).astype('uint8')
    image = image.transpose(1, 2, 0)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(str(output_file), quality=quality)
    print(f'Conversion successful! Saved to {output_file}')


if __name__ == '__main__':
    tiff_to_jpeg()

