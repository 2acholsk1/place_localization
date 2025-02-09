from pathlib import Path
import torch
import click


@click.command()
@click.option('--ckpt-path', type=click.Path(exists=True, file_okay=True, path_type=Path), required=True)
@click.option('--model-path-name', type=click.Path(file_okay=False, path_type=Path), required=True)
def export_model(ckpt_path: Path, model_path_name: Path):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    
    model_path = model_path_name.with_suffix(".pth")
    
    torch.save(checkpoint['state_dict'], model_path)



if __name__ == '__main__':
    export_model()