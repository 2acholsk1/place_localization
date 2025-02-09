from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import torch
from lightning import Trainer
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from place_localization.datamodules.triplet import TripletDataModule
from place_localization.models.embedding import EmbeddingModel


@click.command()
@click.argument('model-path', type=click.Path(exists=True, path_type=Path))
@click.argument('data-path', type=click.Path(exists=True, path_type=Path))
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cuda')
def evaluate(model_path: Path, data_path: Path, device: str):
    model = EmbeddingModel.load_from_checkpoint(model_path).eval()
    data_module = TripletDataModule(
        data_path=data_path,
        number_of_places_per_batch=30,
        number_of_images_per_place=2,
        number_of_batches_per_epoch=1,
        augment=False,
        validation_batch_size=32,
        number_of_workers=8
    )

    trainer = Trainer(accelerator=device)
    predictions = trainer.predict(model, datamodule=data_module)

    places_embeddings = defaultdict(lambda: [])
    places_images = defaultdict(lambda: [])
    for embedding, place_label, place_index, image_index in predictions:
        image_path = data_module.predict_dataset.get_path_by_index(place_index, image_index)
        places_embeddings[place_label].append(embedding)
        places_images[place_label].append(image_path)

    query_vectors = []
    reference_vectors = []
    reference_labels = []
    with torch.inference_mode():
        for data_batch in data_module.val_dataloader():
            images, labels = data_batch
            images = images.to(model.device)
            labels = labels.to(model.device)

            embeddings = model(images)
            embeddings = embeddings.detach().cpu().numpy()

            query_vectors.append(embeddings[::2])
            reference_vectors.append(embeddings[1::2])
            reference_labels.append(labels[1::2].detach().cpu().numpy())

    query_vectors = np.stack(query_vectors)
    query_labels = np.arange(len(query_vectors))
    reference_vectors = np.stack(reference_vectors)
    reference_labels = np.array(reference_labels)

    calculator = AccuracyCalculator(include=('precision_at_1',))
    print(calculator.get_accuracy(query_vectors, reference_vectors, query_labels, reference_labels, False))


if __name__ == '__main__':
    evaluate()