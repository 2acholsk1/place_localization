
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from pathlib import Path
import albumentations as A

from place_localization.models.embedding import EmbeddingModel
from place_localization.datasets.evaluation import EvaluationDataset


def load_model_from_ckpt(ckpt_path, device):
    model = EmbeddingModel.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model


def get_places_dirs(data_dir: Path, num_of_imgs_per_place: int):
    return sorted(
        [place_dir for place_dir in data_dir.iterdir()
         if place_dir.is_dir() and len(list(place_dir.iterdir())) >= num_of_imgs_per_place]
    )


def get_embeddings(model, dataloader, device):
    embeddings = []
    images = []
    image_paths = []

    with torch.no_grad():
        for batch in dataloader:
            x, _, place_idx, img_idx = batch
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            embeddings.append(y_pred)
            images.append(x.cpu().numpy())

            image_paths.append(dataset.get_path_by_index(place_idx.item(), img_idx.item()))

    embeddings = np.vstack(embeddings)
    images = np.vstack(images)

    return embeddings, images, image_paths


def find_nearest_neighbors(embeddings, index, k=5):
    distances = cdist([embeddings[index]], embeddings, metric='euclidean')[0]
    sorted_indices = np.argsort(distances)
    return sorted_indices[1:k+1]


import cv2

def save_comparison_image(images, indices, main_index, image_paths, save_dir="results"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 6, figsize=(15, 5))

    main_place = Path(image_paths[main_index]).parent

    axes[0].imshow(np.transpose(images[main_index], (1, 2, 0)))
    axes[0].set_title("Reference picture")
    axes[0].axis("off")

    print(f"🔹 Wybrany obraz: {image_paths[main_index]}")

    for i, idx in enumerate(indices):
        neighbor_place = Path(image_paths[idx]).parent

        border_color = "green" if neighbor_place == main_place else "red"

        img_with_border = np.transpose(images[idx], (1, 2, 0))
        img_with_border = cv2.copyMakeBorder(
            img_with_border, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
            value=(0, 255, 0) if border_color == "green" else (255, 0, 0)
        )

        axes[i + 1].imshow(img_with_border)
        axes[i + 1].set_title(f"Neighbour {i+1}")
        axes[i + 1].axis("off")

        print(f"🔸 Neighbour {i+1}: {image_paths[idx]} (Accuracy: {border_color.upper()})")

    output_filename = save_path / f"neighbors_{Path(image_paths[main_index]).stem}.png"
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close(fig)



import time
import random
from tqdm import tqdm
from torch.utils.data import Subset

def evaluate_model(model, dataloader, device, k=5, save_dir="results", num_samples=100, max_samples=400):
    total_samples = len(dataloader.dataset)

    selected_indices = random.sample(range(total_samples), min(max_samples, total_samples))

    subset_dataset = Subset(dataloader.dataset, selected_indices)
    subset_dataloader = DataLoader(subset_dataset, batch_size=dataloader.batch_size, shuffle=False)

    start_time = time.time()

    embeddings, images, image_paths = get_embeddings(model, subset_dataloader, device)

    sample_indices = random.sample(range(len(images)), num_samples)

    for i, main_index in enumerate(tqdm(sample_indices, desc="GENERATED", unit="picture")):
        iter_start = time.time()
        
        neighbors = find_nearest_neighbors(embeddings, main_index, k)
        save_comparison_image(images, neighbors, main_index, image_paths, save_dir)

        elapsed_time = time.time() - iter_start
        remaining_time = elapsed_time * (num_samples - (i + 1))

        print(f"✅ ENDED {i+1}/{num_samples} | REMAINING TIME: {remaining_time:.2f}s")

    total_time = time.time() - start_time
    print(f"🎉 DONE, TOTAL TIME: {total_time:.2f}s. SAVED IN {save_dir}")






if __name__ == '__main__':
    ckpt_path = ".neptune/PLAC-388/PLAC-388/checkpoints/epoch=18-val_precision_at_1=0.70361.ckpt"  # 🔹 Podaj ścieżkę do pliku .ckpt
    dataset_path = "naip_data/augmented/eval"  # 🔹 Podaj ścieżkę do datasetu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_ckpt(ckpt_path, device)

    num_of_imgs_per_place = 6
    places_dirs = get_places_dirs(Path(dataset_path), num_of_imgs_per_place)

    transforms = A.Compose([
        A.CenterCrop(512, 512),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        A.pytorch.transforms.ToTensorV2()
    ])

    dataset = EvaluationDataset(
        places_dirs=places_dirs,
        num_of_imgs_per_place=num_of_imgs_per_place,
        transforms=transforms,
        return_indices=True
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    evaluate_model(model, dataloader, device)
