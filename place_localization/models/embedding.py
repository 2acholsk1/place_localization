import timm
import torch
import torch.nn.functional as F
from torch import nn
from lightning import pytorch as pl

from place_localization.models.gem import GeM

class EmbeddingModel(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 encoder_name: str,
                 embedding_size: int,
                 dist_name: str,
                 miner_name: str,
                 loss_func_name: str,
                 lr: float,
                 lr_patience: int,
                 num_classes: int):
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.lr_patience = lr_patience

        match(model_name):
            case 'Basic':
                backbone = timm.create_model(encoder_name,
                                             pretrained=True,
                                             num_classes=0,
                                             global_pool='',
                                             img_size=(512, 512))
                self.network = nn.Sequential(
                    backbone,
                    Normalize(),
                    GeM(),
                    nn.Flatten(),
                    nn.Linear(in_features=backbone.num_features, out_features=embedding_size)
                )
            
            case _:
                raise NotImplementedError(f'Unsupported model: {model_name}')

        distance = get_

class Normalize(nn.Module):
    def __init__(self, order: int = 2, dim: int | tuple[int, ...] = 1):
        super().__init__()

        self._order = order
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=self._order, dim=self._dim)


if __name__ == '__main__':
    def check_model_output():
        model = EmbeddingModel(model_name='Basic', encoder_name='resnet18', lr=3e-4, lr_patience=10)
        print(model(torch.randn(4, 3, 224, 224)).shape)

    check_model_output()