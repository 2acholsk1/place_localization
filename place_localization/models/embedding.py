import timm
import torch
import torch.nn.functional as F
from torch import nn
from lightning import pytorch as pl
from torchmetrics import MetricCollection
from typing import Optional

from place_localization.metrics.multi import MultiMetric
from place_localization.models.gem import GeM
from place_localization.models.utils import get_distance, get_miner, get_loss_function

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
                                             )
                self.network = nn.Sequential(
                    backbone,
                    Normalize(),
                    GeM(),
                    nn.Flatten(),
                    nn.Linear(in_features=backbone.num_features, out_features=embedding_size)
                )
            
            case _:
                raise NotImplementedError(f'Unsupported model: {model_name}')

        distance = get_distance(dist_name)
        self.miner = get_miner(miner_name, distance)
        self.loss = get_loss_function(loss_func_name, distance, num_classes, embedding_size)

        self.val_outputs = None
        self.test_outputs = None
        
        metrics = MetricCollection(MultiMetric(distance=distance))
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y, self.miner(y_pred, y))
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.forward(x)
        self.val_outputs['preds'].append(y_pred.cpu())
        self.val_outputs['targets'].append(y.cpu())

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.forward(x)
        self.test_outputs['preds'].append(y_pred.cpu())
        self.test_outputs['targets'].append(y.cpu())

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = {
            'preds': [],
            'targets': [],
        }

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat(self.val_outputs['preds'], dim=0)
        targets = torch.cat(self.val_outputs['targets'], dim=0)
        self.log_dict(self.val_metrics(preds, targets), sync_dist=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = {
            'preds': [],
            'targets': [],
        }
    
    def on_test_epoch_end(self) -> None:
        preds = torch.cat(self.test_outputs['preds'], dim=0)
        targets = torch.cat(self.test_outputs['targets'], dim=0)
        self.log_dict(self.test_metrics(preds, targets), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


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