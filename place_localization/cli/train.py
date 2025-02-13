import hydra
import torch
import lightning.pytorch as pl
import lightning.pytorch.loggers
import lightning.pytorch.callbacks
from omegaconf import DictConfig
import lightning.pytorch.strategies

from place_localization.datamodules.triplet import TripletDatamodule
from place_localization.models.embedding import EmbeddingModel

def train(config: DictConfig):
    pl.seed_everything(42, workers=True)
    
    torch.set_float32_matmul_precision('medium')
    
    datamodule = TripletDatamodule(
        config.datamodule.data_path,
        config.datamodule.num_of_places_per_batch,
        config.datamodule.num_of_images_per_place,
        config.datamodule.num_of_batches_per_epoch,
        config.datamodule.augment,
        config.datamodule.validation_batch_size,
        config.datamodule.num_of_workers
    )

    model = EmbeddingModel(
        config.model.model_name,
        config.model.encoder_name,
        config.model.embedding_size,
        config.model.distance_name,
        config.model.miner_name,
        config.model.loss_function_name,
        config.model.lr,
        config.model.lr_patience,
        num_classes=datamodule.get_num_of_places('train')
    )

    model_summary_callback = lightning.pytorch.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(filename='{epoch}-{val_precision_at_1:.5f}', mode='max',
                                                       monitor='val_precision_at_1', verbose=True, save_last=True)
    early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(monitor='val_precision_at_1', mode='max', patience=10)
    lr_monitor = lightning.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch')

    logger = lightning.pytorch.loggers.NeptuneLogger(project=config.logger.project, log_model_checkpoints=config.logger.log_model_checkpoints)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            model_summary_callback,
            checkpoint_callback,
            early_stop_callback,
            lr_monitor
            ],
        accelerator='gpu',
        devices=-1,
        precision=config.trainer.precision,
        benchmark=True,
        sync_batchnorm=True,
        max_epochs=100,
        strategy=lightning.pytorch.strategies.DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True
        )
    )
    
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')
    
    logger.finalize('success')

@hydra.main(config_path='../../configs/', config_name='config.yaml', version_base=None)
def main(config: DictConfig):
    
    return train(config)


if __name__ == '__main__':
    main()