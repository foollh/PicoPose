import torch
import torch.nn as nn
from torch import optim
from timm import scheduler
from pytorch_lightning.lite import LightningLite
import pytorch_lightning as pl



class Lite(pl.LightningModule):
    def __init__(self,
        network,
        loss,
        optimizer,
        lr_scheduler_config,
        dataloaders,
        **kwargs,
    ) -> None:
        super().__init__()

        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler_config = lr_scheduler_config
        self.dataloaders = dataloaders

    def train_dataloader(self):
        return self.dataloaders['train']

    def on_train_epoch_start(self) -> None:
        self.train_dataloader().dataset.reset()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        batch = self.network(batch)
        return batch

    def training_step_end(self, training_step_outputs):
        out_dicts = self.loss(training_step_outputs)
        return out_dicts
    
    def test_step(self, batch):
        batch = self.network(batch)
        return batch

    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     return super().on_train_batch_end(outputs, batch, batch_idx)

    def configure_optimizers(self, ):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler_config}

