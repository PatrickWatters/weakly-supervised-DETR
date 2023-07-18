import os

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from numpy import random

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.len = num_samples
        self.data_size = [random.randint(2000, 20000) for _ in range(num_samples)]
        # self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return torch.randn(self.data_size[index], 8) * 255, torch.randint(100, (self.data_size[index], 2)), torch.ones(3, 10)
        # return self.data[index]

    def __len__(self):
        return self.len


def collate_fn_train(batches):
    features_batch = []
    coors_batch = []
    targets = []
    for i, (features, coors, target) in enumerate(batches):
        features_batch.append(features)
        coors[:, -1] = i
        coors_batch.append(coors)
        targets.append(target)
    return torch.cat(features_batch), torch.cat(coors_batch), torch.stack(targets)


class RandomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size):
        super().__init__()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self.batch_size, collate_fn=collate_fn_train, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self.batch_size, collate_fn=collate_fn_train, num_workers=4)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

num_samples = 10000

class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(8, 16)

    def forward(self, x):
        features, coors, targets = x
        y = self.layer(features)
        return y

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_val_data = RandomDataModule(RandomDataset(200), RandomDataset(20), batch_size=2)
    checkpoint_callback = ModelCheckpoint(monitor="valid_loss",
                                          mode="min",
                                          every_n_epochs=2,
                                          save_on_train_epoch_end=False)
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        strategy="ddp",
        accelerator="gpu",
        devices=4,
        num_sanity_val_steps=0,
        max_epochs=8,
        enable_model_summary=False,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_val_data)

if __name__ == "__main__":
    run()