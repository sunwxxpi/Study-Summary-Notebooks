import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class DataModule(L.LightningDataModule):
    """MNIST 데이터 로딩 및 전처리 모듈"""

    def __init__(self, img_size: int = 32, batch_size: int = 64, data_dir: str = "./data", num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose([
            transforms.Resize((self.hparams.img_size, self.hparams.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_ds = datasets.MNIST(self.hparams.data_dir, train=True, download=True, transform=self.transform)
            generator = torch.Generator().manual_seed(42)
            self.train_ds, self.val_ds = random_split(full_ds, [55000, 5000], generator=generator)
            
        if stage == "test" or stage is None:
            self.test_ds = datasets.MNIST(self.hparams.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)