import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class DataModule(L.LightningDataModule):
    """
    데이터 전처리 및 로더 관리를 담당하는 모듈
    데이터셋 다운로드부터 분할, Transform 적용을 수행
    """
    def __init__(self, img_size: int = 32, batch_size: int = 64, data_dir: str = "./data"):
        super().__init__()
        self.save_hyperparameters()
        
        # 기본 전처리 설정
        self.transform = transforms.Compose([
            transforms.Resize((self.hparams.img_size, self.hparams.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        """단계별 데이터셋 로드 (Train/Val/Test 분할)"""
        if stage == "fit" or stage is None:
            full_ds = datasets.MNIST(self.hparams.data_dir, train=True, download=True, transform=self.transform)
            self.train_ds, self.val_ds = random_split(full_ds, [55000, 5000])
            
        if stage == "test" or stage is None:
            self.test_ds = datasets.MNIST(self.hparams.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self): 
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True)
    
    def val_dataloader(self): 
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=8)
    
    def test_dataloader(self): 
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=8)