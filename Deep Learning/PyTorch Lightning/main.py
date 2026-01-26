import torch
from lightning.pytorch.cli import LightningCLI
from models import LitModel
from datamodules import DataModule

torch.set_float32_matmul_precision('high')

def cli_main():
    """
    LightningCLI를 통한 통합 실행부
    모델, 데이터, 최적화 설정을 YAML 파일로부터 주입받습니다.
    """
    cli = LightningCLI(
        model_class=LitModel,
        datamodule_class=DataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == "__main__":
    cli_main()