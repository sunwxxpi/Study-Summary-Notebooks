import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from datamodules import DataModule
from models import LitModel

torch.set_float32_matmul_precision('high')


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Hydra 기반 학습 파이프라인"""
    L.seed_everything(cfg.seed_everything)

    model = LitModel(
        input_size=cfg.model.input_size,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
        optimizer_config=OmegaConf.to_container(cfg.optimizer, resolve=True),
        lr_scheduler_config=OmegaConf.to_container(cfg.lr_scheduler, resolve=True),
    )

    datamodule = DataModule(
        img_size=cfg.data.img_size,
        batch_size=cfg.data.batch_size,
        data_dir=cfg.data.data_dir,
        num_workers=cfg.data.num_workers,
    )

    # Callbacks 설정
    callbacks = []
    checkpoint_val_loss = None
    checkpoint_val_acc = None

    if cfg.trainer.get('use_callbacks', True):
        checkpoint_val_loss = ModelCheckpoint(
            filename=cfg.checkpoint.filename_loss,
            monitor=cfg.checkpoint.monitor_loss,
            mode=cfg.checkpoint.mode_loss,
        )
        callbacks.append(checkpoint_val_loss)

        checkpoint_val_acc = ModelCheckpoint(
            filename=cfg.checkpoint.filename_acc,
            monitor=cfg.checkpoint.monitor_acc,
            mode=cfg.checkpoint.mode_acc,
        )
        callbacks.append(checkpoint_val_acc)

        callbacks.append(ModelSummary(max_depth=cfg.model_summary.max_depth))

    # Logger 설정
    logger = None
    if cfg.trainer.get('use_logger', True):
        logger = TensorBoardLogger(
            save_dir=cfg.logger.save_dir,
            name=cfg.logger.name,
            default_hp_metric=cfg.logger.default_hp_metric,
        )

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        deterministic=cfg.trainer.deterministic,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=cfg.trainer.fast_dev_run,
    )

    if cfg.mode == 'fit':
        trainer.fit(model, datamodule, ckpt_path=cfg.get('ckpt_path'))

    elif cfg.mode == 'test':
        trainer.test(model, datamodule, ckpt_path=cfg.ckpt_path)

    elif cfg.mode == 'fit_and_test':
        trainer.fit(model, datamodule, ckpt_path=cfg.get('ckpt_path'))

        # 지정된 기준(acc 또는 loss)에 맞는 best 체크포인트로 테스트
        test_checkpoint_type = cfg.get('test_checkpoint', 'acc')

        if test_checkpoint_type == 'acc':
            if checkpoint_val_acc is None:
                raise ValueError("Accuracy checkpoint not available. Set trainer.use_callbacks=true")
            test_ckpt = checkpoint_val_acc.best_model_path
            print(f"\n[Testing with BEST ACCURACY checkpoint: {test_ckpt}]\n")
        else:
            if checkpoint_val_loss is None:
                raise ValueError("Loss checkpoint not available. Set trainer.use_callbacks=true")
            test_ckpt = checkpoint_val_loss.best_model_path
            print(f"\n[Testing with BEST LOSS checkpoint: {test_ckpt}]\n")

        trainer.test(model, datamodule, ckpt_path=test_ckpt)

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use 'fit', 'test', or 'fit_and_test'")


if __name__ == "__main__":
    main()