import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchmetrics
from rich.console import Console
from rich.table import Table


class LitModel(L.LightningModule):
    """MNIST 분류를 위한 MLP 기반 LightningModule"""

    def __init__(
        self,
        input_size: int = 32,
        hidden_dim: int = 256,
        num_classes: int = 10,
        optimizer_config: dict = None,
        lr_scheduler_config: dict = None,
    ):
        super().__init__()

        self.optimizer_config = optimizer_config or {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01}
        self.lr_scheduler_config = lr_scheduler_config or {"name": "CosineAnnealingLR", "T_max": 10, "eta_min": 1e-6}
        self.save_hyperparameters()

        self.example_input_array = torch.randn(1, 1, self.hparams.input_size, self.hparams.input_size)

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hparams.input_size**2, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # 테스트용 메트릭
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes),
            torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="macro"),
            torchmetrics.Precision(task="multiclass", num_classes=self.hparams.num_classes, average="macro"),
            torchmetrics.Recall(task="multiclass", num_classes=self.hparams.num_classes, average="macro")
        ])
        self.test_metrics = metrics.clone(prefix="test_")
        self.conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)

    def configure_optimizers(self):
        """Optimizer와 Learning Rate Scheduler 설정"""
        optimizer_name = self.optimizer_config.get("name", "AdamW")
        lr = self.optimizer_config.get("lr", 0.001)
        weight_decay = self.optimizer_config.get("weight_decay", 0.01)

        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            momentum = self.optimizer_config.get("momentum", 0.9)
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler_name = self.lr_scheduler_config.get("name", "CosineAnnealingLR")

        if scheduler_name == "CosineAnnealingLR":
            T_max = self.lr_scheduler_config.get("T_max", 10)
            eta_min = self.lr_scheduler_config.get("eta_min", 1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_name == "StepLR":
            step_size = self.lr_scheduler_config.get("step_size", 5)
            gamma = self.lr_scheduler_config.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "metrics/val_loss",
                }
            }
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("metrics/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("metrics/learning_rate", current_lr, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log_dict({
            "metrics/val_loss": self.loss_fn(logits, y),
            "metrics/val_acc": (logits.argmax(1) == y).float().mean()
        }, prog_bar=True, sync_dist=True)

    def on_fit_end(self):
        """TensorBoard HPARAMS 탭에 최종 검증 지표 기록"""
        metrics = self.trainer.callback_metrics
        compare_metrics = {
            "stats/final_val_loss": metrics.get("metrics/val_loss", float('nan')),
            "stats/final_val_acc": metrics.get("metrics/val_acc", float('nan')),
        }

        if self.logger:
            self.logger.log_hyperparams(self.hparams, compare_metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).argmax(dim=1)

        self.test_metrics(preds, y)
        self.conf_mat(preds, y)

    def on_test_end(self):
        """테스트 결과 테이블 출력 및 Confusion Matrix 이미지 저장"""
        if self.trainer.is_global_zero:
            results = self.test_metrics.compute()

            console = Console()
            table = Table(title="[bold blue]Final Test Results[/bold blue]", show_header=True, header_style="bold magenta")
            table.add_column("Metric Name", style="cyan", justify="left")
            table.add_column("Value", style="green", justify="right")

            for name, value in results.items():
                clean_name = name.replace("test_Multiclass", "")
                table.add_row(clean_name, f"{value.item():.4f}")

            console.print("\n", table)

            cm = self.conf_mat.compute().cpu().numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=range(self.hparams.num_classes),
                        yticklabels=range(self.hparams.num_classes))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix: Test Evaluation')

            save_path = 'test_confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            console.print(f"[yellow]✔ Confusion Matrix saved as '{save_path}'[/yellow]\n")

        self.test_metrics.reset()
        self.conf_mat.reset()