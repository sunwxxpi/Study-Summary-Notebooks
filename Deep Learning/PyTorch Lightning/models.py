import torch
import torch.nn as nn
import lightning as L
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

class LitModel(L.LightningModule):
    """
    [딥러닝 연구용 표준 LightningModule]
    모델 구조 정의, 학습/검증/테스트 루프, 그리고 최종 성능 리포팅을 통합 관리합니다.
    """
    def __init__(self, input_size: int = 32, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        
        # 하이퍼파라미터 저장: YAML 설정값들을 self.hparams로 접근 가능하게 하며 체크포인트에 자동 포함시킵니다.
        self.save_hyperparameters()
        
        # 입출력 크기 확인용 샘플: 모델 초기화 시 텐서 크기 불일치 오류를 미리 확인하기 위한 더미 입력입니다.
        self.example_input_array = torch.randn(1, 1, self.hparams.input_size, self.hparams.input_size)
        
        # 모델 아키텍처: Flatten을 통해 다차원 영상을 1차원 벡터로 변환 후 MLP 레이어를 통과시킵니다.
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hparams.input_size**2, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # 과적합 방지를 위한 드롭아웃 레이어
            nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes)
        )
        
        # 손실 함수: 다중 클래스 분류를 위한 Cross Entropy Loss를 정의합니다.
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 연구용 메트릭 설정: Accuracy 외에도 F1-Score, Precision, Recall을 함께 측정합니다.
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes),
            torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="macro"),
            torchmetrics.Precision(task="multiclass", num_classes=self.hparams.num_classes, average="macro"),
            torchmetrics.Recall(task="multiclass", num_classes=self.hparams.num_classes, average="macro")
        ])
        
        # 테스트 전용 메트릭 객체 생성
        self.test_metrics = metrics.clone(prefix="test_")
        
        # Confusion Matrix: 모델이 어떤 클래스를 헷갈려 하는지 정밀 분석하기 위한 혼동 행렬을 정의합니다.
        self.conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        """추론(Inference) 단계에서 입력 데이터를 모델에 통과시켜 로짓(Logits)을 반환합니다."""
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        """학습 루프: 오차(Loss)를 계산하고 역전파를 위한 그래디언트를 생성합니다."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # 학습 손실 기록: TensorBoard의 Scalars 탭에서 'metrics' 그룹으로 묶어 시각화합니다.
        self.log("metrics/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """검증 루프: 학습 도중 모델의 일반화 성능을 모니터링하여 Best Model을 선별합니다."""
        x, y = batch
        logits = self(x)
        
        metrics = {
        "metrics/val_loss": self.loss_fn(logits, y),
        "metrics/val_acc": (logits.argmax(1) == y).float().mean()
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def on_fit_end(self):
        """학습 종료 훅: 실험 관리를 위해 TensorBoard의 HPARAMS 탭에 최종 검증 지표를 기록합니다."""
        metrics = self.trainer.callback_metrics
        final_val_loss = metrics.get("metrics/val_loss", float('nan'))
        final_val_acc = metrics.get("metrics/val_acc", float('nan'))

        # 하이퍼파라미터와 성능의 상관관계를 분석하기 위한 데이터 매핑
        compare_metrics = {
            "stats/final_val_loss": final_val_loss,
            "stats/final_val_acc": final_val_acc,
        }

        if self.logger:
            self.logger.log_hyperparams(self.hparams, compare_metrics)

    def test_step(self, batch, batch_idx):
        """최종 테스트 단계: 모델의 최종 성능을 측정합니다."""
        x, y = batch
        preds = self(x).argmax(dim=1)
        
        self.test_metrics(preds, y)
        self.conf_mat(preds, y)
        
    def on_test_end(self):
        """테스트 종료 훅: 모든 테스트 데이터에 대한 최종 지표를 테이블과 이미지로 리포팅합니다."""
        
        # 지표 연산: 누적된 데이터를 바탕으로 최종 수치를 도출합니다.
        results = self.test_metrics.compute()
        
        # Rich 테이블 출력: 터미널에 소수점 4째 자리까지 포맷팅된 깔끔한 결과표를 생성합니다.
        console = Console()
        table = Table(title="[bold blue]Final Test Results[/bold blue]", show_header=True, header_style="bold magenta")
        table.add_column("Metric Name", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")

        for name, value in results.items():
            # 가독성을 위해 불필요한 접두어를 제거하고 행을 추가합니다.
            clean_name = name.replace("test_Multiclass", "")
            table.add_row(clean_name, f"{value.item():.4f}")

        console.print("\n", table)

        # Confusion Matrix 시각화: Seaborn을 이용해 히트맵을 생성하고 고해상도 이미지로 저장합니다.
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

        # 리소스 초기화: 다음 실험이나 반복 실행 시 데이터가 섞이지 않도록 버퍼를 비웁니다.
        self.test_metrics.reset()
        self.conf_mat.reset()