# PyTorch Lightning + Hydra 프로젝트

Hydra 기반 설정 관리를 사용하는 PyTorch Lightning 프로젝트입니다.

## 설치

```bash
uv sync
```

## 빠른 시작

```bash
# 학습
python main.py

# 학습 후 자동 테스트
python main.py mode=fit_and_test

# 테스트만 실행
python main.py mode=test ckpt_path=outputs/YYYY-MM-DD/HH-MM-SS/version_0/checkpoints/best-acc-epoch=XX.ckpt
```

## 명령어 모음

### 디버깅

```bash
# 코드 동작 확인 (1 batch만 실행)
python main.py trainer.fast_dev_run=1

# 짧은 학습 테스트
python main.py trainer.max_epochs=1

# CPU로 실행
python main.py trainer.accelerator=cpu
```

### 하이퍼파라미터 변경

```bash
# 학습률
python main.py optimizer.lr=0.0001

# Optimizer
python main.py optimizer.name=Adam
python main.py optimizer.name=SGD optimizer.lr=0.01 optimizer.momentum=0.9

# 배치 크기
python main.py data.batch_size=256

# 모델 크기
python main.py model.hidden_dim=1024

# 에폭 수
python main.py trainer.max_epochs=30

# 시드
python main.py seed_everything=777
```

### Learning Rate Scheduler

```bash
# CosineAnnealingLR
python main.py lr_scheduler.name=CosineAnnealingLR lr_scheduler.T_max=10

# StepLR
python main.py lr_scheduler.name=StepLR lr_scheduler.step_size=5 lr_scheduler.gamma=0.1

# ReduceLROnPlateau
python main.py lr_scheduler.name=ReduceLROnPlateau
```

### Precision

```bash
# Mixed Precision (FP16)
python main.py trainer.precision=16-mixed

# BFloat16 (A100, H100 등)
python main.py trainer.precision=bf16-mixed
```

### 체크포인트에서 학습 재개

```bash
python main.py ckpt_path=outputs/YYYY-MM-DD/HH-MM-SS/version_0/checkpoints/best-acc-epoch=XX.ckpt
```

### 출력 디렉토리 커스터마이징

```bash
# 실험 이름 지정
python main.py hydra.run.dir=outputs/exp_baseline

# 고정 폴더 (매번 덮어쓰기)
python main.py hydra.run.dir=outputs/latest
```

### 설정 확인

```bash
# 전체 설정 출력
python main.py --cfg job

# Hydra 도움말
python main.py --help
```

### Hydra 멀티런 (자동 실험)

```bash
# Hidden dimension 비교 (4번 실행)
python main.py --multirun model.hidden_dim=128,256,512,1024

# Learning rate 그리드 서치
python main.py --multirun optimizer.lr=0.0001,0.001,0.01

# 조합 실험 (3x3=9번 실행)
python main.py --multirun model.hidden_dim=256,512,1024 optimizer.lr=0.0001,0.001,0.01
```

### TensorBoard

```bash
# 전체 실험 모니터링
tensorboard --logdir outputs

# 외부 접속 허용
tensorboard --logdir outputs --host 0.0.0.0
```

브라우저에서 `http://localhost:6006` 접속

## 실전 워크플로우

```bash
# 1단계: 빠른 디버깅
python main.py trainer.fast_dev_run=1

# 2단계: 짧은 학습으로 검증
python main.py trainer.max_epochs=2

# 3단계: 하이퍼파라미터 탐색 (멀티런)
python main.py --multirun optimizer.lr=0.0001,0.001,0.01 model.hidden_dim=256,512

# 4단계: 최적 설정으로 전체 학습
python main.py optimizer.lr=0.001 model.hidden_dim=512 trainer.max_epochs=50

# 5단계: TensorBoard로 결과 확인
tensorboard --logdir outputs
```

## 출력 구조

```
outputs/
  YYYY-MM-DD/
    HH-MM-SS/
      .hydra/
        config.yaml
        overrides.yaml
      version_0/
        events.out.tfevents.*
        checkpoints/
          best-loss-epoch=XX.ckpt
          best-acc-epoch=XX.ckpt
    test_confusion_matrix.png
```

## 파일 구조

```
.
├── main.py          # Hydra 기반 메인 스크립트
├── config.yaml      # 전체 설정 파일
├── models.py        # LightningModule 정의
├── datamodules.py   # LightningDataModule 정의
├── pyproject.toml   # 의존성 관리
└── README.md        # 이 파일
```

## 설정 파일 (config.yaml)

| 섹션 | 설명 |
|------|------|
| `mode` | 실행 모드 (fit, test, fit_and_test) |
| `model` | 모델 구조 설정 |
| `data` | 데이터 로더 설정 |
| `trainer` | Trainer 옵션 |
| `optimizer` | Optimizer (AdamW, Adam, SGD) |
| `lr_scheduler` | LR Scheduler (CosineAnnealingLR, StepLR, ReduceLROnPlateau) |
| `logger` | TensorBoard 설정 |
| `checkpoint` | 체크포인트 저장 설정 |