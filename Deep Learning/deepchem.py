import torch
import torch.nn as nn
import torch.optim as optim
import deepchem as dc
from deepchem.models import TorchModel
from deepchem.feat import ConvMolFeaturizer

class GraphConvNet(nn.Module):
    def __init__(self, n_tasks, n_features, hidden_dim=128, dropout=0.2):
        super(GraphConvNet, self).__init__()
        self.conv1 = dc.models.torch_models.layers.GraphConv(n_features, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = dc.models.torch_models.layers.GraphConv(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dense = nn.Linear(hidden_dim, n_tasks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_list, degree_list):
        x = self.conv1(x, adjacency_list, degree_list)
        x = self.batch_norm1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x, adjacency_list, degree_list)
        x = self.batch_norm2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1)
        x = self.dense(x)
        
        return x

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=ConvMolFeaturizer())
train_dataset, valid_dataset, test_dataset = datasets

# 예측할 Task 수와 입력 Feature 수 정의
n_tasks = len(tasks)
n_features = train_dataset.X.shape[-1]

# PyTorch 기반의 Graph 신경망 모델 생성
model = GraphConvNet(n_tasks, n_features)

# Loss Function 및 Optimizer 지정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TorchModel로 Wrapping
model = TorchModel(model, loss=criterion, optimizer=optimizer)

# 모델 훈련
model.fit(train_dataset, nb_epoch=50, batch_size=32, max_checkpoints_to_keep=5, validation_dataset=valid_dataset, deterministic=True, seed=42)

# Pearson 상관 계수를 평가 지표로 설정
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("Train scores:", train_scores)
print("Validation scores:", valid_scores)
print("Test scores:", test_scores)

# Test Dataset의 첫 10개 샘플에 대해 용해도 예측
solubilities = model.predict_on_batch(test_dataset.X[:10])

# 예측된 용해도, 실제 용해도, 분자 ID를 출력
for molecule, solubility, test_solubility in zip(test_dataset.ids, solubilities, test_dataset.y):
    print(solubility, test_solubility, molecule)