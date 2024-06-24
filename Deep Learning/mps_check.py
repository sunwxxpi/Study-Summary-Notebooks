import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 5)

    def forward(self, x):
            net = self.linear1(x)
            
            return net
        
print(torch.backends.mps.is_built()) # PyTorch가 미지원
print(torch.backends.mps.is_available()) # Mac이 미지원
        
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

x = torch.ones(5)
x = x.to(device)

model = Net()
model.to(device)

pred = model(x)
print(pred)