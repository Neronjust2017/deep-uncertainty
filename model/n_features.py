from .model import CNN
import torch
model = CNN(input_dim=1, output_dim=1, filters=[256,256], kernels=[3,3], activation='ReLU')
x = torch.randn(48, 1, 36)
model(x)