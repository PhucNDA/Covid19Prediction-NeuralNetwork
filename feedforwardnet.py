import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# General
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNetwork,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,1)
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        return torch.sigmoid(out)

# 2Hidden Layers
class DeepNeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size1, hidden_size2):
        super(DeepNeuralNetwork,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size1)
        self.relu1=nn.ReLU()
        self.l2=nn.Linear(hidden_size1,hidden_size2)
        self.relu2=nn.ReLU()
        self.l3=nn.Linear(hidden_size2,1)
    def forward(self,x):
        out=self.l1(x)
        out=self.relu1(out)
        out=self.l2(out)
        out=self.relu2(out)
        out=self.l3(out)
        return torch.sigmoid(out)

# No ReLU
class NoRELU(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NoRELU,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.l2=nn.Linear(hidden_size,1)
    def forward(self,x):
        out=self.l1(x)
        out=self.l2(out)
        return torch.sigmoid(out)
# Tanh
class TanhNet(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(TanhNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.tanh=nn.Tanh()
        self.l2=nn.Linear(hidden_size,1)
    def forward(self,x):
        out=self.l1(x)
        out=self.tanh(out)
        out=self.l2(out)
        return torch.sigmoid(out)

# Without hiddenLayer
class NoHidden(nn.Module):
    def __init__(self,input_size):
        super(NoHidden,self).__init__()
        self.l1=nn.Linear(input_size,1)
    def forward(self,x):
        out=self.l1(x)
        return torch.sigmoid(out)