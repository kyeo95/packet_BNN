import torch
import torch.nn as nn
from torch.nn import Module, Conv1d,Conv2d, Linear
from torch.nn.functional import linear, conv2d, conv1d
from torch.utils.data import DataLoader
from os.path import join
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import numpy as np

import torch.nn as nn
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign().add_(1).div_(2)
    if quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def XNOR(A, B) :
    if A == 0 and B == 0 :
        return torch.tensor(1)
    if A == 0 and B == 1 :
        return torch.tensor(0)    
    if A == 1 and B == 0 :
        return torch.tensor(0)    
    if A == 1 and B == 1 :
        return torch.tensor(1)    
    
    
def initialize_W(Weight) :
    nn.init.kaiming_normal_(Weight, mode='fan_out')
    return 

def Bitcount(tensor) :
    tensor = tensor.type(torch.int8)
    activation = torch.zeros(1,1)
    
    count = torch.bincount(tensor)
    print("count = ",count)
    k = torch.tensor(4)
    #activation
    if count.size(dim=0) == 1:
        activation = torch.tensor([[0.]])        
    elif count[1] > k:
        activation = torch.tensor([[1.]])
    else :
        activation = torch.tensor([[0.]])
    print(activation)            
    return activation


# i : 0~119
def multiplication(Bitinput, Weight, bit):
    activation = torch.zeros(1,bit)
    precount = torch.zeros(bit,bit)
    Weight_B = Binarize(Weight)
    Weight_B = Weight_B.type(torch.int8)
    print("Weight_B = ", Weight_B)
    for i in range(0,bit) :
        for k in range(0,bit) :
             precount[i][k] = XNOR(Bitinput[0][k], Weight_B[i][k])
        #precount[i] = precount.type(torch.int8)
        activation[0][i] = Bitcount(precount[i])
        #activation = torch.cat((activation,new_activation),1)        
    return activation

def predict(activation, Weight, bit):
    precount = torch.zeros(bit)
    Weight_B = Binarize(Weight)
    Weight_B = Weight_B.type(torch.int8)
    for i in range(0,bit) :
        precount[i]= XNOR(activation[0][i], Weight_B[i])
    target = Bitcount(precount)
    
    return target
    
class Bnntrainer():
    def __init__(self, Weights, bit,lr = 0.01, device=None):
        super().__init__()
        self.Weights = Weights
        self.bit = bit
        self.lr = lr
        self.device = device
        
    def read(self):
        #read data

        return

    def train_step(self, criterion, optimizer,Bitinput):
        losses = []
        activation = multiplication(Bitinput, Weight, self.bit)
        predict_target= predict(activation, Weight[bit], self.bit)     
        loss = criterion(predict_target, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self.Weight -= self.lr * self.Weight.grad
        self.Weight.grad = None
        return losses

# activation은 sign으로
if __name__ == '__main__' :
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    bit = 10    
    Weight = torch.randn((bit+1, bit),requires_grad=True)
    initialize_W(Weight)
    print("Weight = ",Weight)
    Bitinput = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 0, 1]])
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    # sample input

    optimizer = torch.optim.Adam(BNN.parameters(), lr=0.01, weight_decay=1e-5)
        
    Bnn = Bnntrainer(Weights, bit = 10)
    Bnn.train_step(criterion,optimizer,Bitinput)

    # target = data load

    
    
    
