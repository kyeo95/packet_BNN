#malicious한 IP로 왔는지 source를 대조
import torch
import os
import torch.nn as nn
from torch.nn import Module, Conv1d, Conv2d, Linear
from torch.nn.functional import linear, conv2d, conv1d
from torch.utils.data import DataLoader
from os.path import join
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import numpy as np
import labeling
import torch.nn as nn
from torch.autograd import Variable
from torch import save, no_grad
from kamene.all import *
import sys

__all__ = ['packetbnn']

class Packetbnn(nn.Module):

    def __init__(self, num_classes=1):
        super(Packetbnn, self).__init__()

        self.features = nn.Sequential(

            BNNConv2d(1, 120, kernel_size=(1,120), stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(120),
            nn.Softsign(),

            nn.Flatten(),
            #nn.BatchNorm1d(120),
            # nn.Hardtanh(inplace=True),
            BNNLinear(120, num_classes),
            #nn.BatchNorm1d(num_classes, affine=False),
            #nn.LogSoftmax(dim=1),
            #1개 데이터용 주석처리
        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.uniform_(m.weight, a= 0., b= 1.)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.5, 0.01)
                nn.init.zeros_(m.bias)
        return

def packetbnn(num_classes=1):
    return Packetbnn(num_classes)

def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        result = (tensor-0.5).sign().add_(1).div_(2)
        return result
    if quant_mode == 'bin':
        return (tensor >= 0).type(type(tensor)) * 2 - 1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)

class BNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(BNNLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        #if (input.size(1) != 784) and (input.size(1) != 3072):
        input.data = Binarize(input.data)

        self.weight.data = Binarize(self.weight_org)
        out = linear(input, self.weight.data)

        # if not self.bias is None:
        #     self.bias.org = self.bias.data.clone()
        #     out += self.bias.view(1, -1).expand_as(out)

        return out


class BNNConv2d(Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BNNConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        input.data = Binarize(input.data)

        self.weight.data = Binarize(self.weight_org)

        out = conv2d(input, self.weight.data, None, self.stride,
                     self.padding, self.dilation, self.groups)

        # if not self.bias is None:
        #     self.bias.org = self.bias.data.clone()
        #     out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def XNOR(A, B):
    if A == 0 and B == 0:
        return torch.tensor(1)
    if A == 0 and B == 1:
        return torch.tensor(0)
    if A == 1 and B == 0:
        return torch.tensor(0)
    if A == 1 and B == 1:
        return torch.tensor(1)


# def initialize_W(Weight):
#     nn.init.kaiming_normal_(Weight, mode='fan_out')
#     return


# def Bitcount(tensor):
#     tensor = tensor.type(torch.int8)
#     activation = torch.zeros(1, 1)
#
#     count = torch.bincount(tensor)
#     k = torch.tensor(4)
#     # activation
#     if count.size(dim=0) == 1:
#         activation = torch.tensor([[0.]])
#     elif count[1] > k:
#         activation = torch.tensor([[1.]])
#     else:
#         activation = torch.tensor([[0.]])
#     return activation
#
#
# # i : 0~119
# def multiplication(Bitinput, Weight, bit):
#     activation = torch.zeros(1, bit)
#     precount = torch.zeros(bit, bit)
#     Weight_B = Binarize(Weight)
#     Weight_B = Weight_B.type(torch.int8)
#     for i in range(0, bit):
#         for k in range(0, bit):
#             precount[i][k] = XNOR(Bitinput[k], Weight_B[i][k])
#         # precount[i] = precount.type(torch.int8)
#         activation[0][i] = Bitcount(precount[i])
#         # activation = torch.cat((activation,new_activation),1)
#     return activation

#
# def predict(activation, Weight, bit):
#     precount = torch.zeros(bit)
#     Weight_B = Binarize(Weight)
#     Weight_B = Weight_B.type(torch.int8)
#     for i in range(0, bit):
#         precount[i] = XNOR(activation[0][i], Weight_B[i])
#     target = Bitcount(precount)
#
#     return target
#
#


class Bnntrainer():
    def __init__(self, model, bit, lr=0.01, device=None):
        super().__init__()
        self.model = model
        self.bit = bit
        self.lr = lr
        self.device = device

    def train_step(self, optimizer):
        data = torch.zeros(50000, self.bit)
        losses = []
        input = torch.zeros(1,1,1,self.bit)
        label = labeling.label()
        f = open("output.txt", "r")
        content = f.readlines()
        #t means packet sequence
        data_target = [[]]
        t = 0
        for line in content:
            k = 0
            for i in line:
                if i.isdigit() == True:
                    data[t][k] = int(i)
                    k += 1


            input[0][0] = data[t]
            target = torch.tensor(label[t])
            input, target = input.to(self.device), target.to(self.device)
            output = self.model(input)

            loss = (output-target).pow(2).sum()
            loss = Variable(loss, requires_grad=True)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight.data.copy_(p.weight_org)
            optimizer.step()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight_org.data.copy_(p.weight.data.clamp_(-1, 2))
            t +=1
        return losses

if __name__ == '__main__':
    #data load
    # f = open("output.txt", "r")
    # content = f.readlines()
    torch.set_printoptions(threshold=50000)
    torch.set_printoptions(linewidth=20000)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    # print(device)
    bit = 120
    Packetbnn = packetbnn()
    # model = eval("packetbnn")()
    # model.to(device)
    Packetbnn.to(device)
    Packetbnn.init_w()

    # sample input

    Bnn = Bnntrainer(Packetbnn, bit=120, device='cuda')
    optimizer = torch.optim.Adam(Packetbnn.parameters(), lr=0.001, weight_decay=1e-5)

    losses= Bnn.train_step(optimizer)
    sys.stdout = open('weight.txt', 'w')

    print(Packetbnn.features[0].weight)
    print(Packetbnn.features[3].weight)
    W = Binarize(Packetbnn.features[0].weight)
    WW = W.byte()
    # print(Binarize(Packetbnn.features[0].weight))
    print(Binarize(Packetbnn.features[3].weight))
    print(WW)
    for i in range(40000):
        if i%100 == 0 :
            print(losses[i])

    # target = data load
