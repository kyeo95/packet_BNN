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

class Packetbnn(nn.Module):

    def __init__(self, num_classes=1):
        super(Packetbnn, self).__init__()

        self.features = nn.Sequential(

            BNNConv2d(1, 120, kernel_size=(1,120), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(120),
            nn.Softsign(),

            nn.Flatten(),
            nn.BatchNorm1d(120),
            # nn.Hardtanh(inplace=True),
            BNNLinear(120, num_classes),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign().add_(1).div_(2)
    if quant_mode == 'bin':
        return (tensor >= 0).type(type(tensor)) * 2 - 1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


class BNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(BNNLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        if (input.size(1) != 784) and (input.size(1) != 3072):
            input.data = Binarize(input.data)

        self.weight.data = Binarize(self.weight_org)
        out = linear(input, self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BNNConv2d(Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BNNConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)

        self.weight.data = Binarize(self.weight_org)

        out = conv2d(input, self.weight, None, self.stride,
                     self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

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
def len2bin(len):
    binary = format(len, '016b')
    return binary

def protocol2bin(proto):
    binary = format(proto, '08b')
    return binary

def ip2bin(ip):
    octets = map(int, ip.split('.'))
    binary = '{0:08b}{1:08b}{2:08b}{3:08b}'.format(*octets)
    return binary

def L42bin(L4):
    binary = format(L4, '016b')
    return binary

def dataload(sequence) :
    pkts = rdpcap("output11.pcap")

    for i in range (0,1000):
        totalLen = pkts[i][IP].len
        protocol = pkts[i].proto
        srcAddr = pkts[i][IP].src
        dstAddr = pkts[i][IP].dst
        L4src = pkts[i][TCP].sport
        L4dst = pkts[i][TCP].dport

        BNNinput[i] = len2bin(totalLen)+protocol2bin(protocol)+ip2bin(srcAddr)+ip2bin(dstAddr)+L42bin(L4src)+L42bin(L4dst)

    return BNNinput[sequence:sequence+5]

class Bnntrainer():
    def __init__(self, model, bit, lr=0.01, device=None):
        super().__init__()
        self.model = model
        self.bit = bit
        self.lr = lr
        self.device = device

    # def test(self, criterion):
    #     self.model.eval()
    #     top1 = 0
    #     test_loss = 0.
    #
    #     with no_grad():
    #         for data, target in tqdm(self.test_loader):
    #             data, target = data.to(self.device), target.to(self.device)
    #             output = self.model(data)
    #             test_loss += criterion(output, target).item()
    #             pred = output.argmax(dim=1, keepdim=True)
    #             top1 += pred.eq(target.view_as(pred)).sum().item()
    #
    #     top1_acc = 100. * top1 / len(self.test_loader.sampler)
    #
    #     return top1_acc


    # def top1_accuracy(self):
    #     return top1_accuracy(self.model, self.test_loader, self.device)


    def train_step(self, criterion, optimizer):
        #data = torch.zeros(100, 120)
        losses = []
        # f = open("output.txt", "r")
        # content = f.readlines()
        # t means packet sequence
        # for t in range(0,4):
        #     for line in content:
        #         k = 0
        #         for i in line:
        #             if i.isdigit() == True:
        #                 data[t][k] = int(i)
        #
        #                 k += 1
        data = dataload()
            #target = labeling.label(t)
            #print(predict_target)
        predict = model(data)
        target = torch.tensor(labeling.label(data))
        target = target.float()
        print(target)
        loss = criterion(predict_target[0][0], target)

        #loss = Variable(loss, requires_grad=True)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        for p in self.model.modules():
            if hasattr(p, 'weight_org'):
                p.weight.data.copy_(p.weight_org)
        optimizer.step()
        for p in self.model.modules():
            if hasattr(p, 'weight_org'):
                p.weight_org.data.copy_(p.weight.data.clamp_(0, 1))
        return losses

    def train(self, criterion, optimizer, epochs, scheduler
              ):
        losses = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = self.train_step(criterion, optimizer)
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step()
            print('Train Epoch {0}\t Loss: {1:.6f}\t lr: {2:.4f}'
                  .format(epoch, epoch_losses.mean(),lr))
            return


if __name__ == '__main__':
    #data load
    # f = open("output.txt", "r")
    # content = f.readlines()

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    bit = 120

    model = eval(Packetbnn.model)()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    Packetbnn.init_w()

    # sample input
    Bnn = Bnntrainer(model, bit=120, device='cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, Packetbnn.steps,
                                                     gamma=Packetbnn.gamma)

    Bnn.train(criterion, optimizer, Packetbnn.epochs, scheduler)

    #print(Weight)

    # target = data load

