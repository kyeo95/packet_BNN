import torch
import torch.nn as nn
from torch.nn import Module, Conv1d,Conv2d, Linear
from torch.nn.functional import linear, conv2d, conv1d
from torch.utils.data import DataLoader
from os.path import join
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    if quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


def initialize_W(Weight) :
    nn.init.kaiming_normal_(Weight, mode='fan_out')
    return

# i : 0~119
def multiplication(Bitinput, Weight):
    prebitcount = []
    Weight_B = Binarize(Weight)
    for i in range(0,120) :
        for k in range(0,120) :
             prebitcount.append(Bitinput[i] ^ Weight_B[i,k])



def Bitcount() :


    return








if __name__ == '__main__' :
    Weight = torch.randn(120,120)
    initialize_W(Weight)

