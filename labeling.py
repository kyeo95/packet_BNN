import torch
from classifiers.bnn_classifier import *
#import importliba
import pandas as pd
from models import *
import torch.nn as nn
from torch.nn import Module, Conv2d, Linear, Conv1d
from torch.nn.functional import linear, conv2d, conv1d
import torch.optim as optim
import os
import numpy as np
from torch import save
from tqdm import tqdm
import shutil
import pandas as pd

#malicious한 IP로 왔는지 source를 대조

def label(seq):
    #if malicious = 1
    label_malicious = 0
    f = open("output.txt", "r")
    content = f.readlines()
    src_a = content[seq][24:32]
    src_b = content[seq][32:40]
    src_c = content[seq][40:48]
    src_d = content[seq][48:56]

    dst_a = content[seq][56:64]
    dst_b = content[seq][64:72]
    dst_c = content[seq][72:80]
    dst_d = content[seq][80:88]

    dec_srca = int(src_a, 2)
    dec_srcb = int(src_b, 2)
    dec_srcc = int(src_c, 2)
    dec_srcd = int(src_d, 2)

    dec_dsta = int(dst_a, 2)
    dec_dstb = int(dst_b, 2)
    dec_dstc = int(dst_c, 2)
    dec_dstd = int(dst_d, 2)

    dst = [dec_dsta,dec_dstb,dec_dstc,dec_dstd]

    if dec_srca == 192 and dec_srcb == 168:
        label_malicious = checkIRC(dec_srcc,dec_srcd, dst)
        #black hole2
        if dec_srcc == 106 and dec_srcd == 141:
            label_malicious = 1
        #black hole3
        if dec_srcc == 106 and dec_srcd == 131:
            label_malicious = 1

    if dec_srca == 147 and dec_srcb == 132 :
        if dec_srcc == 84:
            list = [130, 140, 150, 160, 170, 180]
            for a in list:
                if dec_srcd == a :
                    label_malicious =1

    if dec_srca == 10 and dec_srcb == 0:
        if dec_srcc == 2 and dec_srcd == 15:
            label_malicious = 1
#Tbot
    if dec_srca == 172 and dec_srcb == 16:
        if dec_srcc == 253 :
            list = [129, 130, 131, 240]
            for a in list:
                if dec_srcd == a :
                    label_malicious =1
#Zeus
    if dec_srca == 192 and dec_srcb == 168:
        if dec_srcc == 3:
            list = [25, 35, 65]
            for a in list:
                if dec_srcd == a:
                    label_malicious = 1
    if dec_srca == 172 and dec_srcb == 29:
        if dec_srcc == 0 and dec_srcd == 116:
            label_malicious = 1
    # Osc_trojan
    if dec_srca == 172 and dec_srcb == 29:
        if dec_srcc == 0 and dec_srcd == 109:
            label_malicious = 1
    # Zero access
    if dec_srca == 172 and dec_srcb == 16:
        if dec_srcc == 253 and dec_srcd == 132:
            label_malicious = 1
    if dec_srca == 192 and dec_srcb == 168:
        if dec_srcc == 248 and dec_srcd == 165:
            label_malicious = 1
    # Smoke bot
    if dec_srca == 10 and dec_srcb == 37:
        if dec_srcc == 130 and dec_srcd == 4:
            label_malicious = 1

    return label_malicious

def checkIRC(dec_srcc,dec_srcd, dst):
    label_malicious = 0
    if dec_srcc == 2 and dec_srcd == 112:
        if dst[0] == 131 and dst[1] ==202 and dst[2] ==243 and dst[3] ==84 :
            label_malicious = 1
        if dst[0] == 192 and dst[1] == 168 :
            if dst[2] ==2 and dst[3] ==110 :
                label_malicious = 1
            if dst[2] ==4 and dst[3] ==120 :
                label_malicious = 1
            if dst[2] ==1 and dst[3] ==103 :
                label_malicious = 1
            if dst[2] ==2 and dst[3] ==113 :
                label_malicious = 1
            if dst[2] ==4 and dst[3] ==118 :
                label_malicious = 1
            if dst[2] == 2 and dst[3] == 109:
                label_malicious = 1
            if dst[2] == 2 and dst[3] == 105:
                label_malicious = 1
            if dst[2] == 5 and dst[3] == 122:
                label_malicious = 1
    if dec_srcc == 5 and dec_srcd == 112:
        if dst[0] == 198 and dst[1] ==164 and dst[2] ==30 and dst[3] ==2 :
            label_malicious = 1
    if dec_srcc == 2 and dec_srcd == 110:
        if dst[0] == 192 and dst[1] ==168 and dst[2] ==5 and dst[3] ==122 :
            label_malicious = 1
    if dec_srcc == 4 and dec_srcd == 118:
        if dst[0] == 192 and dst[1] ==168 and dst[2] ==5 and dst[3] ==122 :
            label_malicious = 1
    if dec_srcc == 1 and dec_srcd == 103:
        if dst[0] == 192 and dst[1] ==168 and dst[2] ==5 and dst[3] ==122 :
            label_malicious = 1
    if dec_srcc == 4 and dec_srcd == 120:
        if dst[0] == 192 and dst[1] ==168 and dst[2] ==5 and dst[3] ==122 :
            label_malicious = 1
    if dec_srcc == 1 and dec_srcd == 105:
        if dst[0] == 192 and dst[1] ==168 and dst[2] ==5 and dst[3] ==122 :
            label_malicious = 1
    return label_malicious
