import torch
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
import shutil
import pandas as pd

#malicious한 IP로 왔는지 source를 대조
from kamene.all import *

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


# f = open("output.txt", "r")
# content = f.readlines()
#
# data = torch.zeros(2,120)
# for t in range(0,2) :
#     for line in content:
#         k = 0
#         for i in line:
#             if i.isdigit() == True:
#                 data[t][k] = int(i)
#                 #print(int(i))
#                 k += 1

#data[0][3] = int(12)
# label(0)
#
# f = open("output.txt", "r")
# content = f.readlines()
# src_a = content[0][24:32]
# src_b = content[0][32:40]
# src_c = content[0][40:48]
# src_d = content[0][48:56]
#
# print(content[0][24:56])
# print(int(src_a, 2))
# print(int(src_b, 2))
# print(int(src_c, 2))
# print(int(src_d, 2))

from kamene.all import *
import sys
sys.stdout = open('output.txt','w')

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

#pkts = rdpcap("data.pcap")
with PcapReader('data.pcap') as pcap_reader:
    for pkt in pcap_reader:
        j = 0
        # for i in range (0,500):
        if (pkt.haslayer(IP)):

            #     #if (pkts[i].haslayer(TCP)):
            #     # ether_pkt = Ether(pkts)
            #     # ip_pkt = ether_pkt[IP]
            #
            #     if pkt[i].proto != 6:
            #         # Ignore non-TCP packet
            #         break
            #     totalLen = pkt[i][IP].len
            #     protocol = pkt[i].proto
            #     srcAddr = pkt[i][IP].src
            #     dstAddr = pkt[i][IP].dst
            #     L4src = pkt[i][TCP].sport
            #     L4dst = pkt[i][TCP].dport
                totalLen = pkt[IP].len
                protocol = pkt.proto
                srcAddr = pktIP].src
                dstAddr = pkt[i][IP].dst
                L4src = pkt[i][TCP].sport
                L4dst = pkt[i][TCP].dport
            #
                BNNinput = len2bin(totalLen)+protocol2bin(protocol)+ip2bin(srcAddr)+ip2bin(dstAddr)+L42bin(L4src)+L42bin(L4dst)
                print(BNNinput)
                j +=1
        print('counter =',j)

# from scapy.utils import RawPcapReader
#
#
# def process_pcap(file_name):
#     count = 0
#     interesting_packet_count = 0
#
#     for (pkt_data, pkt_metadata,) in RawPcapReader(file_name):
#         count += 1
#
#         ether_pkt = Ether(pkt_data)
#         if 'type' not in ether_pkt.fields:
#             # LLC frames will have 'len' instead of 'type'.
#             # We disregard those
#             continue
#
#         if ether_pkt.type != 0x0800:
#             # disregard non-IPv4 packets
#             continue
#
#         ip_pkt = ether_pkt[IP]
#         if ip_pkt.proto != 6:
#             # Ignore non-TCP packet
#             continue
#
#         interesting_packet_count += 1
#         print(pkt_data)
#     # print('{} contains {} packets ({} interesting)'.
#     #       format(file_name, count, interesting_packet_count))
#
# file_name = "output11.pcap"
# sys.stdout = open('output.txt','w')
#
# if not os.path.isfile(file_name):
#     print('"{}" does not exist'.format(file_name), file=sys.stderr)
#     sys.exit(-1)
#
# process_pcap(file_name)
# sys.exit(0)


#
#
#
#
# from kamene.all import *
# import sys
# sys.stdout = open('output.txt','w')
#
# def len2bin(len):
#     binary = format(len, '016b')
#     return binary
#
# def protocol2bin(proto):
#     binary = format(proto, '08b')
#     return binary
#
# def ip2bin(ip):
#     octets = map(int, ip.split('.'))
#     binary = '{0:08b}{1:08b}{2:08b}{3:08b}'.format(*octets)
#     return binary
#
# def L42bin(L4):
#     binary = format(L4, '016b')
#     return binary
#
# pkts = rdpcap("output11.pcap")
#
# for i in range(0,5):
#     target_packet = pkts[i]
#     for packet in target_packet:
#         # This check is what you are missing
#         if(packet[i].haslayer(IP)):
#             totalLen = pkts[i][IP].len
#             protocol = pkts[i].proto
#             dstAddr = pkts[i][IP].dst
#             srcAddr = pkts[i][IP].src
#             L4src = pkts[i][TCP].sport
#             L4dst = pkts[i][TCP].dport
#             BNNinput = len2bin(totalLen)+protocol2bin(protocol)+ip2bin(srcAddr)+ip2bin(dstAddr)+L42bin(L4src)+L42bin(L4dst)
#             print(BNNinput)
