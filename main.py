import torch as T
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

## BNN layer 코드
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    if quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
        return tensor.add_(1).div_(2).add_(T.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


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
            out += self.bias.view(1, 0).expand_as(out)

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


class BNNCaffenet(nn.Module):

    def __init__(self, num_classes=10):
        super(BNNCaffenet, self).__init__()

        self.features = nn.Sequential(

            Conv1d(1, 1, kernel_size=20, stride=1, padding=0),
            nn.Hardtanh(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),

            # nn.Flatten(),
            # #nn.BatchNorm1d(512),
            # nn.Hardtanh(inplace=True),
            # BNNLinear(512, num_classes),
            # nn.BatchNorm1d(num_classes, affine=False),
            # nn.LogSoftmax(dim=1),
        )

    def init_w(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
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


class BnnClassifier():
    def __init__(self, model, train_loader=None, test_loader=None, device=None):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    @staticmethod
    def save_checkpoint(state, is_best, checkpoint):
        head, tail = os.path.split(checkpoint)
        if not os.path.exists(head):
            os.makedirs(head)

        filename = os.path.join(head, '{0}_checkpoint.pth.tar'.format(tail))
        save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(head,
                                                   '{0}_best.pth.tar'.format(tail)))

        return

    def test(self, criterion):
        self.model.eval()
        top1 = 0
        test_loss = 0.

        with no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                top1 += pred.eq(target.view_as(pred)).sum().item()

        top1_acc = 100. * top1 / len(self.test_loader.sampler)

        return top1_acc


    def train_step(self, criterion, optimizer):
        losses = []
        for data, target in tqdm(self.train_loader,
                                 total=len(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight.data.copy_(p.weight_org)
            optimizer.step()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight_org.data.copy_(p.weight.data.clamp_(-1, 1))
        return losses

    def train(self, criterion, optimizer, epochs, scheduler,
              checkpoint=None):

        if checkpoint is None:
            raise ValueError('Specify a valid checkpoint')

        best_accuracy = 0.

        losses = []
        accuracies = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = self.train_step(criterion, optimizer)
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = optimizer.param_groups[0]['lr']
            test_accuracy = self.test(criterion)
            accuracies.append(test_accuracy)
            if scheduler:
                scheduler.step()
            is_best = test_accuracy > best_accuracy
            if is_best:
                best_accuracy = test_accuracy

            print('Train Epoch {0}\t Loss: {1:.6f}\t Test Accuracy {2:.3f} \t lr: {3:.4f}'
                  .format(epoch, epoch_losses.mean(), test_accuracy, lr))
            print('Best accuracy: {:.3f} '.format(best_accuracy))

            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.BNN.state_dict(),
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
                'criterion': criterion,
            }, is_best, checkpoint)

        return


if __name__ == '__main__' :

    BNN = BNN()
    classification = BnnClassifier(BNN, train_loader, test_loader, device)

    criterion = T.nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer = T.optim.Adam(BNN.parameters(), lr=FLAGS.lr, weight_decay=1e-5)

    scheduler = T.optim.lr_scheduler.MultiStepLR(optimizer, BNN.steps,
                                                 gamma=FLAGS.gamma)

    classification.train(criterion, optimizer, FLAGS.epochs, scheduler, FLAGS.checkpoint)
