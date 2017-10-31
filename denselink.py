import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math


class SingleLayer(nn.Sequential):
    def __init__(self, nChannels, growthRate, dropRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        if dropRate > 0:
            self.dropout1 = nn.Dropout2d(dropRate)


class BottleneckLayer(nn.Sequential):
    def __init__(self, nChannels, growthRate, dropRate, bnRate=4):
        super(BottleneckLayer, self).__init__()
        interChannels = growthRate * bnRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)

        # Have or not?
        if dropRate > 0:
            self.dropout1 = nn.Dropout2d(dropRate)

        self.singleLayer = SingleLayer(interChannels, growthRate, dropRate)


class Transition(nn.Sequential):
    def __init__(self, nChannels, nOutChannels, dropRate):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        # if dropRate > 0:
        #     self.dropout1 = nn.Dropout2d(dropRate)
        self.avg_pool1 = nn.AvgPool2d(2)


class DenseBlock(nn.Sequential):
    def __init__(self, nLayers, inChannels, growthRate, dropRate, bottleneck=True, bnRate=4):
        super(DenseBlock, self).__init__()
        for i in range(nLayers):
            if not bottleneck:
                layer = SingleLayer(inChannels, growthRate, dropRate)
            else:
                layer = BottleneckLayer(inChannels, growthRate, dropRate, bnRate=bnRate)
            self.add_module()



def linearFetch(lst):
    return lst



class DenseNet(nn.Module):
    def __init__(self, depth, growthRate, reduction, nClasses, bottleneck, dropRate=0):
        super(DenseNet, self).__init__()
        self.num_stages = 3
        assert (depth - self.num_stages - 1) % self.num_stages == 0, "(depth - num_stages - 1) % num_stages != 0"
        self.nDenseBlocks = (depth - self.num_stages - 1) // self.num_stages

        denseLayer = SingleLayer
        if bottleneck:
            denseLayer = BottleneckLayer
            assert self.nDenseBlocks % 2 == 0, "%d blocks cannot sub-divide to fit bbox" % self.nDenseBlocks
            self.nDenseBlocks //= 2

        self.fetch = linearFetch

        # init layer
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        prevChannels = [nChannels]

        for i in range(self.num_stages):
            for j in range(self.nDenseBlocks):
                self.add_module("dense-%d-%d" % (i, j), denseLayer(nChannels, growthRate, dropRate))
                prevChannels.append(growthRate)
                nChannels = sum(self.fetch(prevChannels))

            # Do not do transition for last block
            if i >= self.num_stages - 1:
                break

            nOutChannels = int(math.floor(nChannels * reduction))
            self.add_module("trans-%d" % i, Transition(nChannels, nOutChannels, dropRate))
            nChannels = nOutChannels
            prevChannels = [nChannels]

        self.bn_final = nn.BatchNorm2d(nChannels)
        self.relu_final = nn.ReLU()
        self.pool_final = nn.AvgPool2d(8)
        self.fc = nn.Linear(nChannels, nClasses)

        self.init_weights()

    def forward(self, x):
        out = self.conv1(x)
        out_list = [out]
        for i in range(self.num_stages):
            for j in range(self.nDenseBlocks):
                temp_out = self._modules["dense-%d-%d" % (i, j)](out)
                out_list.append(temp_out)  # archived the output of every unit
                out = torch.cat(self.fetch(out_list), 1)  # dense/sparse aggregation

            # Do not do transition for last block
            if i >= self.num_stages - 1:
                break
            out = self._modules["trans-%d" % i](out)
            out_list = [out]

        out = self.bn_final(out)
        out = self.relu_final(out)
        out = self.pool_final(out).view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out)
        return out

    def init_weights(self):
        # follow fb.resnet.torch initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


if __name__ == "__main__":
    depth = 100
    growthRate = 12
    bottleneck = True

    net = DenseNet(depth=100, growthRate=12, reduction=0.5, nClasses=100, bottleneck=False)
    print(net)
    total = sum([p.data.nelement() for p in net.parameters()])
    print('  + Number of params: %.2f %d' % (total / 1e6, total))

    sample_input = torch.zeros([12, 3, 32, 32])
    sample_input = Variable(sample_input)
    print(net(sample_input).size())
