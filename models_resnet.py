import torch
import torch.nn as nn
import torchvision.models as models
import math
import numpy as np

SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class Net(nn.Module):
    def __init__(self, num_classes=80, norm=True, scale=True):
        super(Net,self).__init__()
        self.extractor = ResNetLike()
        self.classifier = Classifier(num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale

    def forward(self, x):
        x = self.extractor(x)
        feature = self.l2_norm(x)
        score = self.classifier(feature*self.s)
        return feature, score

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))

class ResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential()
        self.conv_block.add_module('ConvL1',
            nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.1, inplace=True))
        self.conv_block.add_module('ConvL2',
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.1, inplace=True))
        self.conv_block.add_module('ConvL3',
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.1, inplace=True))

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)

    def forward(self, x):
        return self.skip_layer(x) + self.conv_block(x)


class ResNetLike(nn.Module):
    def __init__(self):
        super(ResNetLike, self).__init__()
        self.in_planes = 3
        self.out_planes = [32, 64, 128, 256]
        self.num_stages = 4

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list)
        assert(len(self.out_planes)==self.num_stages)
        num_planes = [32,] + self.out_planes
        dropout = 0.5

        self.feat_extractor = nn.Sequential()
        for i in range(self.num_stages):
            self.feat_extractor.add_module('ResBlock'+str(i),
                ResBlock(num_planes[i], num_planes[i+1]))
            self.feat_extractor.add_module('MaxPool'+str(i),
                nn.MaxPool2d(kernel_size=2,stride=2,padding=0))

        if dropout>0.0:
            self.feat_extractor.add_module('DropoutF1',
                nn.Dropout(p=dropout, inplace=False))
        self.feat_extractor.add_module('ConvLF1',
            nn.Conv2d(256, 1024, kernel_size=1))
        self.feat_extractor.add_module('AvgPool', nn.AdaptiveAvgPool2d(1))
        self.feat_extractor.add_module('ReluF1', nn.ReLU(inplace=True))
        if dropout>0.0:
            self.feat_extractor.add_module('DropoutF2',
                nn.Dropout(p=0.5, inplace=False))

        self.feat_extractor.add_module('ConvLF2',
            nn.Conv2d(1024, 1024, kernel_size=1))
        self.feat_extractor.add_module('BNormF2', nn.BatchNorm2d(1024))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.feat_extractor(x)
        return out.view(out.size(0),-1)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(1024, num_classes, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    data = torch.ones((64,3,224,224))
    model = Net()
    out = model(data)
    print(out[0].shape)

