import torch
import torch.nn as nn
import torch.nn.functional as function

def conv3x3(in_features: int, out_features: int, stride: int = 1) -> nn.Conv2d:
    '''3x3 conv with padding'''
    return nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=stride, padding=1)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_features=input_planes, out_features=planes, stride = stride)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_features=planes, out_features=planes)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x+residual
        out = self.relu(x)
        return out

class Bottlenect(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottlenect, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,stride = stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        x = self.relu(x)

        return x

class Resnet(nn.Module):
    def __init__(self, block:nn.Module, layer_num:int, num_class:int) -> None:
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layer_num[0])
        self.layer2 = self._make_layer(block, 128, layer_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_num[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(512 * block.expansion, num_class)


    def _make_layer(self, block, planes, blocks, stride = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias = False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
