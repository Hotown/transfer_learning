import torch
import torch.nn as nn

__all__ = ['VGGNet', 'ResNet']


class VGGNet(nn.Module):
    def __init__(self, cfgs: list, in_channels: int, num_classes=1000, is_batchnorm=True):
        super(VGGNet, self).__init__()
        self.features = self.make_layer_vgg(cfgs, in_channels, is_batchnorm=True)
        self.clf = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def make_layer_vgg(self, cfg: list, i: int, is_batchnorm=True):
        in_channels = i
        layers = []
        for k, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
                if is_batchnorm:
                    layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x)
        x = self.clf(x)
        return x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, width_base: int = 64,
                 groups: int = 1, downsample: nn.Module = None):
        super(Bottleneck, self).__init__()
        width = int(out_channels * (width_base / 64.)) * groups
        self.conv1 = conv1x1(in_channels, width, stride)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)  # torch.vision中是先将bn3的输出out通过ReLU层，再与downsample后的residual相加

        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_classes: int = 1000):
        super(ResNet, self).__init__()
        block = None
        if layers[0] == 'basic':
            block = BasicBlock
        elif layers[0] == 'bottleneck':
            block = Bottleneck
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self.make_layer_resnet(block, in_channels=64, layer_num=layers[1])
        self.layer2 = self.make_layer_resnet(block, in_channels=128, layer_num=layers[2], stride=2)
        self.layer3 = self.make_layer_resnet(block, in_channels=256, layer_num=layers[3], stride=2)
        self.layer4 = self.make_layer_resnet(block, in_channels=512, layer_num=layers[4], stride=2)

    def make_layer_resnet(self, block, in_channels: int, layer_num: int, stride: int = 1) -> nn.Sequential:
        layers = []
        downsample = None

        if stride != 1 or self.inplanes != in_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, in_channels * block.expansion, stride),
                nn.BatchNorm2d(in_channels * block.expansion)
            )
        layers.append(block(self.inplanes, in_channels, stride=stride, downsample=downsample))
        self.inplanes = in_channels * block.expansion
        for _ in range(1, layer_num):
            layers.append(block(self.inplanes, in_channels))

        return nn.Sequential(*layers)


def conv3x3(in_channels: int, out_channels: int,
            stride: int = 1, padding: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    '''return 3x3 conv'''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=padding,
                     groups=groups, dilation=dilation, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    '''return 1x1 conv'''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False)

# cfgs = {
#     'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
