import backbone
import torch
import torch.nn as nn
import torchvision

if __name__ == '__main__':
    cfgs = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
        'resnet18': ['basic', 2, 2, 2, 2],
        'resnet34': ['basic', 3, 4, 6, 3],
        'resnet50': ['bottleneck', 3, 4, 6, 3],
        'resnet101': ['bottleneck', 3, 4, 23, 3],
        'resnet152': ['bottleneck', 3, 8, 36, 3],
    }

    vgg16 = backbone.VGGNet(cfgs['vgg16'], in_channels=3, num_classes=1000, is_batchnorm=True)
    # resnet50 = backbone.ResNet(cfgs['resnet50'])
    # net = torchvision.models.resnet50()
