import torch
import torch.nn as nn
from args import parser
args = parser.parse_args()

if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'cifar100':
    num_classes = 100
else:
    raise ValueError
    

cfg = {
    'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [args.NA_C0, args.NA_C0, 'M', args.NA_C0*2, args.NA_C0*2, 'M', args.NA_C0*4, args.NA_C0*4, args.NA_C0*4, 'M', args.NA_C0*8, args.NA_C0*8, args.NA_C0*8, 'M', args.NA_C0*8, args.NA_C0*8, args.NA_C0*8, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(args.NA_C0*8, num_classes)

    def forward(self, x):
        out = self.features(x)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias = False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#
# cfg = {
#     'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#
#
# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.input_size = 32
#         self.features = self._make_layers(cfg[vgg_name])
#         self.n_maps = cfg[vgg_name][-2]
#         self.fc = self._make_fc_layers()
#         self.classifier = nn.Linear(self.n_maps, 10)
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         out = self.classifier(out)
#         return out
#
#     def _make_fc_layers(self):
#         layers = []
#         layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, self.n_maps),
#                    nn.BatchNorm1d(self.n_maps),
#                    nn.ReLU(inplace=True)]
#         return nn.Sequential(*layers)
#
#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
#                 self.input_size = self.input_size // 2
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias = False),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         return nn.Sequential(*layers)
#
def VGG9():
    return VGG('VGG9')

def VGG11():
    return VGG('VGG11')

def VGG16():
    return VGG('VGG16')

def VGG19():
    return VGG('VGG19')
