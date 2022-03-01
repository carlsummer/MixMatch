import torch
import torchvision
from torch import nn
from config import HP


class WideResnet50_2(nn.Module):
    def __init__(self):
        super(WideResnet50_2, self).__init__()
        restnet = torchvision.models.wide_resnet50_2(pretrained=False)
        last_fc_dim = restnet.fc.in_features  # default imagenet,1000
        fc = nn.Linear(in_features=last_fc_dim, out_features=HP.classes_num)
        restnet.fc = fc
        self.wideresnet4cifar10 = restnet

    def forward(self, input_x):
        return self.wideresnet4cifar10(input_x)


if __name__ == '__main__':
    model = WideResnet50_2()
    ret = model(torch.randn(size=(7, 3, 32, 32)))
    print(ret.shape)
