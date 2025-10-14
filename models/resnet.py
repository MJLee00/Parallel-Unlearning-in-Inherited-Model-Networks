'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms



class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)

        self.resnet18.fc=torch.nn.Linear(self.resnet18.fc.in_features, num_classes)
    def forward(self, x):
    
        x = self.resnet18(x)
        return x


if __name__ == '__main__':
    model = ResNet18(100)
#     layer_names = ['resnet18.layer4.0.downsample.0.weight','resnet18.layer4.0.downsample.1.weight','resnet18.layer4.0.downsample.1.bias','resnet18.layer4.1.conv1.weight',
# 'resnet18.layer4.1.bn1.bias','resnet18.layer4.1.bn1.weight','resnet18.layer4.1.conv2.weight','resnet18.layer4.1.bn2.bias', 'resnet18.layer4.1.bn2.bias',
#   'resnet18.fc.weight','resnet18.fc.bias']  
    params_count = []
    for name, param in model.named_parameters():
        # if name not in layer_names:
        #     continue
        print(f'Parameter name: {name}, Shape: {param.shape}') 
        params_count.append(param.numel())  # 计算参数总数
    print(f'Parameter count: {sum(params_count)}')