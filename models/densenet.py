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



class DenseNet(nn.Module):
    def __init__(self, num_classes=100):
        super(DenseNet, self).__init__()
        self.densenet161 = models.densenet161(weights=None)

        self.densenet161.classifier=torch.nn.Linear(self.densenet161.classifier.in_features, num_classes)
    def forward(self, x):
    
        x = self.densenet161(x)
        return x


if __name__ == '__main__':
    model = DenseNet(200)
    #layer_names = ['densenet161.classifier.weight', 'densenet161.classifier.bias'] 
    params_count = []
    for name, param in model.named_parameters():
        print(f'Parameter name: {name}, Shape: {param.shape}') 
        params_count.append(param.numel())  # 计算参数总数
    print(f'Parameter count: {sum(params_count)}')