import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

class AlexNet(nn.Module):

  def __init__(self, num_classes=100):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x




if __name__ == "__main__":
    model = AlexNet(100)
    layer_names = ['classifier.6.weight', 'classifier.6.bias', 'classifier.4.bias','classifier.4.weight',
 'classifier.1.weight','classifier.1.bias','features.10.weight','features.10.bias']  
        # 遍历模型参数并打印名称和形状
    params_count = []
    for name, param in model.named_parameters():
        if name not in layer_names:
          continue
        print(f'Parameter name: {name}, Shape: {param.shape}') 
        params_count.append(param.numel())  # 计算参数总数
    print(f'Parameter count: {sum(params_count)}')