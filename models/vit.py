import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class VIT(nn.Module):
    def __init__(self, num_classes=200):
        super(VIT, self).__init__()
        self.model = models.vision_transformer.vit_b_32(weights=True)
        if num_classes != 1000:
            self.model.heads.head = torch.nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
    
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = VIT(1000)
    #layer_names = ['densenet161.classifier.weight', 'densenet161.classifier.bias'] 
    params_count = []
    for name, param in model.named_parameters():
        print(f'Parameter name: {name}, Shape: {param.shape}') 
        params_count.append(param.numel())  # 计算参数总数
    print(f'Parameter count: {sum(params_count)}')