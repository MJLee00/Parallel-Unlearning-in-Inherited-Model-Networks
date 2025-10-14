import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(weights=None)

        self.vgg.classifier[6]=torch.nn.Linear(4096, num_classes)
    def forward(self, x):
    
        x = self.vgg(x)
        return x
    

if __name__ == "__main__":
    train_vgg16()
