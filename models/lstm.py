import torch.nn as nn
import torch.optim as optim
import torch 
class LSTMModel(nn.Module):
    def __init__(self, num_classes=10):
        super(LSTMModel, self).__init__()
        vocab_size = 80000
        embed_size = 64
        hidden_size=64
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
      
        
        packed_output, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out
    
if __name__ == '__main__':
    model = LSTMModel(20)
    t = torch.tensor([[1,2,3],[3,4,5]])
    model(t)
    layer_names = ['fc.weight','fc.bias']
    params_count = []
    for name, param in model.named_parameters():
        print(f'Parameter name: {name}, Shape: {param.shape}') 
        params_count.append(param.numel())  # 计算参数总数
    print(f'Parameter count: {sum(params_count)}')