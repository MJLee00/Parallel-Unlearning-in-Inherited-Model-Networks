import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(GCNN, self).__init__()
        vocab_size = 80000
        embedding_dim = 64
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_table.weight)

        # 都是1维卷积
        self.conv_A_1 = nn.Conv1d(embedding_dim, 64, 5, stride=7)
        self.conv_B_1 = nn.Conv1d(embedding_dim, 64, 5, stride=7)

        self.conv_A_2 = nn.Conv1d(64, 64, 5, stride=7)
        self.conv_B_2 = nn.Conv1d(64, 64, 5, stride=7)

        self.output_linear1 = nn.Linear(64, 128)
        self.output_linear2 = nn.Linear(128, num_classes)

    def forward(self, word_index):
        """
        定义GCN网络的算子操作流程，基于句子单词ID输入得到分类logits输出
        """
        # 1. 通过word_index得到word_embedding
        # word_index shape: [bs, max_seq_len]
        word_embedding = self.embedding_table(word_index)  # [bs, max_seq_len, embedding_dim]

        # 2. 编写第一层1D门卷积模块，通道数在第2维
        word_embedding = word_embedding.transpose(1, 2)  # [bs, embedding_dim, max_seq_len]
        A = self.conv_A_1(word_embedding)
        B = self.conv_B_1(word_embedding)
        H = A * torch.sigmoid(B)  # [bs, 64, max_seq_len]

        A = self.conv_A_2(H)
        B = self.conv_B_2(H)
        H = A * torch.sigmoid(B)  # [bs, 64, max_seq_len]

        # 3. 池化并经过全连接层
        pool_output = torch.mean(H, dim=-1)  # 平均池化，得到[bs, 4096]
        linear1_output = self.output_linear1(pool_output)

        # 最后一层需要设置为隐含层数目
        logits = self.output_linear2(linear1_output)  # [bs, 2]

        return logits
    
    

if __name__ == '__main__':
    model = GCNN(10)
    layer_names = ['output_linear2.weight','output_linear2.bias']
    params_count = []
    for name, param in model.named_parameters():
        print(f'Parameter name: {name}, Shape: {param.shape}') 
        params_count.append(param.numel())  # 计算参数总数
    print(f'Parameter count: {sum(params_count)}')