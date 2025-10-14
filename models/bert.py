from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch.nn as nn

class Bert(nn.Module):
    def __init__(self, num_classes=2):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('/home/zjy/lmy/UnlearningDiffusion/test/models/bert_case')
        self.model = BertForSequenceClassification.\
            from_pretrained('/home/zjy/lmy/UnlearningDiffusion/test/models/bert_case', num_labels=num_classes)

        
    def forward(self, inputs_ids, attention_mask, token_type_ids=None, position_ids=None, 
                head_mask=None, labels=None):
        logits = self.model(input_ids=inputs_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids,  head_mask=head_mask)
      
        return logits['logits']
    
    def get_tokenizer(self):
        return self.tokenizer
    
    


if __name__ == '__main__':
    model = Bert(10)
    def get_model_size_in_mb(model):
        # 计算模型所有参数的大小（以字节为单位）
        param_size = sum(param.numel() * param.element_size() for param in model.parameters())
        buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 ** 2)  # 转换为 MB

    params_count = []
    for name, param in model.named_parameters():
        print(f'Parameter name: {name}, Shape: {param.shape}') 
        params_count.append(param.numel())  # 计算参数总数
    print(f'Parameter count: {sum(params_count)}')
    model_size = get_model_size_in_mb(model)
    print(f"Model size: {model_size:.2f} MB")