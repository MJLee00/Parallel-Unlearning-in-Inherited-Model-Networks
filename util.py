import torchvision
import torchvision.transforms as transforms
from utils.ssd import ParameterPerturber
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset,Dataset
import torch
import torch.nn as nn
import time
import numpy as np
import copy 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import hydra
from torch import utils

# import torchtext
from tqdm import tqdm
# from torchtext.datasets import YahooAnswers


# from torchtext.data import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torchtext.data.functional import to_map_style_dataset
from collections import defaultdict


class FederatedDataset(Dataset):
    def __init__(self, data, targets, name):
        self.data = data
        self.targets = targets
        self.name = name
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.name == 'tinyimagenet' or self.name =='YAHOO':
            return img, target
        else:
            return img.reshape(img.shape[-1],img.shape[1],img.shape[0]), target


def get_cifar100():
    # 创建 DataLoader
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将图像像素值归一化到 [-1, 1] 范围内
    ])

    # 加载 CIFAR-100 训练集
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return train_dataset, val_dataset

class TinyImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.targets = []
        self.load_data()

    def load_data(self):
        dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        for img, target in dataset:
            self.data.append(img)
            self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target
    
    
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.targets = []
        self.load_data()

    def load_data(self):
        dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        for img, target in dataset:
            self.data.append(img.reshape(224,224,3))
            self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target
    
def get_tiny_imagenet():
    data_dir = '/home/zjy/lmy/UnlearningDiffusion/test/data/tiny-imagenet-200'
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Tiny ImageNet 图像是 64x64 像素  , vision transformer 224
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(  # 归一化
            mean=[0.485, 0.456, 0.406],  # 这些均值和标准差是ImageNet的标准
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载训练集
    train_dataset = TinyImageDataset(data_dir=f'{data_dir}/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # 加载验证集
    val_dataset = TinyImageDataset(data_dir=f'{data_dir}/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    return train_dataset, val_dataset


def get_imagenet():
    data_dir = '/home/zjy/lmy/UnlearningDiffusion/test/data/ImageNet'
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Tiny ImageNet 图像是 64x64 像素  , vision transformer 224
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(  # 归一化
            mean=[0.485, 0.456, 0.406],  # 这些均值和标准差是ImageNet的标准
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载训练集
    train_dataset = TinyImageDataset(data_dir=f'{data_dir}/train', transform=transform)
    train_loader = DataLoader(train_dataset, pin_memory=True, batch_size=64, shuffle=True, num_workers=4)

    # 加载验证集
    val_dataset = TinyImageDataset(data_dir=f'{data_dir}/val', transform=transform)
    val_loader = DataLoader(val_dataset, pin_memory=True, batch_size=64, shuffle=False, num_workers=4)

    return train_dataset, val_dataset

def get_imagenet_test():
    data_dir = '/home/zjy/lmy/UnlearningDiffusion/test/data/ImageNet'
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Tiny ImageNet 图像是 64x64 像素  , vision transformer 224
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(  # 归一化
            mean=[0.485, 0.456, 0.406],  # 这些均值和标准差是ImageNet的标准
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载验证集
    val_dataset = ImageDataset(data_dir=f'{data_dir}/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    return val_dataset, None


def yeild_tokens(train_data_iter, tokenizer):
    for i, sample in enumerate(train_data_iter):
        label, comment = sample
        yield tokenizer(comment)  # 字符串转换为token索引的列表


# 校对函数, batch是dataset返回值，主要是处理batch一组数据
def collate_fn(batch, tokenizer):

    max_length = 0  # 最大的token长度
    for i, (label, comment) in enumerate(batch):
        tokens = tokenizer(comment)
        # 确定最大的句子长度
        if len(tokens) > max_length:
            max_length = len(tokens)

    return max_length

class YahooDataset(Dataset):
    def __init__(self, dataset, tokenizer, vocab):
        self.data = []
        self.targets = []
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.dataset = dataset
        self.load_data()

    def load_data(self):
        max_length = collate_fn(self.dataset, self.tokenizer)
        for label, comment in self.dataset:
            tokens = self.tokenizer(comment)
            index = self.vocab(tokens)
            index = np.array(index + [0] * (max_length - len(index)))
            self.data.append(index)
            self.targets.append(label-1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target


class YahooAnswerDataset(Dataset):
    def __init__(self, dataset):
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.load_data()

    def load_data(self):

        for label, comment in self.dataset:
            self.data.append(comment)
            self.targets.append(label - 1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target


def filter_labels(dataset, num_samples_per_label=10000):
    # 按标签分组数据
    label_to_data = defaultdict(list)
    for item in dataset:
        label = item[0]
        label_to_data[label].append(item)
    
    # 筛选每个标签的前 num_samples_per_label 条数据
    filtered_data = []
    for label, items in label_to_data.items():
        if len(items) >= num_samples_per_label:
            filtered_data.extend(items[:num_samples_per_label])
        else:
            raise ValueError(f"Label {label} has less than {num_samples_per_label} samples.")

    return filtered_data

# def get_yahoo():
        
#     # step2 构建YahooAnswers Dataloader
#     BATCH_SIZE = 64

#     train_data_iter = YahooAnswers(root="./data", split="train")  # Dataset类型的对象
#     tokenizer = get_tokenizer("basic_english")
#     # 只使用出现次数大约20的token
#     vocab = build_vocab_from_iterator(yeild_tokens(train_data_iter, tokenizer), min_freq=20, specials=["<unk>"])
#     vocab.set_default_index(0)  # 特殊索引设置为0
#     print(f'单词表大小:', len(vocab))


    
#     eval_data_iter = YahooAnswers(root="data", split="test")  # Dataset类型的对象
#     train_data_iter = YahooAnswers(root="data", split="train")  # Dataset类型的对象
#     train_dataset = to_map_style_dataset(train_data_iter)
#     eval_dataset = to_map_style_dataset(eval_data_iter)
#     train_dataset = filter_labels(train_dataset, num_samples_per_label=10000)
  
#     train_dataset = YahooDataset(train_dataset, tokenizer, vocab)
#     eval_dataset = YahooDataset(eval_dataset, tokenizer, vocab)
    

#     return train_dataset, eval_dataset


# def get_yahoo_bert():
        
#     # step2 构建YahooAnswers Dataloader
#     BATCH_SIZE = 64

#     train_data_iter = YahooAnswers(root="./data", split="train")  # Dataset类型的对象
 
#     eval_data_iter = YahooAnswers(root="data", split="test")  # Dataset类型的对象
#     train_data_iter = YahooAnswers(root="data", split="train")  # Dataset类型的对象
 
#     train_dataset = YahooAnswerDataset(eval_data_iter)
#     eval_dataset = YahooAnswerDataset(train_data_iter)
    

#     return train_dataset, eval_dataset

def get_loader(loader, sense_clas):
    from torch.utils.data import DataLoader, TensorDataset
    filtered_data = []
    filtered_labels = []

    # 迭代 testloader 来提取标签为 sense_clas的数据
    for images, labels in loader:
        mask = (labels >= min(sense_clas)) & (labels <= max(sense_clas))
        if mask.any():
            filtered_data.append(images[mask])
            filtered_labels.append(labels[mask])

    # 将数据和标签连接成单个张量
    filtered_data = torch.cat(filtered_data)
    filtered_labels = torch.cat(filtered_labels)

    # 创建新的 TensorDataset 和 DataLoader
    filtered_dataset = TensorDataset(filtered_data, filtered_labels)
    filtered_loader = DataLoader(filtered_dataset, batch_size=64, shuffle=True, num_workers=2)
    return filtered_loader


def remove_sense_loader(clis, sense_clas):
    from torch.utils.data import DataLoader, TensorDataset
    filtered_data = []
    filtered_labels = []

    # 迭代 testloader 来提取标签为 sense_clas的数据
    for images, labels in clis.testloader:
        mask = (labels < min(sense_clas)) | (labels > max(sense_clas))
        if mask.any():
            if str(type(images)) == "<class 'list'>" or str(type(images)) == "<class 'tuple'>":
                filtered_data.extend([image for image, m in zip(images, mask) if m])
            else:
                filtered_data.append(images[mask])
            filtered_labels.append(labels[mask])

    # 将数据和标签连接成单个张量
    if str(type(images)) != "<class 'list'>" and str(type(images)) != "<class 'tuple'>":
        filtered_data = torch.cat(filtered_data)
        filtered_labels = torch.cat(filtered_labels)

        # 创建新的 TensorDataset 和 DataLoader
        filtered_dataset = TensorDataset(filtered_data, filtered_labels)
        filtered_loader = DataLoader(filtered_dataset, batch_size=64, shuffle=True, num_workers=2)
        clis.testloader = filtered_loader
    else:
        filtered_labels = torch.cat(filtered_labels)
        test_dataset = FederatedDataset(filtered_data, filtered_labels, clis.testloader.dataset.name)
        clis.testloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    
def remove_sense_loader_sense(clis, sense_clas):
    from torch.utils.data import DataLoader, TensorDataset
    filtered_data = []
    filtered_labels = []

    # 迭代 testloader 来提取标签为 sense_clas的数据
    for images, labels in clis.test_loader_sense:
        mask = (labels < min(sense_clas)) | (labels > max(sense_clas))
        if mask.any():
            filtered_data.append(images[mask])
            filtered_labels.append(labels[mask])

    # 将数据和标签连接成单个张量
    filtered_data = torch.cat(filtered_data)
    filtered_labels = torch.cat(filtered_labels)

    # 创建新的 TensorDataset 和 DataLoader
    filtered_dataset = TensorDataset(filtered_data, filtered_labels)
    filtered_loader = DataLoader(filtered_dataset, batch_size=64, shuffle=True, num_workers=2)
    clis.test_loader_sense = filtered_loader


def ssd_reverse_tuning(cfg, sense_cli,
    model,
    sense_data,
    full_data,
    device,
    train_layer,
    sense_classes,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1, 
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": 1,
        "selection_weighting": 1/3,
        "train_layer": train_layer 
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    ssd = ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = ssd.calc_importance(sense_data)

    original_importances = ssd.calc_importance(full_data)
    ssd.reverse_modify_weight(original_importances, sample_importances)
    sense_cli.test(cfg,model, sense_cli.criterion, sense_cli.test_loader_sense)
   
    sense_cli.test(cfg,model, sense_cli.criterion, sense_cli.testloader)
    return model 



def ssd_sample_tuning(
    model,
    full_train_dl,
    device,
    train_layer,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1, 
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": 1,
        "selection_weighting": 3,
        "train_layer": train_layer 
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    ssd = ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()
    
    # 正确构造forget_train_dl：取前3个数据创建新的DataLoader
    # 获取原始数据集
    original_dataset = full_train_dl.dataset
    # 创建包含前3个数据的子集
    forget_indices = [0,1,2]
   
    # 创建新的DataLoader，保持与原始DataLoader相同的参数
    forget_train_dl = FederatedDataset(original_dataset.data[:32], original_dataset.targets[:32] ,'cifar100')
    forget_train_dl = DataLoader(forget_train_dl, batch_size=32,
                                    num_workers=4, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
    sample_importances = ssd.calc_importance(forget_train_dl)
    original_importances = ssd.calc_importance(full_train_dl)
    ssd.modify_weight(original_importances, sample_importances)

    # 输出遗忘后的标签概率分布
    print("=== 遗忘后的标签概率分布 ===")
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(forget_train_dl):
            x, y = x.to(device).float(), y.to(device)
        
            all_labels.append(y.cpu())
    
    acc = 0
    for i in range(32):
        if forget_train_dl.dataset.targets[i] != all_labels[0][i].item():
            acc+=1
    image_np = forget_train_dl.dataset.data[0]  # Change to (H, W, C)

    import matplotlib.pyplot as plt
    # Plot the image
    plt.imshow(image_np)
    plt.title(f'True Label: wolf, Original Predict Label: wolf, Unlearn Predict Label: fox', fontsize=26)
    plt.axis('off')  # Hide axes
    plt.savefig('test.png')
    print('acc  ',  acc / 32.0)

    
    return model 
###############################################


def ssd_tuning(
    model,
    forget_train_dl,
    full_train_dl,
    device,
    train_layer,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1, 
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": 1,
        "selection_weighting": 3,
        "train_layer": train_layer 
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    ssd = ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = ssd.calc_importance(forget_train_dl)

    original_importances = ssd.calc_importance(full_train_dl)
    ssd.modify_weight(original_importances, sample_importances)

    return  model 
###############################################



def ssd_graph_tuning(cfg, i,
    sense_models,sense_clis,
    models,clis,
    device,
    train_layer,
    sense_classes,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1, 
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": 1,
        "selection_weighting": i,
        "train_layer": train_layer 
    }
    # 先计算敏感模型的FIM
    for i in range(len(sense_models)):
        sense_model = sense_models[i]
        sense_cli = sense_clis[i]
        optimizer = torch.optim.SGD(sense_model.parameters(), lr=0.1)

        ssd = ParameterPerturber(sense_model, optimizer, device, parameters)
        sense_model = sense_model.eval()
        forget_train_dl = sense_cli.test_loader_sense
        start_time = time.time()
        sample_importances = ssd.calc_importance(forget_train_dl)
        full_train_dl = sense_cli.testloader
        ssd.test=False
        
        original_importances = ssd.calc_importance(full_train_dl)
        ssd.modify_weight(original_importances, sample_importances)
        end_time = time.time()
        execution_time = end_time - start_time
        print("训练时间", execution_time)
        sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.test_loader_sense)
        remove_sense_loader(sense_cli, sense_classes)
        sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.testloader)
        for j in range(len(models)):
            extend_model = models[j]
            cli = clis[j]
            extend_optimizer = torch.optim.SGD(extend_model.parameters(), lr=0.1)
            extend_ssd = ParameterPerturber(extend_model, extend_optimizer, device, parameters)
            extend_model = extend_model.eval()
            full_train_dl = cli.testloader
            start_time = time.time()
            original_importances = extend_ssd.calc_importance(full_train_dl)
            end_time = time.time()
            # forget_train_dl = cli.test_loader_sense
            # sample_importances = ssd.calc_importance(forget_train_dl)
            
            extend_ssd.modify_weight(original_importances, sample_importances)
            
            execution_time = end_time - start_time
            print("训练时间", execution_time)
            cli.test(cfg,extend_model, cli.criterion, cli.test_loader_sense)
            remove_sense_loader(cli, sense_classes)
            cli.test(cfg,extend_model, cli.criterion, cli.testloader)




def ssd_graph_tuning_multi( cfg,
    sense_models,sense_clis,
    models,clis,
    device,
    train_layer,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1, 
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": 1,
        "selection_weighting": 1,
        "train_layer": train_layer 
    }
    # 先计算敏感模型的FIM
    pre_fim = []
    for i in range(len(sense_models)):
        
        sense_model = sense_models[i]
        sense_cli = sense_clis[i]
  
        optimizer = torch.optim.SGD(sense_model.parameters(), lr=0.1)

        ssd = ParameterPerturber(sense_model, optimizer, device, parameters)
        sense_model = sense_model.eval()
        forget_train_dl = sense_cli.test_loader_sense
        start_time = time.time()
        sample_importances = ssd.calc_importance(forget_train_dl)
        full_train_dl = sense_cli.testloader
        original_importances = ssd.calc_importance(full_train_dl)
        pre_fim.append(sample_importances)
        ssd.modify_weight(original_importances, sample_importances)
        end_time = time.time()
        execution_time = end_time - start_time
        print("训练时间", execution_time)
        sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.test_loader_sense)
        
        unique_labels = set()

        # Iterate through the DataLoader
        for images, labels in sense_cli.test_loader_sense:
            unique_labels.update(labels.tolist())  # Convert tensor to list and update the set

        # Convert the set to a sorted list for readability
        sense_classes = sorted(list(unique_labels))
        
        
        remove_sense_loader(sense_cli, sense_classes)
        sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.testloader)
  
    start_time = time.time()
    for l in clis[0].train_layer:
        for j in range(1, len(pre_fim)):
            pre_fim[0][l] = torch.max(pre_fim[0][l], pre_fim[j][l]) 
    
    for l in clis[0].train_layer:
        for j in range(1, len(pre_fim)):
            pre_fim[0][l] /= 5  # alexnet 50  0%  # densenet  #resnet 1
    sample_importances = pre_fim[0]
    
    for j in range(len(models)):
        extend_model = models[j]
        cli = clis[j]
        extend_optimizer = torch.optim.SGD(extend_model.parameters(), lr=0.1)
        extend_ssd = ParameterPerturber(extend_model, extend_optimizer, device, parameters)
        extend_model = extend_model.eval()
        full_train_dl = cli.testloader
        
        original_importances = extend_ssd.calc_importance(full_train_dl)
        
        #forget_train_dl = cli.test_loader_sense
        #sample_importances = extend_ssd.calc_importance(forget_train_dl)
        
        extend_ssd.modify_weight(original_importances, sample_importances)
        models[j] = copy.copy(extend_model)
        end_time = time.time()
        execution_time = end_time - start_time
        print("训练时间", execution_time)
        cli.test(cfg,extend_model, cli.criterion, cli.test_loader_sense)
        
    
        
        remove_sense_loader(cli, sense_classes)
        cli.test(cfg,extend_model, cli.criterion, cli.testloader)






def ssd_graph_tuning_multi_abalation( cfg,
    sense_models,sense_clis,
    models,clis,
    device,
    train_layer,
    **kwargs,
):
    parameters = {
        "lower_bound": 0.1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1, 
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": 1,
        "selection_weighting": 1,
        "train_layer": train_layer 
    }
    # 先计算敏感模型的FIM
    pre_fim = []
    for i in range(len(sense_models)):
        
        sense_model = sense_models[i]
        sense_cli = sense_clis[i]
  
        optimizer = torch.optim.SGD(sense_model.parameters(), lr=0.1)

        ssd = ParameterPerturber(sense_model, optimizer, device, parameters)
        sense_model = sense_model.eval()
        forget_train_dl = sense_cli.test_loader_sense
        start_time = time.time()
        sample_importances = ssd.calc_importance(forget_train_dl)
        full_train_dl = sense_cli.testloader
        original_importances = ssd.calc_importance(full_train_dl)
        pre_fim.append(sample_importances)
        ssd.modify_weight(original_importances, sample_importances)
        end_time = time.time()
        execution_time = end_time - start_time
        print("训练时间", execution_time)
        sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.test_loader_sense)
        
        unique_labels = set()

        # Iterate through the DataLoader
        for images, labels in sense_cli.test_loader_sense:
            unique_labels.update(labels.tolist())  # Convert tensor to list and update the set

        # Convert the set to a sorted list for readability
        sense_classes = sorted(list(unique_labels))
        
        
        remove_sense_loader(sense_cli, sense_classes)
        sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.testloader)
  
        for j in range(len(models)):
            extend_model = models[j]
            cli = clis[j]
            extend_optimizer = torch.optim.SGD(extend_model.parameters(), lr=0.1)
            extend_ssd = ParameterPerturber(extend_model, extend_optimizer, device, parameters)
            extend_model = extend_model.eval()
            full_train_dl = cli.testloader
            
            original_importances = extend_ssd.calc_importance(full_train_dl)
            
            #forget_train_dl = cli.test_loader_sense
            #sample_importances = extend_ssd.calc_importance(forget_train_dl)
            
            extend_ssd.modify_weight(original_importances, sample_importances)
            models[j] = copy.copy(extend_model)
            end_time = time.time()
            execution_time = end_time - start_time
            print("训练时间", execution_time)
            cli.test(cfg,extend_model, cli.criterion, cli.test_loader_sense)
            
        
            
            remove_sense_loader(cli, sense_classes)
            cli.test(cfg,extend_model, cli.criterion, cli.testloader)


def remove_loader_by_index(test_sense):
    from torch.utils.data import DataLoader, TensorDataset
    filtered_data = []
    filtered_labels = []

    # 迭代 testloader 来提取标签为 sense_clas的数据
    for images, labels in test_sense:
        
        filtered_data.append(images[:32,:,:,:])
        filtered_labels.append(labels[:32])
        
    # 将数据和标签连接成单个张量
    filtered_data = torch.cat(filtered_data)
    filtered_labels = torch.cat(filtered_labels)

    # 创建新的 TensorDataset 和 DataLoader
    filtered_dataset = TensorDataset(filtered_data, filtered_labels)
    filtered_loader = DataLoader(filtered_dataset, batch_size=64, shuffle=True, num_workers=2)
    return  filtered_loader


def get_distance():
    
    return 

def ga(cfg, 
    sense_models,sense_clis,
    models,clis,
    device,
    train_layer,
    sense_classes,
    **kwargs,
):
   
    criterion = nn.CrossEntropyLoss()
    for i in range(len(sense_models)):
        
        sense_model = sense_models[i]
        sense_cli = sense_clis[i]
        sense_model = sense_model.train()
        forget_train_dl = sense_cli.test_loader_sense
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(sense_cli.testloader):
            
            max_length = min(max([len(text) for text in x]) + 2, 128)
            if cfg.model['_target_'] == 'models.bert.Bert':
                batch_input_ids, batch_att_mask = [], []
                for text in x:
                    encoding = sense_model.get_tokenizer().encode_plus(
                        text,
                        add_special_tokens=True,  # Add [CLS] and [SEP]
                        padding='max_length',  # Pad to max_length
                        truncation=True,  # Truncate longer sequences
                        max_length=max_length,
                        return_tensors='pt'  # Return PyTorch tensors
                    )
                    batch_input_ids.append(encoding['input_ids'])
                    batch_att_mask.append(encoding['attention_mask'])
                x = torch.cat(batch_input_ids)
                att_mask = torch.cat(batch_att_mask).to('cuda:7')
                
            x, y = x.to(device), y.to(device)
            if cfg.data.dataset != 'YAHOO':
                x = x.float()
            if cfg.model['_target_'] == 'models.bert.Bert':
                outputs1 = sense_model(x, att_mask)
            else:
                outputs1 = sense_model(x) 
            loss1 = criterion(outputs1, y)
            sense_model.zero_grad()
            loss1.backward()
            grads1 = [param.grad.clone() for param in sense_model.parameters()]
            break
        
        unique_labels = set()
        # Iterate through the DataLoader
        for images, labels in sense_cli.test_loader_sense:
            unique_labels.update(labels.tolist())  # Convert tensor to list and update the set
        # Convert the set to a sorted list for readability
        sense_classes = sorted(list(unique_labels))
        sense_cli.testloader.dataset.name = cfg.data.dataset
        remove_sense_loader(sense_cli, sense_classes)
        
        #sense_model = hydra.utils.instantiate(cfg.model).to(device)
        optimizer = torch.optim.SGD(sense_model.parameters(), lr=8e0)
        for batch_idx, (x, y) in enumerate(forget_train_dl):
            
            max_length = min(max([len(text) for text in x]) + 2, 128)
            if cfg.model['_target_'] == 'models.bert.Bert':
                batch_input_ids, batch_att_mask = [], []
                for text in x:
                    encoding = sense_model.get_tokenizer().encode_plus(
                        text,
                        add_special_tokens=True,  # Add [CLS] and [SEP]
                        padding='max_length',  # Pad to max_length
                        truncation=True,  # Truncate longer sequences
                        max_length=max_length,
                        return_tensors='pt'  # Return PyTorch tensors
                    )
                    batch_input_ids.append(encoding['input_ids'])
                    batch_att_mask.append(encoding['attention_mask'])
                x = torch.cat(batch_input_ids)
                att_mask = torch.cat(batch_att_mask).to('cuda:7')
                
            x, y = x.to(device), y.to(device)
            if cfg.data.dataset != 'YAHOO':
                x = x.float()
            if cfg.model['_target_'] == 'models.bert.Bert':
                outputs1 = sense_model(x, att_mask)
            else:
                outputs1 = sense_model(x)
            loss1 = -criterion(outputs1, y)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            acc = sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.test_loader_sense)
            grads2 = [param.grad.clone() for param in sense_model.parameters()]
            if acc < 20:
                break
        grad_diffs = [g1 - g2 for g1, g2 in zip(grads1, grads2)]
        end_time = time.time()
        execution_time = end_time - start_time
        print("训练时间", execution_time)
        
        sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.testloader)
  
        for j in range(len(models)):
            extend_model = models[j]
            cli = clis[j]
            
            
            extend_model = extend_model.train()
            full_train_dl = cli.testloader
            start_time = time.time()
            with torch.no_grad():
                for param2, gd in zip(extend_model.parameters(), grad_diffs):
                    param2.add_(2*gd)
            cli.test(cfg,extend_model, cli.criterion, cli.test_loader_sense)
                
            end_time = time.time()
            execution_time = end_time - start_time
            print("训练时间", execution_time)

            cli.testloader.dataset.name = cfg.data.dataset
            remove_sense_loader(cli, sense_classes)
            cli.test(cfg,extend_model, cli.criterion, cli.testloader)



def fix_partial_model(train_list, net):
    print(train_list)
    index = [0,1,2,3]
    for name, weights in net.named_parameters():
        # if name == 'linear.weight':
        #     for i in range(weights.shape[0]):
        #         if i not in index:
        #             weights.detach()[i,:].requires_grad = False
            
        if name not in train_list:
            
            weights.requires_grad = False



def finetune(cfg, 
    sense_models,sense_clis,
    models,clis,
    device,
    train_layer,
    sense_classes,
    **kwargs,
):
   
    criterion = nn.CrossEntropyLoss()
    for i in range(len(sense_models)):
        
        sense_model = sense_models[i]
        sense_cli = sense_clis[i]
        sense_model = sense_model.train()
        forget_train_dl = sense_cli.test_loader_sense
        start_time = time.time()
        
   
        
        unique_labels = set()
        # Iterate through the DataLoader
        for images, labels in sense_cli.test_loader_sense:
            unique_labels.update(labels.tolist())  # Convert tensor to list and update the set
        # Convert the set to a sorted list for readability
        sense_classes = sorted(list(unique_labels))
        remove_sense_loader(sense_cli, sense_classes)
        
        #sense_model = hydra.utils.instantiate(cfg.model).to(device)
        lr = 0.03
        optimizer = torch.optim.SGD(sense_model.parameters(), lr=lr)
        #fix_partial_model(train_layer, sense_model)
        epoch = 100
        for i in range(epoch):
            for batch_idx, (x, y) in enumerate(sense_cli.testloader):
                x, y = x.to(device).float(), y.to(device)
                outputs1 = sense_model(x)
                loss1 = criterion(outputs1, y)
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()
                acc = sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.test_loader_sense)
                if acc < 80:
                    break

        end_time = time.time()
        execution_time = end_time - start_time
        test_acc = sense_cli.test(cfg,sense_model, sense_cli.criterion, sense_cli.testloader)
        print("训练时间", execution_time)
        print("unlearn performance", test_acc)
        for j in range(len(models)):
            extend_model = models[j]
            cli = clis[j]
            
            
            extend_model = extend_model.train()
            full_train_dl = cli.testloader
            start_time = time.time()
            optimizer = torch.optim.SGD(extend_model.parameters(), lr=lr)
            #fix_partial_model(train_layer, extend_model)
            remove_sense_loader(cli, sense_classes)
            for i in range(epoch):
                for batch_idx, (x, y) in enumerate(cli.testloader):
                    x, y = x.to(device).float(), y.to(device)
                    outputs1 = extend_model(x)
                    loss1 = criterion(outputs1, y)
                    optimizer.zero_grad()
                    loss1.backward()
                    optimizer.step()
                    acc = cli.test(cfg,extend_model, cli.criterion, cli.test_loader_sense)
                    if acc < 80:
                        break
                
            end_time = time.time()
            execution_time = end_time - start_time
            print("训练时间", execution_time)

            
           
            test_acc=cli.test(cfg,extend_model, cli.criterion, cli.testloader)
            print("unlearn performance", test_acc)


def distillation_loss(y, teacher_scores, targets, T, alpha):
    ce_loss = nn.CrossEntropyLoss()(y, targets)
    # KL 散度损失
    kd_loss = nn.KLDivLoss()(torch.log_softmax(y / T, dim=1), torch.softmax(teacher_scores / T, dim=1)) * (T * T)
    # 总损失
    return ce_loss * (1. - alpha) + kd_loss * alpha

T = 2.0  # 温度参数
alpha = 0.8  # 权重参数
def distill(cfg, 
    sense_models,sense_clis,
    models,clis,
    device,
    train_layer,
    sense_classes,
    **kwargs,
):
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    for i in range(len(sense_models)):
        
        sense_model = sense_models[i]
        sense_cli = sense_clis[i]
        sense_model.eval()
        forget_train_dl = sense_cli.test_loader_sense
        
        
        stu = hydra.utils.instantiate(cfg.model).to(device)
        stu_optimizer = optim.SGD(stu.parameters(), lr=0.01)
        stu.train()
        unique_labels = set()
        # Iterate through the DataLoader
        for images, labels in sense_cli.test_loader_sense:
            unique_labels.update(labels.tolist())  # Convert tensor to list and update the set
        # Convert the set to a sorted list for readability
        sense_classes = sorted(list(unique_labels))
        remove_sense_loader(sense_cli, sense_classes)
        start_time = time.time()
        epoch = 100
        for i in range(epoch):
            j = 0
            for batch_idx, (x, y) in enumerate(sense_cli.testloader):
                x, y = x.to(device).float(), y.to(device)
                outputs1 = stu(x)

                inference_time_start = time.time()
                stu_optimizer.zero_grad()
                with torch.no_grad():
                    teacher_outputs = sense_model(x)
                inference_time_end = time.time()
                
                kl_loss_start = time.time()
                loss = distillation_loss(outputs1, teacher_outputs, y, T, alpha)
                loss.backward()
                stu_optimizer.step()
                kl_loss_end = time.time()
                j += 1
                
            acc = sense_cli.test(cfg, stu, sense_cli.criterion, sense_cli.test_loader_sense)
            test_acc = sense_cli.test(cfg,stu, sense_cli.criterion, sense_cli.testloader)
            # if acc < 10 and test_acc > 80:
            #     break
            print("前向传播时间", (inference_time_end-inference_time_start)*j)
            print("KL_loss time", (kl_loss_end-kl_loss_start))
            
        end_time = time.time()
        execution_time = end_time - start_time
        test_acc = sense_cli.test(cfg,stu, sense_cli.criterion, sense_cli.testloader)
        print("训练时间", execution_time)
        
        print("unlearn performance", test_acc)
        for j in range(len(models)):
            extend_model = models[j]
            cli = clis[j]
            
            
            extend_model = extend_model.eval()
            stu = hydra.utils.instantiate(cfg.model).to(device)
            stu_optimizer = optim.SGD(stu.parameters(), lr=cfg.optimizer.lr)
            stu.train()
            full_train_dl = cli.testloader
            start_time = time.time()
            
         
            remove_sense_loader(cli, sense_classes)
            for i in range(epoch):
                for batch_idx, (x, y) in enumerate(cli.testloader):
                    x, y = x.to(device).float(), y.to(device)
                    outputs1 = stu(x)
                
                    stu_optimizer.zero_grad()
                    with torch.no_grad():
                        teacher_outputs = extend_model(x)
                    loss = distillation_loss(outputs1, teacher_outputs, y, T, alpha)
                    loss.backward()
                    stu_optimizer.step()
                acc = cli.test(cfg, stu, cli.criterion, cli.test_loader_sense)
                test_acc = cli.test(cfg,stu, cli.criterion, cli.testloader)
                if acc < 10 and test_acc > 80:
                    break
                
            end_time = time.time()
            execution_time = end_time - start_time
            print("训练时间", execution_time)

            
           
            test_acc=cli.test(cfg,stu, cli.criterion, cli.testloader)
            print("unlearn performance", test_acc)

