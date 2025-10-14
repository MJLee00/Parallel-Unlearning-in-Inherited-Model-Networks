import hydra
from omegaconf import DictConfig
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset
import torch
import os 
import time
import datetime
from utils import *
import json 
import glob
from utils.clinet import Client,FederatedDataset,average_weights
import numpy as np 
from sklearn.model_selection import train_test_split
from util import *
import random 
import gc
import copy

def set_device(device_config):
    # set the global cuda device
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
    torch.cuda.set_device(device_config.cuda)
    torch.set_float32_matmul_precision('medium')
    # warnings.filterwarnings("always")



def init_experiment(cfg, **kwargs):
    cfg = cfg   
    seed = 59 #53
    random.seed(seed)
    
    # 设置随机种子
    torch.manual_seed(seed)

    # 如果您同时使用了CUDA，还需要设置 CUDA 的随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 可选：如果您希望在多次运行时得到相同的结果，还可以设置以下两个参数
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("config:")
    for k, v in cfg.items():
        print(k, v)
    print("=" * 20)

    print("kwargs:")
    for k, v in kwargs.items():
        print(k, v)
    print("=" * 20)


    # set device
    set_device(cfg.device)

    # set process title
    #set_processtitle(cfg)

@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_for_data(config: DictConfig):
    train_task_for_data(config)

def train_task_for_data(cfg, **kwargs):
    init_experiment(cfg, **kwargs)
    cfg = cfg.task

    epoch = cfg.epoch
    save_num = cfg.save_num_model
    all_epoch = epoch + save_num
    batch_size = getattr(cfg, 'batch_size', 320)
    num_workers = getattr(cfg, 'num_workers', 4)

    best_acc = 0

    # 加载 CIFAR-100 训练集
    if cfg.data.dataset == 'cifar100':
        train_dataset, val_dataset = get_cifar100()
    elif cfg.data.dataset == 'tinyimagenet':
        # 加载 tinyimagenet 
        train_dataset, val_dataset = get_tiny_imagenet()
    elif cfg.data.dataset == 'YAHOO':
       
        train_dataset, val_dataset = get_yahoo()
  

  
 
    data_path = getattr(cfg, 'save_root', 'param_data')

    tmp_path = os.path.join(data_path, 'tmp_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    # tmp_path = os.path.join(data_path, 'tmp')
    final_path = os.path.join(data_path, cfg.data.dataset)

    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(final_path, exist_ok=True)


    save_model_accs = []
    parameters = []

    # 划分训练数据用于客户端和测试客户端数据
    client_data, test_client_data, client_targets, test_client_targets = train_test_split(train_dataset.data, train_dataset.targets, test_size=0.2, random_state=42)

    #======================================== 联邦学习代码 =============================================
    # 将 CIFAR-100 数据集分为联邦学习的客户端, 一共5个用户,6次操作
    # 要删除的类别列表
    classes_to_remove = list(range(4))  # 类别0到10
    num_clients = 5  
    client_data = np.array_split(client_data, num_clients)
    client_targets = np.array_split(client_targets, num_clients)
    test_client_data = client_data
    test_client_targets = client_targets
    clients = []
    for i in range(num_clients):
        net = hydra.utils.instantiate(cfg.model)
        optimizer = hydra.utils.instantiate(cfg.optimizer, net.parameters())
        criterion = nn.CrossEntropyLoss()
        scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer)
        train_layer = cfg.train_layer
        if train_layer == 'all':
            train_layer = [name for name, module in net.named_parameters()]
        net = net.to('cuda:7')
        
        dataset = FederatedDataset(client_data[i], client_targets[i], cfg.data.dataset)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
        test_dataset = FederatedDataset(test_client_data[i], test_client_targets[i],cfg.data.dataset)
        eval_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
        # 除了用户1和2有敏感数据，其他都没有
        if i != 1 and i != 2:
            # 创建经过筛选后的数据加载器
            train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, classes_to_remove, batch_size, num_workers)

        # 定义目标类别
        target_classes = classes_to_remove

        # 对验证集进行子集选择，获取目标类别的数据
        val_indices = [idx for idx, (_, label) in enumerate(test_dataset) if label in target_classes]
        val_subset = Subset(test_dataset, val_indices)

        # 创建数据加载器
        eval_loader_sense = DataLoader(val_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, persistent_workers=True)


        client = Client(net, criterion, optimizer, scheduler, train_loader, eval_loader, eval_loader_sense, train_layer, tmp_path, save_model_accs)
        clients.append(client)
    # all_node = []    
    # for i in range(50):
    #     clients[0].train(epoch, all_epoch,name='a'+str(i), cfg=cfg)
    #     all_node.append(copy.deepcopy(clients[0].net))
    #     if len(all_node) > 2:
    #         clients[i%2+1].train(epoch, all_epoch,[all_node[-1],all_node[-2]],name='d'+str(i), cfg=cfg)
    #         all_node.append(copy.deepcopy(clients[i%2+1].net))
    #     else:
    #         clients[i%2+1].train(epoch, all_epoch,name='d'+str(i), cfg=cfg)
    #         all_node.append(copy.deepcopy(clients[i%2+1].net))
        
        
    # ======================================= unlearning代码 =================================================
      
    # ##创建经过筛选后的数据加载器
    tmp_path = '/home/zjy/lmy/UnlearningDiffusion/test/param_data_fed/tinyimagenet-dense161'
    his_path = tmp_path + '/'
    all_nodes = []
    for i in range(50):
        all_nodes.append(torch.load(his_path+'d'+str(i)+'.pth').to('cuda:7'))
        

    print('original model acc')
    for i in range(50):
        clients[i%2+1].test(cfg,all_nodes[i%2], clients[i%2+1].criterion, clients[i%2+1].testloader)
        clients[i%2+1].test(cfg,all_nodes[i%2], clients[i%2+1].criterion, clients[i%2+1].test_loader_sense)
        
    ssd_graph_tuning(cfg,[all_nodes[0]],[clients[1]], [all_nodes[i] for i in range(1,50)], [clients[i%2+1] for i in range(1,50)],  next(clients[1].net.parameters()).device, clients[1].train_layer, classes_to_remove)

  

if __name__ == "__main__":

        
    # 定义训练数据集
    # 假设训练数据集为一个大小为 [N, input_dim] 的张量

    # 实例化Autoencoder模型
    training_for_data()
    