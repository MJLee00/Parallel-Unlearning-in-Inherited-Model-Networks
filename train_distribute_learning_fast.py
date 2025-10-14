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
    num_workers = getattr(cfg, 'num_workers', 1)

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
    all_client_data = np.array_split(client_data, 1)
    all_client_targets = np.array_split(client_targets, 1)
    #======================================== 联邦学习代码 =============================================
    # 将 CIFAR-100 数据集分为联邦学习的客户端, 一共5个用户,6次操作
    # 要删除的类别列表
    sense_classes = list(range(1))  # 类别0到10
    num_clients = 5  
    client_data = np.array_split(client_data, num_clients)
    client_targets = np.array_split(client_targets, num_clients)
    client_data[0] = all_client_data[0]
    client_targets[0] = all_client_targets[0]
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
        # 分布式学习中，不同客户端有不同数据集
        dataset = FederatedDataset(client_data[i], client_targets[i], cfg.data.dataset)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
        test_dataset = FederatedDataset(test_client_data[i], test_client_targets[i],cfg.data.dataset)
        eval_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)

        if cfg.data.dataset == 'cifar100':
            
            # 用户0有0-100类的数据
            cur_sense = sense_classes
            # 用户1有0-80类的数据 且只有敏感标签1的数据
            if i == 1:
                rm_class = [i for i in range(80,100)]
            
                #cur_sense = sense_classes[:int(len(sense_classes)/3.0)+1]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
            # 用户2有0-90类的数据 且只有敏感标签2的数据
            if i == 2:
                rm_class = [i for i in range(90,100)]
                #cur_sense = sense_classes[int(len(sense_classes)/3.0)+1:int(len(sense_classes)/3.0)+2]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
            # 用户3有0-10, 20-100
            if i == 3:
                rm_class = [i for i in range(10,20)]
                #cur_sense = sense_classes[int(len(sense_classes)/3.0)+2:]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
        elif cfg.data.dataset == 'tinyimagenet':
           
            # 用户0有0-100类的数据
            cur_sense = sense_classes
            # 用户1有0-80类的数据 且只有敏感标签1的数据
            if i == 1:
                rm_class = [i for i in range(180,200)]
            
                #cur_sense = sense_classes[:int(len(sense_classes)/3.0)+1]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
            # 用户2有0-90类的数据 且只有敏感标签2的数据
            if i == 2:
                rm_class = [i for i in range(190,200)]
                #cur_sense = sense_classes[int(len(sense_classes)/3.0)+1:int(len(sense_classes)/3.0)+2]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
            # 用户3有0-10, 20-200
            if i == 3:
                rm_class = [i for i in range(10,20)]
                #cur_sense = sense_classes[int(len(sense_classes)/3.0)+2:]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
        elif  cfg.data.dataset == 'YAHOO':
           
            # 用户0有0-100类的数据
            cur_sense = sense_classes
            # 用户1有0-80类的数据 且只有敏感标签1的数据
            if i == 1:
                rm_class = [9]
            
                #cur_sense = sense_classes[:int(len(sense_classes)/3.0)+1]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
            # 用户2有0-90类的数据 且只有敏感标签2的数据
            if i == 2:
                rm_class = [8]
                #cur_sense = sense_classes[int(len(sense_classes)/3.0)+1:int(len(sense_classes)/3.0)+2]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
            # 用户3有0-10, 20-200
            if i == 3:
                rm_class = [7]
                #cur_sense = sense_classes[int(len(sense_classes)/3.0)+2:]
                #rm_class.extend(cur_sense)
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
        
        
        # 定义目标类别 
        target_classes = cur_sense

        # 对验证集进行子集选择，获取目标类别的数据
        val_indices = [idx for idx, (_, label) in enumerate(test_dataset) if label in target_classes]
        val_subset = Subset(test_dataset, val_indices)

        # 创建数据加载器
        eval_loader_sense = DataLoader(val_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, persistent_workers=True)


        client = Client(net, criterion, optimizer, scheduler, train_loader, eval_loader, eval_loader_sense, train_layer, tmp_path, save_model_accs)
        clients.append(client)


    # u0用户下发任务给1，2，3；1，2，3执行完聚合到u0
    
    # clients[1].train_continue(epoch, all_epoch, name='u1_distri', cfg=cfg)
    # u1 = clients[1].net
    # clients[2].train_continue(epoch, all_epoch,name='u2_distri', cfg=cfg)
    # u2 = clients[2].net
    # clients[3].train_continue(epoch, all_epoch,name='u3_distri', cfg=cfg)
    # u3 = clients[3].net
    

    # aggre = average_weights([u1,u2,u3])
    # clients[0].net.load_state_dict(aggre)
    # clients[0].train_continue(epoch, all_epoch,name='u0_start', cfg=cfg)
    # u0 = clients[0].net
    # clients[0].test(cfg,u0, clients[0].criterion, clients[0].testloader)
    # clients[0].test(cfg,u0, clients[0].criterion, clients[0].test_loader_sense)
    # ======================================= unlearning代码 =================================================
    # ##创建经过筛选后的数据加载器
    tmp_path = '/home/zjy/lmy/UnlearningDiffusion/test/param_data/tmp_2024-08-28_15-32-02'
    his_path = tmp_path + '/'
    
    r1 = torch.load(his_path + 'u1_distri.pth')

    r2 = torch.load(his_path + 'u2_distri.pth')
    r3 = torch.load(his_path + 'u3_distri.pth')
    r0 = torch.load(his_path + 'u0_start.pth')
    

    # print('original model acc')
    # clients[0].test(cfg,r0, clients[0].criterion, clients[0].testloader)
    # clients[0].test(cfg,r0, clients[0].criterion, clients[0].test_loader_sense)
    
    # clients[1].test(cfg,r1, clients[1].criterion, clients[1].testloader)
    # clients[1].test(cfg,r1, clients[1].criterion, clients[1].test_loader_sense)
    
    # clients[2].test(cfg,r2, clients[2].criterion, clients[2].testloader)
    # clients[2].test(cfg,r2, clients[2].criterion, clients[2].test_loader_sense)
    
    # clients[3].test(cfg,r3, clients[3].criterion, clients[3].testloader)
    # clients[3].test(cfg,r3, clients[3].criterion, clients[3].test_loader_sense)
    
    #ssd_graph_tuning(cfg,[r0],[clients[0]], [r1,r2,r3], [clients[1],clients[2],clients[3]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #ga(cfg,[r0],[clients[0]], [r1], [clients[1]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #ga(cfg,[r0],[clients[0]], [r2], [clients[2]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    ga(cfg,[r0],[clients[0]], [r3], [clients[3]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #distill(cfg,[r0],[clients[0]], [r1,r2,r3], [clients[1],clients[2],clients[3]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #finetune(cfg,[r0],[clients[0]], [r1,r2,r3], [clients[1],clients[2],clients[3]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)

   
  

if __name__ == "__main__":

        
    # 定义训练数据集
    # 假设训练数据集为一个大小为 [N, input_dim] 的张量

    # 实例化Autoencoder模型
    training_for_data()
 