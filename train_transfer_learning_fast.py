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

    #=====================+
    # =================== 联邦学习代码 =============================================
    # 将 CIFAR-100 数据集分为联邦学习的客户端, 一共5个用户,6次操作
    # 要删除的类别列表
    sense_classes = list(range(5))  # 类别0到10
    num_clients = 5
    client_data = np.array_split(client_data, num_clients)
    client_targets = np.array_split(client_targets, num_clients)
    test_client_data = client_data
    test_client_targets = client_targets
    clients = []
    for i in range(num_clients):
        net = hydra.utils.instantiate(cfg.model).to('cuda:7')
        optimizer = hydra.utils.instantiate(cfg.optimizer, net.parameters())
        criterion = nn.CrossEntropyLoss()
        scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer)
        train_layer = cfg.train_layer
        if train_layer == 'all':
            train_layer = [name for name, module in net.named_parameters()]
        net = net
        
        dataset = FederatedDataset(client_data[0], client_targets[0],cfg.data.dataset)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
        test_dataset = FederatedDataset(test_client_data[0], test_client_targets[0],cfg.data.dataset)
        eval_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
        
        if cfg.data.dataset == 'cifar100' or cfg.data.dataset == 'tinyimagenet':
            
            if i == 0:
                rm_class = [i for i in range(90,100)]
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
      
            if i == 1:
                rm_class = [i for i in range(92,100)]
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
          
            if i == 2:
                rm_class = [i for i in range(94,100)]
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
        
            if i == 3:
                rm_class = [i for i in range(98,100)]
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
        if cfg.data.dataset == 'YAHOO':
            
            if i == 0:
                rm_class = [6,7,8,9]
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
      
            if i == 1:
                rm_class = [6,7,8]
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
          
            if i == 2:
                rm_class = [6,7]
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)
        
            if i == 3:
                rm_class = [6]
                train_loader, eval_loader = remove_sense_get_loader(dataset, test_dataset, rm_class, batch_size, num_workers)

        # 定义目标类别
        target_classes = sense_classes

        # 对验证集进行子集选择，获取目标类别的数据
        val_indices = [idx for idx, (_, label) in enumerate(test_dataset) if label in target_classes]
        val_subset = Subset(test_dataset, val_indices)

        # 创建数据加载器
        eval_loader_sense = DataLoader(val_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, persistent_workers=True)


        client = Client(net, criterion, optimizer, scheduler, train_loader, eval_loader, eval_loader_sense, train_layer, tmp_path, save_model_accs)
        clients.append(client)

   
    # 用户1有敏感数据，且用户2，3，4使用用户1迁移的模型进行训练
    # clients[0].train_transfer(epoch, all_epoch,name='u0_start',is_transfer=False, cfg=cfg)
    # d = copy.deepcopy(clients[0].net)
    # clients[1].train_transfer(epoch, all_epoch,aggre_wei=[d], name='u1_trans', cfg=cfg)
    # clients[2].train_transfer(epoch, all_epoch, aggre_wei=[d],name='u2_trans', cfg=cfg)
    # clients[3].train_transfer(epoch, all_epoch,aggre_wei=[d],name='u3_trans', cfg=cfg)
    # clients[4].train_transfer(epoch, all_epoch,aggre_wei= [d],name='u4_trans', cfg=cfg)

    # ## ======================================= unlearning代码 =================================================
    # ## 删除user 1, 2中的敏感数据然后重新训练
  
    # ## 创建经过筛选后的数据加载器
  
    tmp_path = '/home/zjy/lmy/UnlearningDiffusion/test/exp_result_cifar100/alexnet/param_data_trans/tmp_2024-06-07_17-44-33'
    his_path = tmp_path + '/'
    # original model acc
    clients[0].net = torch.load(his_path + 'u0_start.pth').to('cuda:7')
    clients[1].net = torch.load(his_path + 'u1_trans.pth').to('cuda:7')
    clients[2].net = torch.load(his_path + 'u2_trans.pth').to('cuda:7')
    clients[3].net = torch.load(his_path + 'u3_trans.pth').to('cuda:7')
    
    
    # start_time = time.time()
    # clients[0].train_transfer(epoch, all_epoch,name='u0_start',is_unlearn=True,sense_labels=sense_classes,cfg=cfg)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("训练时间", execution_time)
    # start_time = time.time()
    # clients[1].train_transfer(epoch, all_epoch,name='u1_trans',is_unlearn=True,sense_labels=sense_classes, cfg=cfg)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("训练时间", execution_time)
    # start_time = time.time()
    # clients[2].train_transfer(epoch, all_epoch, name='u2_trans',is_unlearn=True,sense_labels=sense_classes, cfg=cfg)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("训练时间", execution_time)
    # start_time = time.time()
    # clients[3].train_transfer(epoch, all_epoch,name='u3_trans',is_unlearn=True,sense_labels=sense_classes, cfg=cfg)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("训练时间", execution_time)
   
    
    # print('original model acc')
    # for i in range(4):
    #     clients[i].test(cfg,clients[i].net, clients[i].criterion, clients[i].testloader)
    #     clients[i].test(cfg,clients[i].net, clients[i].criterion, clients[i].test_loader_sense)
        
        
    # print('unlearn model acc')
    for i in range(1,30):
        a=copy.deepcopy(clients[0].net)
        b=copy.deepcopy(clients[1].net)
        c=copy.deepcopy(clients[2].net)
        d=copy.deepcopy(clients[3].net)
        ssd_graph_tuning(cfg,i, [a],[clients[0]], [b,c,d], [clients[1],clients[2],clients[3]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
     
    #ssd_graph_tuning(cfg,[clients[0].net],[clients[0]], [clients[1].net,clients[2].net,clients[3].net],clients[1:4],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #ga(cfg,[clients[0].net],[clients[0]], [clients[1].net], [clients[1]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #ga(cfg,[clients[0].net],[clients[0]], [clients[2].net], [clients[2]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #ga(cfg,[clients[0].net],[clients[0]], [clients[3].net], [clients[3]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #finetune(cfg,[clients[0].net],[clients[0]], [clients[1].net,clients[2].net,clients[3].net], [clients[1],clients[2],clients[3]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    #distill(cfg,[clients[0].net],[clients[0]], [clients[1].net,clients[2].net,clients[3].net], [clients[1],clients[2],clients[3]],  next(clients[0].net.parameters()).device, clients[0].train_layer, sense_classes)
    

 



if __name__ == "__main__":

        
    # 定义训练数据集
    # 假设训练数据集为一个大小为 [N, input_dim] 的张量

    # 实例化Autoencoder模型
    training_for_data()
    