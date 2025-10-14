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
import random 
import gc

def set_device(device_config):
    # set the global cuda device
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
    torch.cuda.set_device(device_config.cuda)
    torch.set_float32_matmul_precision('medium')
    # warnings.filterwarnings("always")


def set_processtitle(cfg):
    # set process title
    import setproctitle
    setproctitle.setproctitle(cfg.process_title)

def init_experiment(cfg, **kwargs):
    cfg = cfg   
    # seed = 59 #53
    # random.seed(seed)
    
    # # 设置随机种子
    # torch.manual_seed(seed)

    # # 如果您同时使用了CUDA，还需要设置 CUDA 的随机种子
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # # 可选：如果您希望在多次运行时得到相同的结果，还可以设置以下两个参数
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
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
    set_processtitle(cfg)

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

    # 创建 DataLoader
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将图像像素值归一化到 [-1, 1] 范围内
    ])

    # 加载 CIFAR-100 训练集
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

  
 
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
    classes_to_remove = list(range(2))  # 类别0到10
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

    # r1 = torch.load('/home/wuqiang/lmy/UnlearningDiffusion/param_data/tmp_2024-04-02_10-28-27/r1.pth')
    # r2 = torch.load('/home/wuqiang/lmy/UnlearningDiffusion/param_data/tmp_2024-04-02_10-28-27/r2.pth')
    # r5 = torch.load('/home/wuqiang/lmy/UnlearningDiffusion/param_data/tmp_2024-04-02_10-28-27/r5.pth')
    # r6 = torch.load('/home/wuqiang/lmy/UnlearningDiffusion/param_data/tmp_2024-04-02_10-28-27/r6.pth')
    # #clients[1].net = torch.load('/home/wuqiang/lmy/UnlearningDiffusion/param_data/tmp_2024-03-29_19-07-50_sense1/r1.pth')
    # clients[1].train(epoch, all_epoch, [r1,r2,r5,r6],name='r7')


    # clients[0].train(epoch, all_epoch,name='d')
    # d = clients[0].net
    # clients[3].train(epoch, all_epoch,name='s')
    # s = clients[3].net
    # clients[1].train(epoch, all_epoch, [d,s],name='r1')
    # r1 = clients[1].net 
    # clients[1].train(epoch, all_epoch, [r1],name='r2')
    # r2 = clients[1].net 
    # clients[4].train(epoch, all_epoch, [r1, r2],name='r4')
    # r4 = clients[4].net 
    # clients[2].train(epoch, all_epoch, [r1, r2],name='r3')
    # r3 = clients[2].net 
    # clients[1].train(epoch, all_epoch, [d, r2],name='r5')
    # r5 = clients[1].net 
    # clients[1].train(epoch, all_epoch, [s, r3],name='r6')
    # r6 = clients[1].net 
    # ======================================= unlearning代码 =================================================
    # 删除user 1, 2中的敏感数据然后重新训练
    
    # 创建经过筛选后的数据加载器
  
    his_path = tmp_path + '/'
    his_path = '/home/wuqiang/lmy/UnlearningDiffusion/param_data/tmp_2024-04-02_10-28-27/'
    clients[0].net = torch.load(his_path + 'd.pth')
    d = clients[0].net
    clients[3].net = torch.load(his_path + 's.pth')
    s = clients[3].net
    clients[1].net = torch.load(his_path + 'r1.pth')
    clients[1].train(epoch, all_epoch, name='r1', is_unlearn=True, sense_labels=classes_to_remove, cfg=cfg)
    r1 = clients[1].net 
    # clients[1].net = torch.load(his_path + 'r2.pth')
    # clients[1].train(epoch, all_epoch, name='r2', is_unlearn=True, sense_labels=classes_to_remove, cfg=cfg)
    # r2 = clients[1].net 
    # # clients[4].net = torch.load(his_path + 'r4.pth')
    # # clients[4].train(epoch, all_epoch, name='r4', is_unlearn=True, sense_labels=classes_to_remove, cfg=cfg)
    # # r4 = clients[4].net 
    # # clients[2].net = torch.load(his_path + 'r3.pth')
    # # clients[2].train(epoch, all_epoch, name='r3', is_unlearn=True, sense_labels=classes_to_remove, cfg=cfg)
    # # r3 = clients[2].net 

    # clients[1].net = torch.load(his_path + 'r5.pth')
    # clients[1].train(epoch, all_epoch, name='r5', is_unlearn=True, sense_labels=classes_to_remove, cfg=cfg)
    # r5 = clients[1].net 

    # clients[1].net = torch.load(his_path + 'r6.pth')
    # clients[1].train(epoch, all_epoch, name='r6', is_unlearn=True, sense_labels=classes_to_remove, cfg=cfg)
    # r6 = clients[1].net 
    #tmp_path = '/home/wuqiang/lmy/UnlearningDiffusion/param_data/tmp_2024-04-03_05-10-39'
    pdata = []
    for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
        buffers = torch.load(file)
        for buffer in buffers:
            param = []
            for key in buffer.keys():
                if key in train_layer:
                    param.append(buffer[key].data.reshape(-1))
            param = torch.cat(param, 0)
            pdata.append(param)
    batch = torch.stack(pdata)
    mean = torch.mean(batch, dim=0)
    std = torch.std(batch, dim=0)

    #check the memory of p_data
    useage_gb = get_storage_usage(tmp_path)
    print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")
    tmp_path = '/home/wuqiang/lmy/UnlearningDiffusion/param_data/tmp_2024-04-02_10-28-27'
    state_dic = {
        'pdata': batch.cpu().detach(),
        'mean': mean.cpu(),
        'std': std.cpu(),
        'model': torch.load(os.path.join(tmp_path, "r1.pth")),
        'train_layer': train_layer,
        'performance': save_model_accs,
        'cfg': config_to_dict(cfg)
    }

    torch.save(state_dic, os.path.join(final_path, "data.pt"))
    json_state = {
        'cfg': config_to_dict(cfg),
        'performance': save_model_accs

    }
    json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

   

    print("data process over")
    global clis
    clis = clients



from system.ae_ddpm import AE_DDPM
import hydra
from omegaconf import DictConfig,OmegaConf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import *
from system.parameters import PData

@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def get_config(cfg: DictConfig):
    global config
    config = cfg


# 定义Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_auto_encoder(batch):
    
    input_dim = batch.shape[1]  # 输入维度
    encoding_dim = 2048  # 编码维度
    model = Autoencoder(input_dim, encoding_dim).to('cuda:7')

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练Autoencoder模型
    num_epochs = 50
    for epoch in range(num_epochs):
        batch = batch.to('cuda:7')
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        print(loss)
    # 提取编码器输出作为降维后的表示
    encoded_data = model.encoder(batch)
    torch.save(model, '/home/wuqiang/lmy/UnlearningDiffusion/param_data/cifar100/ae_m.pth')
    return encoded_data.to('cuda:7'), model

def training_for_diffusion_data(config, clients):

    model = AE_DDPM(config)
    
    cfg = config.task
    batch_size = getattr(cfg, 'batch_size', 320)
    num_workers = getattr(cfg, 'num_workers', 4)

    best_acc = 0

    
    num_epochs = 500

    # 创建数据加载器
    eval_loader_sense = clients[1].test_loader_sense
    train_loader = clients[1].train_loader
    eval_loader = clients[1].testloader
    train_param = PData(config.task.param)
    running_loss = 0.0
    encoder_batch, ae_m = train_auto_encoder(train_param.train_dataset.data)
    encoder_batch = encoder_batch.detach().cpu().to('cuda:7')
    valid_batch = encoder_batch.clone()
    for epoch in range(num_epochs):
        running_loss += model.training_step(encoder_batch, 0, epoch)['loss'].data
        
        #model.validation_step(valid_batch, ae_m, 0, epoch, train_param, eval_loader, eval_loader_sense)
    torch.save(model, '/home/wuqiang/lmy/UnlearningDiffusion/param_data/cifar100/ae_ddpm.pth')

    

def calculate_parameter_changes(model1, model2):
    # 计算两个模型中每个层参数的变化
    layer_changes = []
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if param1.requires_grad and param2.requires_grad:  # 只考虑需要梯度更新的参数
            change = torch.norm(param1 - param2)  # 使用欧氏距离作为参数变化的度量
            layer_changes.append((name1, change))
    
    # 根据参数变化的大小对层进行排序
    layer_changes.sort(key=lambda x: x[1], reverse=True)
    
    return layer_changes
    

import torch
import torch.nn as nn



if __name__ == "__main__":

        
    # 定义训练数据集
    # 假设训练数据集为一个大小为 [N, input_dim] 的张量

    # 实例化Autoencoder模型
    training_for_data()
    get_config()
    

    # 获取unlearning之前的参数，复制10份
    his_path = '/home/wuqiang/lmy/UnlearningDiffusion/param_data/'
    
    #layer_changes  = calculate_parameter_changes(torch.load(his_path + 'tmp_2024-04-02_10-28-27/r1.pth'), torch.load(his_path + 'tmp_2024-04-03_04-16-04/unlearn_r1.pth'))
    #model = hydra.utils.instantiate(config.task.model).to('cuda:7') 
    clinet_index = 1
    models = []
    def get_param(path):
        model = torch.load(path)
        models.append(model)
        model.eval()
        acc_sense = clis[clinet_index].test(cfg,model, clis[clinet_index].criterion, clis[clinet_index].test_loader_sense)
        print('original model sense acc:', acc_sense)
        before_unlearn_params = state_part(clis[clinet_index].train_layer, model)
        param = []
        for key in before_unlearn_params.keys():
            if key in clis[clinet_index].train_layer:
                param.append(before_unlearn_params[key].data.reshape(-1))
        param = torch.cat(param, 0).view(1,-1)
        return param
    param_batch = []
   #param_batch.append(get_param(his_path + 'tmp_2024-04-02_10-28-27/r1.pth'))
    #param_batch.append(get_param(his_path + 'tmp_2024-04-02_10-28-27/r2.pth'))
    #param_batch.append(get_param(his_path + 'tmp_2024-04-02_10-28-27/r5.pth'))
    param_batch.append(get_param(his_path + 'tmp_2024-04-02_10-28-27/r7.pth'))
    #param_batch = torch.cat(param_batch, 0) 
    training_for_diffusion_data(config, clis)

    #原来模型对敏感数据标签的准确率
    ae_m = torch.load(his_path + 'cifar100/ae_m.pth').to('cuda:7')

    diff_model = torch.load(his_path + 'cifar100/ae_ddpm.pth').to('cuda:7')
    # param_dic = state_part(clis[clinet_index].train_layer, model)
    # param = []
    # for key in param_dic.keys():
    #     param.append(param_dic[key].data.reshape(-1))
    #batch = torch.cat(param, 0)
    start_time = time.time()
    best_params = []
    for i in range(len(models)):
        batch_p = param_batch[i]
        model = models[i]
        best_acc_and_senseacc_dic = []
        
        batch_p= batch_p.to('cuda:7')
        torch.no_grad()
        batch_p = ae_m.encoder(batch_p)
        batch = diff_model.pre_process(batch_p)
        while len(best_acc_and_senseacc_dic) == 0:
            outputs = diff_model.generate(batch, 10)
            params = diff_model.post_process(outputs)
            params = ae_m.decoder(params)
            #params = batch_p
            params = params.cpu()
            accs = []
            accs_sense = []
            for i in range(params.shape[0]):
                param = params[i].to(batch_p.device)
                acc, test_loss, output_list = test_g_model(param, model, clis[clinet_index].train_layer, clis[clinet_index].testloader)
                acc_sense, _, _ = test_g_model(param, model, clis[clinet_index].train_layer, clis[clinet_index].test_loader_sense)
                accs.append(acc)
                accs_sense.append(acc_sense)
                if acc > 70 and acc_sense < 10: 
                    print('acc {0} and acc sense {1}'.format(acc, acc_sense))
                    best_acc_and_senseacc_dic.append(param.cpu())
            

            
            best_acc = np.max(accs)
            best_acc_sense = np.min(accs_sense)
            print("generated models accuracy:", accs)
            print("generated models mean accuracy:", np.mean(accs))
            print("generated models best accuracy:", best_acc)
            print("generated models sense accuracy:", accs_sense)
            print("generated models best sense accuracy:", best_acc_sense)
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.init()
            # 切换到 CUDA 设备0
            torch.cuda.device(i)
        best_params.append(best_acc_and_senseacc_dic)
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码执行时间：", execution_time)
    print(best_params)