import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .utils import *
from .format import *
import os 
import copy
import hydra
import torch.nn.init as init
from PIL import Image

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


class Client:
    def __init__(self, net, criterion, optimizer, scheduler, train_loader, testloader, test_loader_sense, train_layer, tmp_path, save_model_accs):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.testloader = testloader
        self.test_loader_sense = test_loader_sense
        self.train_layer = train_layer
        self.tmp_path = tmp_path
        self.save_model_accs = save_model_accs
        

    def train(self, epoch, all_epoch=100, aggre_wei=None, name=None, is_unlearn=False, sense_labels=None, cfg=None):
        if aggre_wei != None:
            aggre = average_weights(aggre_wei)
            self.net.load_state_dict(aggre)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, self.net.parameters())
        self.scheduler = hydra.utils.instantiate(cfg.lr_scheduler,self.optimizer)
        if is_unlearn:
            self.optimizer = hydra.utils.instantiate(cfg.optimizer, self.net.parameters())
            self.scheduler = hydra.utils.instantiate(cfg.lr_scheduler,self.optimizer)
            train_loader,testloader = remove_sense_get_loader(self.train_loader.dataset, self.testloader.dataset, sense_labels, 32, 4)
            fix_partial_model(self.train_layer, self.net)
           
            # # # 初始化指定层的模型参数
            # for name, param in self.net.named_parameters():
            #     if name in self.train_layer:
            #         # 初始化卷积层的权重为正态分布，均值为 0，标准差为 0.01
            #         if 'weight' in name:
            #             init.normal_(param, mean=0, std=0.01)
            #         # 初始化卷积层的偏置为 0
            #         elif 'bias' in name:
            #             init.constant_(param, 0)
        else:
            train_loader = self.train_loader
            testloader = self.testloader
        best_acc = 0
        for i in range(0, all_epoch):
            print('\nEpoch: %d' % i)
        
            self.net.train()
            train_loss = 0
            correct = 0
            total = 0
            j = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                max_length = min(max([len(text) for text in inputs]) + 2, 128)
              
                if cfg.model['_target_'] == 'models.bert.Bert':
                    batch_input_ids, batch_att_mask = [], []
                    for text in inputs:
                        encoding = self.net.get_tokenizer().encode_plus(
                            text,
                            add_special_tokens=True,  # Add [CLS] and [SEP]
                            padding='max_length',  # Pad to max_length
                            truncation=True,  # Truncate longer sequences
                            max_length=max_length,
                            return_tensors='pt'  # Return PyTorch tensors
                        )
                        batch_input_ids.append(encoding['input_ids'])
                        batch_att_mask.append(encoding['attention_mask'])
                    inputs = torch.cat(batch_input_ids)
                    att_mask = torch.cat(batch_att_mask)

                inputs, targets = inputs, targets
                if cfg.data.dataset != 'YAHOO':
                    inputs = inputs.to(torch.float32)  
                
                    
                self.optimizer.zero_grad()
                if cfg.model['_target_'] == 'models.bert.Bert':
                    outputs = self.net(inputs, att_mask)
                else:
                    outputs = self.net(inputs)
            
                if is_unlearn:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)
            
                loss_start = time.time()    
                loss.backward()

                self.optimizer.step()
                loss_end = time.time()  
                
                j+=1
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print("loss time", (loss_end-loss_start))
                
            acc = self.test(cfg,self.net, self.criterion, testloader)
            acc_sense = self.test(cfg,self.net, self.criterion, self.test_loader_sense)
            
            best_acc = max(acc, best_acc)
            # if is_unlearn:
            #     if i == (epoch - 1):
            #         print("saving the model")
            #         torch.save(self.net, os.path.join(self.tmp_path, name +"_unlearn.pth"))
            #         fix_partial_model(self.train_layer, self.net)
            #         parameters = []
            #     # 大于epoch次数保存模型参数
            #     if i >= epoch:
            #         parameters.append(state_part(self.train_layer, self.net))
            #         self.save_model_accs.append(acc)
            #         if len(parameters) == 10 or i == all_epoch - 1:
            #             torch.save(parameters, os.path.join(self.tmp_path, "p_data_{}.pt".format(str(i)+"_"+name)))
            #             parameters = []

            self.scheduler.step()
        if is_unlearn is False:
            torch.save(self.net, os.path.join(self.tmp_path, name + ".pth"))

    def train_transfer(self, epoch, all_epoch=100, is_transfer=True, aggre_wei=None, name=None, is_unlearn=False, sense_labels=None, cfg=None):
        if aggre_wei != None:
            aggre = average_weights(aggre_wei)
            self.net.load_state_dict(aggre)
        
        if is_transfer:
            fix_partial_model(self.train_layer, self.net)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, self.net.parameters())
        self.scheduler = hydra.utils.instantiate(cfg.lr_scheduler,self.optimizer)
        if is_unlearn:
            self.optimizer = hydra.utils.instantiate(cfg.optimizer, self.net.parameters())
            self.scheduler = hydra.utils.instantiate(cfg.lr_scheduler,self.optimizer)
            train_loader,test_loader = remove_sense_get_loader(self.train_loader.dataset, self.testloader.dataset, sense_labels, 32, 4)
          
            #fix_partial_model(self.train_layer, self.net)
        else:
            train_loader = self.train_loader
            test_loader = self.testloader
        best_acc = 0
        for i in range(0, all_epoch):
            print('\nEpoch: %d' % i)
        
            self.net.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs, targets
                if cfg.data.dataset != 'YAHOO':
                    inputs = inputs.to(torch.float32)  
                self.optimizer.zero_grad()
        
              
                outputs = self.net(inputs)
                if is_unlearn:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)
                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            if is_unlearn is not True:      
                progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))            
                acc = self.test(cfg,self.net, self.criterion, test_loader)
                acc_sense = self.test(cfg,self.net, self.criterion, self.test_loader_sense)
           
            #best_acc = max(acc, best_acc)
            # if is_unlearn:
            #     if i == (epoch - 1):
            #         print("saving the model")
            #         torch.save(self.net, os.path.join(self.tmp_path, name +"_unlearn.pth"))
            #         fix_partial_model(self.train_layer, self.net)
            #         parameters = []
            #     # 大于epoch次数保存模型参数
            #     if i >= epoch:
            #         parameters.append(state_part(self.train_layer, self.net))
            #         self.save_model_accs.append(acc)
            #         if len(parameters) == 10 or i == all_epoch - 1:
            #             torch.save(parameters, os.path.join(self.tmp_path, "p_data_{}.pt".format(str(i)+"_"+name)))
            #             parameters = []

            self.scheduler.step()
        acc = self.test(cfg,self.net, self.criterion, test_loader)
        acc_sense = self.test(cfg,self.net, self.criterion, self.test_loader_sense)
        if is_unlearn is False:
            torch.save(self.net, os.path.join(self.tmp_path, name + ".pth"))

    def train_continue(self, epoch, all_epoch=100, aggre_wei=None, name=None, is_unlearn=False, sense_labels=None, cfg=None):
        if aggre_wei != None:
            aggre = average_weights(aggre_wei)
            self.net.load_state_dict(aggre)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, self.net.parameters())
        self.scheduler = hydra.utils.instantiate(cfg.lr_scheduler,self.optimizer)
        if is_unlearn:
            self.optimizer = hydra.utils.instantiate(cfg.optimizer, self.net.parameters())
            train_loader,_ = remove_sense_get_loader(self.train_loader.dataset, self.testloader.dataset, sense_labels, 32, 4)
            fix_partial_model(self.train_layer, self.net)
            for module in self.net.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.track_running_stats = False

            # # # 初始化指定层的模型参数
            # for name, param in self.net.named_parameters():
            #     if name in self.train_layer:
            #         # 初始化卷积层的权重为正态分布，均值为 0，标准差为 0.01
            #         if 'weight' in name:
            #             init.normal_(param, mean=0, std=0.01)
            #         # 初始化卷积层的偏置为 0
            #         elif 'bias' in name:
            #             init.constant_(param, 0)
        else:
            train_loader = self.train_loader
        best_acc = 0
        for i in range(0, all_epoch):
            print('\nEpoch: %d' % i)
        
            self.net.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs, targets
                if cfg.data.dataset != 'YAHOO':
                    inputs = inputs.to(torch.float32)
             
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                if is_unlearn:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)
                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                                    
            acc = self.test(cfg,self.net, self.criterion, self.testloader)
            acc_sense = self.test(cfg,self.net, self.criterion, self.test_loader_sense)
            if is_unlearn:
                if acc > 80 and acc_sense < 5:
                    parameters = []
                    parameters.append(state_part(self.train_layer, self.net))
                    torch.save(parameters, os.path.join(self.tmp_path, "p_data_{}.pt".format(str(i)+"_"+name)))
                    torch.save(self.net, os.path.join(self.tmp_path, "unlearn_r1.pth"))
                # elif acc < 70:
                #     break
            best_acc = max(acc, best_acc)
            # if is_unlearn:
            #     if i == (epoch - 1):
            #         print("saving the model")
            #         torch.save(self.net, os.path.join(self.tmp_path, name +"_unlearn.pth"))
            #         fix_partial_model(self.train_layer, self.net)
            #         parameters = []
            #     # 大于epoch次数保存模型参数
            #     if i >= epoch:
            #         parameters.append(state_part(self.train_layer, self.net))
            #         self.save_model_accs.append(acc)
            #         if len(parameters) == 10 or i == all_epoch - 1:
            #             torch.save(parameters, os.path.join(self.tmp_path, "p_data_{}.pt".format(str(i)+"_"+name)))
            #             parameters = []

            self.scheduler.step()
        if is_unlearn is False:
            torch.save(self.net, os.path.join(self.tmp_path, name + ".pth"))

    def test(self, cfg, net, criterion, testloader):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                max_length = min(max([len(text) for text in inputs]) + 2, 128)
                if cfg.model['_target_'] == 'models.bert.Bert':
                    batch_input_ids, batch_att_mask = [], []
                    for text in inputs:
                        encoding = net.get_tokenizer().encode_plus(
                            text,
                            add_special_tokens=True,  # Add [CLS] and [SEP]
                            padding='max_length',  # Pad to max_length
                            truncation=True,  # Truncate longer sequences
                            max_length=max_length,
                            return_tensors='pt'  # Return PyTorch tensors
                        )
                        batch_input_ids.append(encoding['input_ids'])
                        batch_att_mask.append(encoding['attention_mask'])
                    inputs = torch.cat(batch_input_ids)
                    att_mask = torch.cat(batch_att_mask)
                    
                inputs, targets = inputs, targets
                if cfg.data.dataset != 'YAHOO':
                    inputs = inputs.to(torch.float32)
                    
                if cfg.model['_target_'] == 'models.bert.Bert':
                    outputs = net(inputs, att_mask)
                else:
                    outputs = net(inputs)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            return 100. * correct / total

def average_weights(weights):
    weights_avg = copy.deepcopy(weights[0].state_dict())

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i].state_dict()[key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg