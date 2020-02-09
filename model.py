import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, MaxPool2d, Sigmoid, ReLU
from torch.autograd import Variable

import torch.nn.functional as F
import torch.nn as nn
from layers import ALinear, AConv2d

import numpy as np

import math

class Classifier(nn.Module):
    def __init__(self, layer_size=64, output_shape=55, num_channels=1, keep_prob=1.0, image_size=28, tasks = 1, bn_boole=False):
        super(Classifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.conv1 = AConv2d(num_channels, layer_size, 3, 1, 1, datasets=tasks)
        self.conv2 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        self.conv3 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        self.conv4 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(layer_size) for j in range(tasks)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(layer_size) for j in range(tasks)])
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(layer_size) for j in range(tasks)])
        self.bn4 = nn.ModuleList([nn.BatchNorm2d(layer_size) for j in range(tasks)])
        
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        self.do = nn.Dropout(keep_prob)
        self.relu = nn.ReLU()
        self.sm = nn.Sigmoid()
        
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

        self.linear = ALinear(self.outSize, output_shape, datasets=tasks, multi=True)   
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, AConv2d):
                m.weight.data.normal_(0, 1e-2)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)
            elif isinstance(m, ALinear):
                m.weight.data.normal_(0, 2.0 * 1e-1)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)
        
        # self.linear = ALinear(self.outsize,1)

    def forward(self, image_input, task = 0, round_ = False):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        # x = self.mp1(self.bn1(self.relu(self.conv1(image_input, dataset=task, round_ = round_))))
        # x = self.mp2(self.bn2(self.relu(self.conv2(x, dataset=task, round_ = round_))))
        # x = self.mp3(self.bn3(self.relu(self.conv3(x, dataset=task, round_ = round_))))
        # x = self.mp4(self.bn4(self.relu(self.conv4(x, dataset=task, round_ = round_))))

        x = self.mp1(self.relu(self.bn1[task]((self.conv1(image_input, dataset=task, round_ = round_)))))
        x = self.mp2(self.relu(self.bn2[task](self.conv2(x, dataset=task, round_ = round_))))
        x = self.mp3(self.relu(self.bn3[task](self.conv3(x, dataset=task, round_ = round_))))
        x = self.mp4(self.relu(self.bn4[task](self.conv4(x, dataset=task, round_ = round_))))

        x = x.view(x.size()[0], -1)
        x = self.linear(x, dataset=task, round_ = round_)
        # x = self.sm(x)
        return x

class ClassifierMLP(nn.Module):
    def __init__(self, layer_size=64, output_shape=55, input_size=784, num_channels=1, keep_prob=1.0, image_size=28, tasks = 1, bn_boole=False):
        super(ClassifierMLP, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.conv1 = ALinear(num_channels*input_size, layer_size, datasets=tasks)
        self.conv2 = ALinear(layer_size, layer_size, datasets=tasks)
        self.conv3 = ALinear(layer_size, layer_size, datasets=tasks)
        self.conv4 = ALinear(layer_size, layer_size, datasets=tasks)
        
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(layer_size) for j in range(tasks)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(layer_size) for j in range(tasks)])
        self.bn3 = nn.ModuleList([nn.BatchNorm1d(layer_size) for j in range(tasks)])
        self.bn4 = nn.ModuleList([nn.BatchNorm1d(layer_size) for j in range(tasks)])
                        
        self.do = nn.Dropout(keep_prob)
        self.relu = nn.ReLU()
        self.sm = nn.Sigmoid()
        
        self.outSize = layer_size

        self.linear = ALinear(layer_size, output_shape, datasets=tasks, multi=True)   
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, AConv2d):
                m.weight.data.normal_(0, 1e-2)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)
            elif isinstance(m, ALinear):
                m.weight.data.normal_(0, 2.0 * 1e-1)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)
        
        # self.linear = ALinear(self.outsize,1)

    def forward(self, image_input, task = 0, round_ = False):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        # x = self.mp1(self.bn1(self.relu(self.conv1(image_input, dataset=task, round_ = round_))))
        # x = self.mp2(self.bn2(self.relu(self.conv2(x, dataset=task, round_ = round_))))
        # x = self.mp3(self.bn3(self.relu(self.conv3(x, dataset=task, round_ = round_))))
        # x = self.mp4(self.bn4(self.relu(self.conv4(x, dataset=task, round_ = round_))))

        x = self.relu(self.bn1[task](self.conv1(image_input, dataset=task, round_ = round_)))
        x = self.relu(self.bn2[task](self.conv2(x, dataset=task, round_ = round_)))
        x = self.relu(self.bn3[task](self.conv3(x, dataset=task, round_ = round_)))
        x = self.relu(self.bn4[task](self.conv4(x, dataset=task, round_ = round_)))

        x = x.view(x.size()[0], -1)
        x = self.linear(x, dataset=task, round_ = round_)
        # x = self.sm(x)
        return x
    
    
def _prune(module, task, prune_para):
    if any([isinstance(module, ALinear), isinstance(module, AConv2d)]):
        if not module.multi:
            mask = (module.soft_round(module.adjx[task]) > prune_para).data
            print("Params alive:",mask.sum().float()/np.prod(mask.shape))
            l = module.adjx[task]*mask.float()
            module.adjx[task].data.copy_(l.data)
            # A = module.soft_round(module.adjx[task]).byte().float() * module.adjx[task]
            # module.adjx[task].data.copy_(A.data)
            # print("Params alive:",module.soft_round(module.adjx[task]).byte().sum().float()/np.prod(module.adjx[task].shape))
        
    if hasattr(module, 'children'):
        for submodule in module.children():
            _prune(submodule, task, prune_para)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
            
def _adj_ind_loss(module, task, S=0):
    if task==0:
        return 0
    if any([isinstance(module, ALinear), isinstance(module, AConv2d), module.__class__.__name__=="AConv2d"]):
        if not module.multi:
            mask = (module.soft_round(module.adjx[task-1]) > 0.85).data
            if (mask.sum().float()/np.prod(mask.shape))>0.13:
                A = torch.stack(list(module.adjx[:task])).view(task,-1)
                # A = torch.stack([module.soft_round(m) for m in list(module.adjx[:task])]).view(task,-1)
                # if torch.cuda.is_available():
                #     A = A.cuda()
                # S = pairwise_distances(A).mean()/2
                S += pairwise_distances(A).mean()/(2*(A.shape[1]))
            return S
        
    if hasattr(module, 'children'):
        n = 0
        for submodule in module.children():
            S += _adj_ind_loss(submodule, task=task, S=S)
            n+=1
        if n > 0:
            S /= n
            
        # S += torch.sum(torch.Tensor([_adj_ind_loss(submodule, S) for submodule in module.children()]))
        return S
            

def _prune_freeze(module, task, prune_para):
    if any([isinstance(module, ALinear), isinstance(module, AConv2d)]):
        if not module.multi:
            mask = (module.soft_round(module.adjx[task]) <= prune_para).data
            print("Params to prune:",mask.sum().float()/np.prod(mask.shape))
            for k in range(len(module.adjx)):
                if k==task:
                    continue
                l = module.adjx[k]*mask.float()
                module.adjx[k].data.copy_(l.data)
    if hasattr(module, 'children'):
        for submodule in module.children():
            _prune_freeze(submodule, task, prune_para)

def _turn_off_adj(module, task):
    if any([isinstance(module, ALinear), isinstance(module, AConv2d)]):
        if not module.multi:
            module.adjx[task].requires_grad=False
    if hasattr(module, 'children'):
        for submodule in module.children():
            _turn_off_adj(submodule, task)

def _turn_off_weights(module):
    if any([isinstance(module, ALinear), isinstance(module, AConv2d)]):
        if not module.multi:
            module.weight.requires_grad=False
    if hasattr(module, 'children'):
        for submodule in module.children():
            _turn_off_weights(submodule)


def _turn_off_multi_weights(module, task):
    if any([isinstance(module, ALinear), isinstance(module, AConv2d)]):
        if module.multi:
            module.weight[task].requires_grad=False
    if hasattr(module, 'children'):
        for submodule in module.children():
            _turn_off_weights(submodule)

    
def _adj_spars_loss(module, task, S=0, tol = 0.13, prune_para = 0.95):
    if any([isinstance(module, ALinear), isinstance(module, AConv2d), module.__class__.__name__=="AConv2d"]):
        if not module.multi:
            mask = (module.soft_round(module.adjx[task]) > prune_para).data
            if (mask.sum().float()/np.prod(mask.shape)) > tol:
                S += (module.adjx[task].norm(p=1)/module.adjx[task].view(-1).shape[0])
            return S
        
    if hasattr(module, 'children'):
        n = 0
        for submodule in module.children():
            S += _adj_spars_loss(submodule, task=task, S=S, tol = tol, prune_para = prune_para)
            n+=1
        if n > 0:
            S /= n
            
        # S += torch.sum(torch.Tensor([_adj_ind_loss(submodule, S) for submodule in module.children()]))
        return S

def _freeze_grads(module, task, hooks = []):
    if task == 0:
        return []
    if any([isinstance(module, ALinear), isinstance(module, AConv2d)]):
        if not module.multi and module.weight.requires_grad:
            gradient_mask = (module.soft_round(module.adjx[0]*module.weight).round().float()<=1e-6).data
            # gradient_mask = (module.adjx[0]==0.).data
            for k in range(1, task):
                # gradient_mask = gradient_mask * (module.adjx[k]==0.).data
                gradient_mask = gradient_mask * (module.soft_round(module.adjx[k]*module.weight).round().float()<=1e-6).data
            gradient_mask = gradient_mask.float()
            h = module.weight.register_hook(lambda grad: grad.mul_(gradient_mask))
            return hooks + [h]                
    if hasattr(module, 'children'):
        for submodule in module.children():
            hooks = hooks + _freeze_grads(submodule, task, hooks)
        return hooks
    
        

    
# def prune(self, p_para=0.5, task=None):
#     for module in list(self.children()):
#         if hasattr(module,'l1_loss'):
#             mask = (module.soft_round(module.adjx[task]) > p_para).data
#             l = module.adjx[task]*mask.float()
#             module.adjx[task].data.copy_(l.data)











