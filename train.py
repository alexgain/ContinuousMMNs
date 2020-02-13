import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomApply, Resize
import torchvision.transforms.functional as F

from model import Classifier, ClassifierMLP, _prune, _prune_freeze, _adj_ind_loss, _turn_off_adj, _turn_off_weights, _turn_off_multi_weights, _adj_spars_loss, _freeze_grads

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import imageio
import glob
import os

import time

from anode.conv_models import ConvODENet, AConvODENet
from anode.models import ODENet, AODENet
from anode.training import Trainer


gpu_boole = torch.cuda.is_available()

import argparse

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--tasks', default=10, type=int, help='no. of tasks')
parser.add_argument('--hidden_size', default=64, type=int, help='hidden neurons')
parser.add_argument('--im_size', default=28, type=int, help='image dimensions')
parser.add_argument('--save_path', default='./saved_models/default.pt', type=str, help='save path')
parser.add_argument('--load_path', default='', type=str, help='load path')
args = parser.parse_args()


train_data_aug = Compose([
    Resize(size = args.im_size),
    RandomApply(
        [RandomAffine(degrees=(-10, 10), scale=(0.8, 1.2), translate=(0.05, 0.05))],
        p=0.5
    ),
    ToTensor(),
])

train_data_aug_simple = Compose([
    Resize(size = args.im_size),
    ToTensor()
])


test_data_aug = Compose([
    Resize(size = args.im_size),
    ToTensor()
])

training = torchvision.datasets.MNIST(root ='./data', transform = train_data_aug_simple, train=True, download=True)
testing =  torchvision.datasets.MNIST(root ='./data', transform = test_data_aug, train=False, download=True)
train_loader = torch.utils.data.DataLoader(dataset=training, batch_size = args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testing, batch_size = args.batch_size, shuffle=False)

permutations = [torch.Tensor(np.random.permutation(784).astype(np.float64)).long() for _ in range(args.tasks)]
torch.save(torch.stack(permutations),args.save_path[:len(args.save_path)-2]+'permutations.pt')

## model and optimizer instantiations:
# net = ClassifierMLP(image_size = args.im_size, output_shape=10, tasks=args.tasks, layer_size=args.hidden_size, bn_boole=True)
if args.load_path != '':
    net = torch.load(args.load_path)
else:
    # net = ConvODENet(img_size=(1, 28, 28), num_filters=32, augment_dim=1, output_dim=10)
    net = AConvODENet(img_size=(1, 28, 28), num_filters=32, augment_dim=1, output_dim=10)
if gpu_boole:
    net = net.cuda()


optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)

## train, test eval:
loss_metric = torch.nn.CrossEntropyLoss()

def dataset_eval(data_loader, verbose = 1, task = 0, round_=False):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in data_loader:
        try:
            if gpu_boole:
                images, labels = images.cuda(), labels.cuda()
            
            images = images.view(-1,28*28)[:,permutations[task]]
            images = images.view(-1,1,28,28)
            labels = labels.view(-1).cpu()
            outputs = net(images, task = task, round_=round_).cpu()            
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted.float() == labels.float()).sum().cpu().data.numpy().item()
    
            loss_sum += loss_metric(outputs,labels).cpu().data.numpy().item()
            
            del images; del labels; del outputs; del _; del predicted;
        except Exception as e:
            print(e)

    
    correct = np.float(correct)
    total = np.float(total)
    if verbose:
        print('Accuracy:',(100 * np.float(correct) / np.float(total)))
        print('Loss:', (loss_sum / np.float(total)))

    acc = 100.0 * (np.float(correct) / np.float(total))
    loss = (loss_sum / np.float(total))
    del total; del correct; del loss_sum
    return acc, loss
    

## Task Loop:
for j in range(args.tasks):
                    
    for epoch in range(args.epochs):
                
        t1 = time.time()
        
        print("Task:",j,"- Epoch:",epoch)

        for i, (x,y) in enumerate(train_loader):
            try:
            
                if gpu_boole:
                    x, y = x.cuda(), y.cuda()                
                    
                x = x.view(-1,28*28)[:,permutations[j]]
                x = x.view(-1,1,28,28)                
                y = y.view(-1)
                
                optimizer.zero_grad()
                
                outputs = net(x,task=j)
                            
                loss = loss_metric(outputs,y)
                
                loss.backward()
                optimizer.step()
                
                del loss; del x; del y; del outputs;
            except Exception as e:
                print(e)

        
        train_acc, train_loss = dataset_eval(train_loader, verbose = 0, task = j)
        test_acc, test_loss= dataset_eval(test_loader, verbose = 0, task = j)
        test_acc_true, test_loss_true= dataset_eval(test_loader, verbose = 0, task = j, round_=True)
        print("Train acc, Train loss", train_acc, train_loss)
        print("Test acc, Test loss", test_acc, test_loss)
        print("Test acc, Test loss (Rounded Adj)", test_acc_true, test_loss_true)
        t2 = time.time()
        print('Time left for task:',((t2-t1)/60)*(args.epochs-epoch),'minutes')
        print()
                    
    
    print("--------------------------------")
    print("Test acc for all tasks:")
    total_test_acc = 0
    for j2 in range(j+1):
        print("Task:",j2)
        test_acc, test_loss = dataset_eval(test_loader, verbose = 0, task = j2)
        print("Test acc, Test loss:",test_acc, test_loss)

        test_acc, test_loss = dataset_eval(test_loader, verbose = 0, task = j2, round_=True)
        print("Test acc, Test loss: (Rounded Adj)",test_acc, test_loss)
        
        total_test_acc += test_acc
    
    total_test_acc /= j+1
    print("Total test acc:",total_test_acc)
    print("--------------------------------")
    print()
    print("Saving model...")
    torch.save(net, args.save_path)


