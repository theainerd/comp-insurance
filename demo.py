import argparse, time, os
from utils_data import *
from utils_algo import *
from models import *

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
	prog='complementary-label learning demo file.',
	usage='Demo with complementary labels.',
	description='A simple demo file with MNIST dataset.',
	epilog='end',
	add_help=True)

parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=5e-5, type=float)
parser.add_argument('-bs', '--batch_size', help='batch_size of ordinary labels.', default=63, type=int)
# parser.add_argument('-me', '--method', help='method type. ga: gradient ascent. nn: non-negative. free: Theorem 1. pc: Ishida2017. forward: Yu2018.', choices=['ga', 'nn', 'free', 'pc', 'forward'], type=str, required=True)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=200)
parser.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-4, type=float)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def train_model(model, chosen_loss_c, optimizer, scheduler,K,ccp,num_epochs,meta_method):
    accuracy_stats = {
    'train': [],
    "val": []
        }
    loss_stats = {
        'train': [],
        "val": []
    }

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        
        model.train()
        for X_train_batch, y_train_batch in complementary_train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            
            loss, loss_vector = chosen_loss_c(f=y_train_pred, K=K, labels=y_train_batch, ccp=ccp, meta_method=meta_method)

            # backward + optimize only if in training phase
            if meta_method == 'ga':
                if torch.min(loss_vector).item() < 0:
                    loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)), 1)
                    min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                    loss = torch.sum(min_loss_vector)
                    loss.backward()
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            p.grad = -1*p.grad
                else:
                    loss.backward()

            train_acc = multi_acc(y_train_pred, y_train_batch)
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            train_epoch_acc += train_acc.item()
            
            
        # VALIDATION    
        with torch.no_grad():
            
            val_epoch_loss = 0
            val_epoch_acc = 0
            
            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                
                y_val_pred = model(X_val_batch)
                            
                val_loss, val_loss_vector = chosen_loss_c(f=y_val_pred, K=K, labels=y_val_batch, ccp=ccp, meta_method=meta_method)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.2f} | Val Loss: {val_epoch_loss/len(val_loader):.2f} | Train Acc: {train_epoch_acc/len(train_loader):.2f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
    return model,accuracy_stats['val']

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

full_train_loader, train_loader,val_loader,ordinary_train_dataset,num_feat = prepare_insurance_data(batch_size=args.batch_size)
ordinary_train_loader, complementary_train_loader, ccp = prepare_train_loaders(full_train_loader=full_train_loader, batch_size=args.batch_size, ordinary_train_dataset=ordinary_train_dataset)

# meta_method = 'free' if args.method =='ga' else args.method

K = 8
model = MulticlassClassification(num_feature = num_feat, num_class=K)
model.to(device)


# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_forward,val_acc_forward = train_model(model, chosen_loss_c, optimizer_conv, exp_lr_scheduler,
                       K,ccp,num_epochs=args.epochs,meta_method = 'forward')

model_free,val_acc_free = train_model(model, chosen_loss_c, optimizer_conv, exp_lr_scheduler,
                       K,ccp,num_epochs=args.epochs,meta_method = 'free')

model_ga,val_acc_ga = train_model(model, chosen_loss_c, optimizer_conv, exp_lr_scheduler,
                       K,ccp,num_epochs=args.epochs,meta_method='ga')

model_nn,val_acc_nn = train_model(model, chosen_loss_c, optimizer_conv, exp_lr_scheduler,
                       K,ccp,num_epochs=args.epochs,meta_method = 'nn')

# Data
df=pd.DataFrame({'epoch': range(0,args.epochs), 'y1_values': val_acc_forward, 'y2_values': val_acc_free, 'y3_values': val_acc_ga,'y4_values': val_acc_nn })
 
# multiple line plots
plt.plot( 'epoch', 'y1_values', data=df, marker='', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label = 'forward')
plt.plot( 'epoch', 'y2_values', data=df, marker='', color='red', linewidth=2,label = 'free')
plt.plot( 'epoch', 'y3_values', data=df, marker='', color='green', linewidth=2, linestyle='dashed', label="pc")
plt.plot( 'epoch', 'y4_values', data=df, marker='', color='blue', linewidth=2, linestyle='dashed', label="nn")
# show legend
plt.legend()

# show graph
plt.show()
plt.savefig("output.png")