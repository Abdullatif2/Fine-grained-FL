#!/usr/bin/env python
# coding: utf-8

#

import sys
import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data.dataset import Dataset
# from torchvision import transforms 
# from torchvision.transforms import Compose 
from data import *
from models import *
from Sys_func import *
from param_setting import *
import csv
import gc
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
torch.backends.cudnn.benchmark=True
np.random.seed(10000)
##### Hyperparameters for federated learning #########
torch.cuda.empty_cache() 
classes_pc = 1
num_clients = 10
num_selected =10
num_rounds = 200
num_anntena = 8  #2,4,8
epochs = 20
batch_size = 20
baseline_num = 100
retrain_epochs = 20
deadline = 20

def baseline_data(num):
  '''
  Returns baseline data loader to be used on retraining on global server
  Input:
        num : size of baseline data
  Output:
        loader: baseline data loader
  '''
  xtrain, ytrain, xtmp,ytmp = get_cifar10()
  x , y = shuffle_list_data(xtrain, ytrain)

  x, y = x[:num], y[:num]
  transform, _ = get_default_data_transforms(train=True, verbose=False)
  loader = torch.utils.data.DataLoader(CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)

  return loader

###################################################################################

def global_model_sync(client_model, global_model):
  '''
  This function synchronizes the client model with global model
  '''
  client_model.load_state_dict(global_model.state_dict())
  
###################################################################################


def server_aggregate(global_model, client_models,client_lens):
    """
    This function has aggregation method 'wmean'
    wmean takes the weighted mean of the weights of models
    """
    total = sum(client_lens)
    n = len(client_models)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()*(n*client_lens[i]/total) for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
        

###################################################################################

def client_update(client_model, optimizer, train_loader, epoch=5,Threshold_prop=0.9):
    """
    This function updates/trains client model on client data
    """
    # run for one epoch
    model.train()
    if Threshold_prop>=1.0:
        num_samples = (len(train_loader)*batch_size)
        for e in range(epoch):
        # if e==1: 

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = client_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
    else:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
        newsamples = predict_samples(client_model,train_loader,Threshold_prop)
        num_samples = (len(newsamples)*batch_size)
        # print('Len new :',len(newsamples),'len before:',len(train_loader))
        model.train()
        for e in range(epoch):
            # if e==1: 
    
            for batch_idx, (data, target) in enumerate(newsamples):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = client_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
    return loss.item(),num_samples
###################################################################################

def predict_samples(client_model,trainloaderfull,Threshold_prop=0.9):
    model.eval()
    X_new=[]
    Y_new=[]

    with torch.no_grad():
        for x, y in trainloaderfull:
            x, y = x.cuda(), y.cuda()

            output = client_model(x)
            prob = F.softmax(output, dim=1)
            top_p, top_class = prob.topk(1, dim = 1)
            for i in range(len(x)):
                if top_p[i]<Threshold_prop:
                    X_new.append(x[i])
                    Y_new.append(y[i])
                    
    Y_new = torch.Tensor(Y_new).type(torch.int64)
    predicted_samples = [(x, y) for x, y in zip(X_new, Y_new)]
    # print('predict_sample',len(predict_sample))
    newdata_loader = torch.utils.data.DataLoader(predicted_samples,batch_size)
    return newdata_loader
###################################################################################
def test(global_model, test_loader):
    """
    This function test the global model on test 
    data and returns test loss and test accuracy 
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc
##############################################################Save Results ###########################################
def saveresults(test_accu,test_loss,E_total_upload,E_total_cmp,num_samples,num_anntenas):
     with open('./results/noniid_testaccu_Dataset_cifar_Round_'+str(num_rounds)+'_deadline_'+str(deadline)+'num_users_'+str(num_selected)+'total_clients_'+str(num_clients)+'_num_anntena_'+str(num_anntenas)+'.csv', 'a+', newline='') as resultsfile:
        wr = csv.writer(resultsfile)
        wr.writerow(test_accu)
     with open('./results/noniid_testloss_Dataset_cifar_Round_'+str(num_rounds)+'_deadline_'+str(deadline)+'num_users_'+str(num_selected)+'total_clients_'+str(num_clients)+'_num_anntena_'+str(num_anntenas)+'.csv', 'a+', newline='') as resultsfile:
        wr = csv.writer(resultsfile)
        wr.writerow(test_loss)
     with open('./results/noniid_E_total_upload_Dataset_cifar_Round_'+str(num_rounds)+'_deadline_'+str(deadline)+'num_users_'+str(num_selected)+'total_clients_'+str(num_clients)+'_num_anntena_'+str(num_anntenas)+'.csv', 'a+', newline='') as resultsfile:
        wr = csv.writer(resultsfile)
        wr.writerow(E_total_upload)
     with open('./results/noniid_E_total_cmp_Dataset_cifar_Round_'+str(num_rounds)+'_deadline_'+str(deadline)+'num_users_'+str(num_selected)+'total_clients_'+str(num_clients)+'_num_anntena_'+str(num_anntenas)+'.csv', 'a+', newline='') as resultsfile:
        wr = csv.writer(resultsfile)
        wr.writerow(E_total_cmp)
     with open('./results/noniid_num_samples_Dataset_cifar_Round_'+str(num_rounds)+'_deadline_'+str(deadline)+'num_users_'+str(num_selected)+'total_clients_'+str(num_clients)+'_num_anntena_'+str(num_anntenas)+'.csv', 'a+', newline='') as resultsfile:
        wr = csv.writer(resultsfile)
        wr.writerow(num_samples)
############################################
#### Initializing models and optimizer  ####
############################################
Threshold = [1.1,0.9,0.8,0.7,0.6,0.5,0.2]#### [0.9,0.8,....]global model ##########
train_loader, test_loader = get_data_loaders(classes_pc=classes_pc, nclients= num_clients,
                                                          batch_size=batch_size,verbose=True)
# for x, y in train_loader[2]:
#     print('train_loader:',y)
Antennts = [1,2,4,8]#[1,2,4,8]
for num_anntennas in Antennts:
    for count,threshold in enumerate(Threshold):
        global_model =  VGG('VGG19').cuda()
        channel_gain, Beta= RF_channel_gain(num_clients,num_anntennas)
        Beta = [Beta[k].item(0) for k in range(num_clients)]
        ############# client models ###############################
        client_models = [ VGG('VGG19').cuda() for _ in range(num_selected)]
        for model in client_models:
            model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global modle 
        
        ###### optimizers ################
        opt = [optim.SGD(model.parameters(), lr=0.001) for model in client_models]
        
        ####### baseline data ############

        # loader_fixed = baseline_data(baseline_num)
        
        
        ###### Loading the non-i.i.d data ######
        # train_loader, test_loader = get_data_loaders(classes_pc=classes_pc, nclients= num_clients,
        #                                                       batch_size=batch_size,verbose=True)
        ###### Loading i.i.d data #################################
        # train_loader, test_loader= Process_IID_cifar10(num_clients,batch_size)
        
        losses_train = []
        losses_test = []
        acc_test = []
        losses_retrain=[]
        T_total = []
        E_total_upload = []
        E_total_cmp = []
        num_samples = []

        # Runnining FL
        for r in range(num_rounds):    #Communication round
            # select random clients
            np.random.seed(r)
            client_idx = np.random.permutation(num_clients)[:num_selected]
            client_lens = [len(train_loader[idx]) for idx in client_idx]
            # print(client_idx)
            # print(np.array(client_lens)*batch_size)
            # client update
            loss = 0
            E_up = 0
            E_cmp = 0
            num_smpl=0
            for i in tqdm(range(num_selected)):
                global_model_sync(client_models[i], global_model)
                #### COMPUTE THE NEW SAMPLES AND RETURNED LOSS ############
                l,D_k_NEW= client_update(client_models[i], opt[i], train_loader[client_idx[i]], epochs,threshold)
                loss =loss+ l
                # print('size:',sys.getsizeof(D_k_NEW))
                num_smpl = num_smpl + D_k_NEW
                T_upload = get_uploading_time(deadline,len(train_loader[client_idx[i]])*batch_size,D_k_NEW,phi,fmin,fmax,Beta[i],epochs)
                #print('Beta[k]',Beta[i],'Tupload:',T_upload)
                P_upload = get_Ptransmit(T_upload, Beta[i])
                E_upload = P_upload*T_upload     
                E_up = E_up + E_upload
                E_computaion = get_E_cmp(T_upload,D_k_NEW*10**(4), epochs, deadline) 
                E_cmp = E_cmp +E_computaion
                # print('T_upload',T_upload, 'P_upload',P_upload,'E_upload',E_upload,'E_computaion',E_computaion)
                ################# Compute the uploading time based on new samples ########
            E_total_upload.append(E_up)
            E_total_cmp.append(E_cmp)
            losses_train.append(loss)
            num_samples.append(num_smpl)
        
            # server aggregate
            #### retraining on the global server
            loss_retrain =0
            # for i in tqdm(range(num_selected)):
            #   l2,_= client_update(client_models[i], opt[i], loader_fixed,epochs, threshold)
            #   loss_retrain =loss_retrain + l2
            losses_retrain.append(loss_retrain)
            
            ### Aggregating the models
            server_aggregate(global_model, client_models,client_lens)
            test_loss, acc = test(global_model, test_loader)
            losses_test.append(test_loss)
            acc_test.append(acc)
            print('%d-th round' % r)
            print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))
        saveresults(acc_test,losses_test,E_total_upload,E_total_cmp,num_samples,num_anntennas) 
        gc.collect()
        torch.cuda.empty_cache()  
        del global_model
        del client_models     
