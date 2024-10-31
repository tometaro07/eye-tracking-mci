# Fix randomness and hide warnings
seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'



from pdb import run
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)

import logging

import random
random.seed(seed)

# Import tensorflow
import torch
import torch.version
from torch.utils.data import DataLoader
import torch.optim as optim
from time import gmtime, strftime
import csv

torch.manual_seed(seed)

torch.version.__version__
print('PyTorch Version:',torch.version.__version__)
print('Cuda Version:',torch.version.cuda,'\n')

print('Available devices:')
for i in range(torch.cuda.device_count()):
   print('\t',torch.cuda.get_device_properties(i).name)
   print('\t\tMultiprocessor Count:',torch.cuda.get_device_properties(i).multi_processor_count)
   print('\t\tTotal Memory:',torch.cuda.get_device_properties(i).total_memory/1024/1024, 'MB')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\n',device)

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), 'start')

# Import other libraries
import cv2
from skimage import transform
import pickle
from data_tools import *
from models import *
from tqdm import tqdm


DATASETS_DIR = ''

batchsize = 32
classes = ['CONTROL', 'PATIENT']

with open(f'{DATASETS_DIR}guides.pkl', 'rb') as file:
    guides = pickle.load(file)

with open(f'{DATASETS_DIR}data.pkl', 'rb') as file:
    dataset = pickle.load(file)

cnnChannels = (8,8)
cnnVectorSize = 16
rnnVectorSize = 32

TP = 0
FN = 0
FP = 0
TN = 0

TP_sub = 0
FN_sub = 0
FP_sub = 0
TN_sub = 0

values = []

for k,(t,v,tt) in enumerate(guides):
    graphs = []
    # Load Data 
    train_set = VTNet_Dataset(scanpaths=dataset.scanpaths[t,:,:,:], rawdata=dataset.rawdata[t], subject=dataset.subject[t], groups=dataset.groups[t])

    mean_pupil_1 = torch.mean(train_set.rawdata[:,:,3])
    std_pupil_1 = torch.std(train_set.rawdata[:,:,3])

    mean_pupil_2 = torch.mean(train_set.rawdata[:,:,6])
    std_pupil_2 = torch.std(train_set.rawdata[:,:,6])

    train_set.rawdata[:,:,3] = (train_set.rawdata[:,:,3]-mean_pupil_1)/std_pupil_1
    train_set.rawdata[:,:,6] = (train_set.rawdata[:,:,6]-mean_pupil_2)/std_pupil_2

    trainloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    
    del train_set

    test_set = VTNet_Dataset(scanpaths=dataset.scanpaths[tt,:,:,:], rawdata=dataset.rawdata[tt], subject=dataset.subject[tt], groups=dataset.groups[tt])

    test_set.rawdata[:,:,3] = (test_set.rawdata[:,:,3]-mean_pupil_1)/std_pupil_1
    test_set.rawdata[:,:,6] = (test_set.rawdata[:,:,6]-mean_pupil_2)/std_pupil_2

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    
    val_set = VTNet_Dataset(scanpaths=dataset.scanpaths[v,:,:,:], rawdata=dataset.rawdata[v], subject=dataset.subject[v], groups=dataset.groups[v])

    val_set.rawdata[:,:,3] = (val_set.rawdata[:,:,3]-mean_pupil_1)/std_pupil_1
    val_set.rawdata[:,:,6] = (val_set.rawdata[:,:,6]-mean_pupil_2)/std_pupil_2
    
    valloader = DataLoader(val_set, batch_size=batchsize, shuffle=False)
    
    del val_set
    
    # Create Model
    model = VETNet(timeseries_size=dataset[0][0].shape, scanpath_size=dataset[0][1].shape,vector_shape=(rnnVectorSize,cnnVectorSize),cnn_shape=cnnChannels).to(device)
    
    # Set CallBacks and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    val_criterion = nn.CrossEntropyLoss()
    lr_tracker = ReduceLROnPlateau(5, 0.5, mode='min', minimum_lr=1e-6)
    earlystop_tracker = EarlyStopping(10, mode='min')
    
    # Train Model
    
    running_loss = []
    val_running_loss = []
    for epoch in range(1,101):  # loop over the dataset multiple times

        running_loss += [0.0]
        val_running_loss += [0.0]
        
        for input_rawdata, input_scanpath, labels in trainloader:
            input_rawdata = input_rawdata[:,:,1:].to(device)
            input_scanpath = (input_scanpath/128-1).to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_rawdata, input_scanpath)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss[-1] += loss.item()
    
        for val_rawdata, val_scanpath, val_labels in valloader:
            val_rawdata = val_rawdata.to(device)
            val_scanpath = (val_scanpath/128-1).to(device)
            val_labels = val_labels.to(device)
            
            val_outputs = model(val_rawdata[:,:,1:], val_scanpath)
            val_loss = val_criterion(val_outputs, val_labels)
            val_running_loss[-1] += val_loss.item()
                    
        graphs+=[[epoch, optimizer.param_groups[-1]['lr'], val_running_loss[-1], running_loss[-1]]]
        
        lr_tracker.check(value=val_running_loss[-1], optimizer=optimizer, model=model)
        
        if earlystop_tracker.check(value=val_running_loss[-1], model=model):
            break
    # Test Results
    with torch.no_grad():
        aTP = 0
        aFN = 0
        aFP = 0
        aTN = 0
        for input_rawdata, input_scanpath, labels in test_loader:
            
            input_rawdata = input_rawdata[:,:,1:].to(device)
            input_scanpath = (input_scanpath/128-1).to(device)
            labels = labels.to(device)
            outputs = model(input_rawdata, input_scanpath)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            TP += torch.sum((predicted==0)[labels==0])
            FN += torch.sum((predicted==1)[labels==0])
            FP += torch.sum((predicted==0)[labels==1])
            TN += torch.sum((predicted==1)[labels==1])
            
            aTP += torch.sum((predicted==0)[labels==0])
            aFN += torch.sum((predicted==1)[labels==0])
            aFP += torch.sum((predicted==0)[labels==1])
            aTN += torch.sum((predicted==1)[labels==1])

    # Test Results per subject
    subjects = test_set.subject*(test_set.groups*2-1)
    groups = torch.unique(subjects)
    groups[groups<0]=0
    groups[groups>0]=1
    
    aTP_sub = 0
    aFN_sub = 0
    aFP_sub = 0
    aTN_sub = 0
    
    with torch.no_grad():
        prev_subject = None
        
        predicted = []
        
        for j,(input_rawdata, input_scanpath, labels) in enumerate(test_loader):
            
            input_rawdata = input_rawdata[:,:,1:].to(device)
            input_scanpath = (input_scanpath/128-1).to(device)
            labels = labels.to(device)
            
            if prev_subject == subjects [j]:
                outputs += model(input_rawdata, input_scanpath)
            else:
                if prev_subject!=None:
                    values += [[k,prev_subject.item(),(1+np.sign(prev_subject.item()))/2,outputs[0,0].item(),outputs[0,1].item()]]
                predicted += [torch.max(outputs, 1)[1]]
                outputs = model(input_rawdata, input_scanpath)
            
            prev_subject = subjects[j]

        values += [[k,prev_subject.item(),(1+np.sign(prev_subject.item()))/2,outputs[0,0].item(),outputs[0,1].item()]]
        predicted += [torch.max(outputs, 1)[1]]
        predicted = torch.tensor(predicted[1:], dtype=torch.int64)
            
        TP_sub += torch.sum((predicted==0)[groups == 0])
        FN_sub += torch.sum((predicted==1)[groups == 0])
        FP_sub += torch.sum((predicted==0)[groups == 1])
        TN_sub += torch.sum((predicted==1)[groups == 1])
        
        aTP_sub += torch.sum((predicted==0)[groups == 0])
        aFN_sub += torch.sum((predicted==1)[groups == 0])
        aFP_sub += torch.sum((predicted==0)[groups == 1])
        aTN_sub += torch.sum((predicted==1)[groups == 1])
    
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), rnnVectorSize, cnnChannels, cnnVectorSize, k)
    
    sensitivity = aTP/(aTP+aFN)
    specificity = aTN/(aTN+aFP)
    
    sensitivity_sub = aTP_sub/(aTP_sub+aFN_sub)
    specificity_sub = aTN_sub/(aTN_sub+aFP_sub)
    with open('/home/tomasrocha/workspace/workspace/results.txt','a') as file:
        file.write(f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} {epoch} {rnnVectorSize} {cnnChannels} {cnnVectorSize} {k}\n')
        file.write(f'\tSensitivity: {sensitivity*100} %\n')
        file.write(f'\tSpecificity: {specificity*100} %\n\n')
        file.write(f'\t         | {classes[0]} | {classes[1]} \n\t---------|---------|----------\n')
        file.write(f'\tnegative |   {int(aTP)}   |   {int(aFP)}   \n')
        file.write(f'\tpositive |   {int(aFN)}   |   {int(aTN)}   \n\n')
    
    with open('/home/tomasrocha/workspace/workspace/results.txt','a') as file:
        file.write(f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} {epoch} {rnnVectorSize} {cnnChannels} {cnnVectorSize} {k} subject\n')
        file.write(f'\tSensitivity: {sensitivity_sub*100} %\n')
        file.write(f'\tSpecificity: {specificity_sub*100} %\n\n')
        file.write(f'\t         | {classes[0]} | {classes[1]} \n\t---------|---------|----------\n')
        file.write(f'\tnegative |   {int(aTP_sub)}   |   {int(aFP_sub)}   \n')
        file.write(f'\tpositive |   {int(aFN_sub)}   |   {int(aTN_sub)}   \n\n')
        
    with open(f'/home/tomasrocha/workspace/workspace/progress{k}_{rnnVectorSize}_{cnnChannels[0]}_{cnnChannels[1]}_{cnnVectorSize}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(graphs)

with open(f'/home/tomasrocha/workspace/workspace/{rnnVectorSize}_{cnnChannels[0]}_{cnnChannels[1]}_{cnnVectorSize}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(values)

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), rnnVectorSize, cnnChannels, cnnVectorSize,)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

sensitivity_sub = TP_sub/(TP_sub+FN_sub)
specificity_sub = TN_sub/(TN_sub+FP_sub)

with open('/home/tomasrocha/workspace/workspace/results.txt','a') as file:
    file.write(f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} {epoch} {rnnVectorSize} {cnnChannels} {cnnVectorSize}\n')
    file.write(f'\tSensitivity: {sensitivity*100} %\n')
    file.write(f'\tSpecificity: {specificity*100} %\n\n')
    file.write(f'\t         | {classes[0]} | {classes[1]} \n\t---------|---------|----------\n')
    file.write(f'\tnegative |   {int(TP)}   |   {int(FP)}   \n')
    file.write(f'\tpositive |   {int(FN)}   |   {int(TN)}   \n\n')

with open('/home/tomasrocha/workspace/workspace/results.txt','a') as file:
    file.write(f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} {epoch} {rnnVectorSize} {cnnChannels} {cnnVectorSize} subject\n')
    file.write(f'\tSensitivity: {sensitivity_sub*100} %\n')
    file.write(f'\tSpecificity: {specificity_sub*100} %\n\n')
    file.write(f'\t         | {classes[0]} | {classes[1]} \n\t---------|---------|----------\n')
    file.write(f'\tnegative |   {int(TP_sub)}   |   {int(FP_sub)}   \n')
    file.write(f'\tpositive |   {int(FN_sub)}   |   {int(TN_sub)}   \n\n')
