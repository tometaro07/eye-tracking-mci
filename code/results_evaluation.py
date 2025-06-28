import os
import warnings

import pandas as pd
import numpy as np
import random

import torch
from torch import nn
import torch.version
from torch.utils.data import DataLoader
import torch.optim as optim

from time import gmtime, strftime
import pickle

from data_tools import VTNet_Dataset
from models import VETNet, ReduceLROnPlateau, EarlyStopping

seed = 42

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)

np.random.seed(seed)

random.seed(seed)

torch.manual_seed(seed)

torch.version.__version__
print("PyTorch Version:", torch.version.__version__)
print("Cuda Version:", torch.version.cuda, "\n")

print("Available devices:")
for i in range(torch.cuda.device_count()):
    print("\t", torch.cuda.get_device_properties(i).name)
    print(
        "\t\tMultiprocessor Count:",
        torch.cuda.get_device_properties(i).multi_processor_count,
    )
    print(
        "\t\tTotal Memory:",
        torch.cuda.get_device_properties(i).total_memory / 1024 / 1024,
        "MB",
    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\n", device)
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "start")

# Import other libraries

RESULTS_DIR = ""

df = pd.read_csv(f'{RESULTS_DIR}results.csv') 

test_sets = np.unique(df['test_set'])

for test_set in test_sets:
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    TPS = [0,0,0,0,0]
    TNS = [0,0,0,0,0]
    FPS = [0,0,0,0,0]
    FNS = [0,0,0,0,0]
    
    test_index = df['test_set']==test_set
    
    participants = np.unique(df[test_index]['participant'])
    
    for participant in participants:
        results = []
        participant_index = df['participant']==participant
        for i in range(1,6):
            results+=[np.quantile(np.argmax(df[test_index & participant_index][[f'mci_likelyhood_{i}',f'hc_likelyhood_{i}']], axis=1),q=0.5)]
            #results+=[np.argmax(np.mean(df[test_index & participant_index][[f'mci_likelyhood_{i}',f'hc_likelyhood_{i}']], axis=0))]
        result = np.quantile(results,q=0.5)
        true = df[test_index & participant_index]['group'].iloc[0]
        
        if true == 0:
            if result == 0:
                TP+=1
            else:
                FN+=1
        else:
            if result == 1:
                TN+=1
            else:
                FP+=1
                
        for i in range(5):
            if true == 0:
                if results[i] == 0:
                    TPS[i]+=1
                else:
                    FNS[i]+=1
            else:
                if results[i] == 1:
                    TNS[i]+=1
                else:
                    FPS[i]+=1

    print(participants)
    print(f'{test_set}        | MCI | HCs')
    print(f'Positive |  {TP}  | {FP}\t\tPositive |  {TPS[0]}  | {FPS[0]}\t\tPositive |  {TPS[1]}  | {FPS[1]}\t\tPositive |  {TPS[2]}  | {FPS[2]}\t\tPositive |  {TPS[3]}  | {FPS[3]}\t\tPositive |  {TPS[4]}  | {FPS[4]}\t\t')
    print(f'Negative |  {FN}  | {TN}\t\tNegative |  {FNS[0]}  | {TNS[0]}\t\tNegative |  {FNS[1]}  | {TNS[1]}\t\tNegative |  {FNS[2]}  | {TNS[2]}\t\tNegative |  {FNS[3]}  | {TNS[3]}\t\tNegative |  {FNS[4]}  | {TNS[4]}\t\t')
    print('Sensitivity:', TP/(TP+FN), 'Specificity:', TN/(TN+FP))
    print()
