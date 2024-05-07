# Fix randomness and hide warnings
seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)

import logging

import pickle
import os

from data_tools import *

PROCCESS_DIR = '/home/tometaro/Documents/Thesis/processed_data/processed/'
UNPROCESS_DIR = '/home/tometaro/Documents/Thesis/processed_data/raw/'

# LOAD DATA

data = []

for file in os.listdir(UNPROCESS_DIR):
    with open(UNPROCESS_DIR+file, 'rb+') as f:
        data += [pickle.load(f)]


# REMOVE MISSING VALUES

encoding=''
recognition = ''
for i in range(len(data)):
    for j in range(len(data[i])):
        guide = np.any(np.isnan(data[i][j].encoding),axis=1)
        
        if np.sum(guide)>len(data[i][j].encoding)*0.2:
            data[i][j].encoding = None
            encoding+= f'{data[i][j].group} {data[i][j].subject} {data[i][j].trial}\n'
        else:
            data[i][j].encoding=data[i][j].encoding[~guide]

        guide = np.any(np.isnan(data[i][j].recognition),axis=1)
        
        if np.sum(guide)>len(data[i][j].recognition)*0.2:
            data[i][j].recognition = None
            recognition+= f'{data[i][j].group} {data[i][j].subject} {data[i][j].trial}\n'
        else:
            data[i][j].recognition=data[i][j].recognition[~guide]

with open("enconding_removed.txt", 'w') as output:
    output.write(encoding)
    
with open("recognition_removed.txt", 'w') as output:
    output.write(recognition)


# REMOVE BEST AND WORST PERFORMERS

results={'control':{},'patient': {}}

max_control=(0,-np.inf)
max_patient=(0,-np.inf)
min_control=(0,np.inf)
min_patient=(0,np.inf)

for i in range(len(data)):
    group = data[i][0].group
    subject = data[i][0].subject
    results[group][subject] = 0
    for j in range(len(data[i])):
        results[group][subject] += data[i][j].isCorrect
    
    if subject%1==0:
        if group == 'control':
            max_control = (i,results[group][subject]) if results[group][subject]>max_control[1] else max_control
            min_control = (i,results[group][subject]) if results[group][subject]<min_control[1] else min_control
        else:
            max_patient = (i,results[group][subject]) if results[group][subject]>max_patient[1] else max_patient
            min_patient = (i,results[group][subject]) if results[group][subject]<min_patient[1] else min_patient
            
print(f'Worst Performer - Patients: {data[min_patient[0]][0].subject} with {min_patient[1]} correct answers')
print(f'Best Performer - Patients: {data[max_patient[0]][0].subject} with {max_patient[1]} correct answers')
print(f'Worst Performer - Controls: {data[min_control[0]][0].subject} with {min_control[1]} correct answers')
print(f'Best Performer - Controls: {data[max_control[0]][0].subject} with {max_control[1]} correct answers')

data.pop(min_patient[0])
data.pop(max_patient[0])
data.pop(min_control[0])
data.pop(max_control[0])


# SAVE DATA

for d in data:
    group=d[0].group
    subject=int(d[0].subject//1)
    directory = 'first/' if d[0].subject%1==0 else 'second/'
    
    for i in range(len(d)):
        if d[i].encoding is not None:
            d[i].build_scanpath(crop = ((1600-700)/2, (900-700)/2,1600-(1600-700)/2, 900-(900-700)/2), resize=(512,512))
    
    with open(f'{PROCCESS_DIR}{directory}{group}_s{subject}.pkl', 'wb') as file:
        pickle.dump(d, file)