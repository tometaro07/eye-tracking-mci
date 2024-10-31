import pickle
import os
import torch

from data_tools import *

PROCESS_DIR = ''
DATASET_DIR = ''

dataset=VTNet_Dataset(path=PROCESS_DIR,useHeatmap=True)

with open(f'{DATASET_DIR}VTNet_heatmap/data.pkl', 'wb') as file:
    pickle.dump(dataset, file)

guides = dataset.doubleStratifiedSplit()

with open(f'{DATASET_DIR}guides.pkl', 'wb') as file:
    pickle.dump(guides, file)

del dataset, guides

dataset2= VTNet_Dataset(path=PROCESS_DIR)

with open(f'{DATASET_DIR}VTNet/data.pkl', 'wb') as file:
    pickle.dump(dataset2, file)
