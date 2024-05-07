# Fix randomness and hide warnings
import torch
from multiprocessing import Pool
from data_tools import *

PROCCESS_DIC = '../processed_data/'

dataset = VTNet_Dataset(path=PROCCESS_DIC)

torch.save(dataset, 'tensor.pt')