from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
import pickle
import os
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import gc
import psutil

class Trial:
    
    def __init__(self, group: str, subject: float, age: int, education: int, trial: int) -> None:
        
        self.group = group
        self.subject = subject
        self.age = age
        self.education = education
        self.trial = trial
        
        self.encoding = None
        self.recognition = None
        
        self.encodingScanPath = None
        self.recognitionScanPath = None
        
        self.oldImage = None
        self.newImage = None
        self.isCorrect = None
        self.isOldRight = None
        
    def set_encoding(self, data):
        self.encoding = data
    
    def set_recognition(self, data, oldImage : str, newImage : str, isCorrect : bool, isOldRight : bool):
        
        self.recognition = data
        
        self.oldImage = oldImage
        self.newImage = newImage
        self.isCorrect = isCorrect
        self.isOldRight = isOldRight
        
    
    def build_scanpath(self, line_width = 5, point_size = 7, resize = None, crop = None):
        coords = list(zip(self.encoding[:,1],self.encoding[:,2]))
    
        scan_path = Image.new(mode='L',size=(1600,900),color=255)
        draw = ImageDraw.Draw(scan_path)
        draw.line(xy=coords, 
                fill=0, width = line_width)
        
        for coord in coords:
            draw.regular_polygon(bounding_circle=(coord,point_size),
                    n_sides=20 , fill=0, width = point_size)
        
        scan_path = scan_path.resize(resize) if resize else scan_path
        scan_path = scan_path.crop(crop) if crop else scan_path
        
        self.encodingScanPath = scan_path

class VTNet_Dataset (Dataset):

    def __init__(self, scanpaths = [], rawdata = [], groups = [], subject = []):
        
        if len(scanpaths)==0:
            path = '../processed_data/'

            self.scanpaths = []
            self.rawdata = []
            self.groups = []
            self.subject = []

            for file in os.listdir(path):
                with open(path+file, 'rb+') as f:
                    self.scanpaths += list(map(lambda x: x.encodingScanPath or None, pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.rawdata += list(map(lambda x: x.encoding[:,[0,3,4,5,6,7,8]] if x.encodingScanPath else None,pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.groups += list(map(lambda x: (0 if x.group[0]=='c' else 1) if x.encodingScanPath else None,pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.subject += list(map(lambda x: x.subject if x.encodingScanPath else None,pickle.load(f)))

            self.groups = np.array(self.groups)

            def new_len(x):
                return len(x) if hasattr(x, '__iter__') else 0

            max_len = max(map(new_len,self.rawdata))
            self.rawdata = np.array([np.pad(rd,((max_len-len(rd),0),(0,0)),constant_values=0) if hasattr(rd, '__iter__') else None for rd in self.rawdata], dtype=object)[self.groups!=None]
            self.rawdata = torch.tensor(np.array(list(self.rawdata)), dtype=torch.float32)
            self.rawdata[:,:,[3,6]] = (self.rawdata[:,:,[3,6]]-torch.unsqueeze(torch.mean(self.rawdata[:,:,[3,6]],dim=1),dim=1))/(torch.unsqueeze(torch.std(self.rawdata[:,:,[3,6]],dim=1),dim=1)+1e-10)
            self.rawdata[:,:,0] = self.rawdata[:,:,0]/1500 - 1
            self.rawdata[:,:,[1,4]] = self.rawdata[:,:,[1,4]]/800 -1
            self.rawdata[:,:,[2,5]] = self.rawdata[:,:,[2,5]]/450 - 1
            self.subject = torch.tensor(np.array(self.subject)[self.groups!=None].astype(int))
            self.scanpaths = np.array(self.scanpaths, dtype=object)[self.groups!=None]
            self.scanpaths=[pil_to_tensor(rd) for rd in self.scanpaths]
            self.scanpaths = torch.stack(self.scanpaths, dim=0)
            self.groups = torch.tensor(self.groups[self.groups!=None].astype(int))

        else:
            self.scanpaths = scanpaths
            self.rawdata = rawdata
            self.groups = groups
            self.subject = subject

        print(torch.sum(self.groups==0),torch.sum(self.groups==1))

    def __getitem__(self, index):
        return self.rawdata[index], self.scanpaths[index], self.groups[index]
    
    def __len__(self):
        return len(self.groups)
    
    def doubleStratifiedSplit(self, split_fractions = 0.8, downsample=False):
        
        ind = np.arange(len(self.groups))
        
        if downsample:
            aux = self.groups==0 if torch.sum(self.groups==0)>=self.__len__()/2 else self.groups==1
            aux = ind[aux]
            aux = ind[aux[torch.randperm(aux.shape[0])[len(self.groups)-len(aux):]]]
            mask = torch.ones(len(ind),dtype=torch.bool)
            mask[aux] = False
            ind = ind[mask]

        sub_group = (self.groups[ind]*2-1)*self.subject[ind]
        
        sgkf = StratifiedGroupKFold(n_splits=int(1/(1-split_fractions)), random_state=None, shuffle=False)
        
        trainset_ind, aux_ind = list(sgkf.split(ind, self.groups[ind], sub_group))[0]
        
        sgkf = StratifiedGroupKFold(n_splits=2, random_state=None, shuffle=False)
        
        valset_ind, testset_ind = list(sgkf.split(ind[aux_ind], self.groups[aux_ind], sub_group[aux_ind]))[0]
        
        return ind[trainset_ind], ind[aux_ind[valset_ind]], ind[aux_ind[testset_ind]]