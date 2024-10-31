from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from torchvision.io import ImageReadMode, read_image
import pickle
import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedGroupKFold

class Trial:
    
    def __init__(self, group: str, subject: float, age: int, education: int, trial: str) -> None:
        
        self.group = group
        self.subject = subject
        self.age = age
        self.education = education
        self.trial = trial
        
        self.encoding = None
        self.recognition = None
        
        self.encodingScanPath = None
        self.recognitionScanPath = None
        
        self.encodingHeatMap = None
        self.recognitionHeatMap = None
        
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
        
    
    def build_scanpath(self, line_width = 5, point_size = 7, isEncoding = True, resize = None):
        
        if isEncoding:
            coords = list(zip(self.encoding[:,1]-450,self.encoding[:,2]-100))
        else:
            coords = list(zip(self.recognition[:,1]-100,self.recognition[:,2]-170))
    
        size = (700,700) if isEncoding else (1400,560)
    
        scan_path = Image.new(mode='L',size=size,color=255)
        draw = ImageDraw.Draw(scan_path)
        draw.line(xy=coords, 
                fill=0, width = line_width)
        
        for coord in coords:
            draw.regular_polygon(bounding_circle=(coord,point_size),
                    n_sides=20 , fill=0, width = point_size)
        
        scan_path = scan_path.resize(resize) if resize else scan_path
        
        if isEncoding:
            self.encodingScanPath = scan_path
        else:
            self.recognitionScanPath = scan_path
        
    def build_heatmap(self, distance, angle, isEncoding = True,resize = None):
        def multivariate_gaussian(shape, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            X = np.linspace(-shape[0]/2, shape[0]/2, shape[0])
            Y = np.linspace(-shape[1]/2, shape[0]/2, shape[1])
            
            #N = np.sqrt((2*np.pi)**2 * (Sigma[0]*Sigma[1])**2)
            N=1

            return np.outer(np.exp(-(Y-mu[1])**2/Sigma[1]**2/2),np.exp(-(X-mu[0])**2/Sigma[0]**2/2)) / N
            
        if isEncoding:
            coords = list(zip(self.encoding[:,1]-800,self.encoding[:,2]-450))
        else:
            coords = list(zip(self.recognition[:,1]-800,self.recognition[:,2]-450))
    
        size = (700,700) if isEncoding else (1400,560)
        Sigma = np.array([1800*np.tan(angle*np.pi/180)*distance/33.8 , 900*np.tan(angle*np.pi/180)*distance/27.1])
        
        heatmap = np.zeros(shape=(size[1],size[0]))
        
        for coord in coords:
            heatmap += multivariate_gaussian(size,coord,Sigma)
            
        #heatmap = heatmap/np.max(heatmap)
        #heatmap /= np.sum(heatmap)
        heatmap *= 255/len(coords)
            
        heatmap = cv2.resize(heatmap, dsize=resize) if resize else heatmap
        
        heatmap = heatmap.astype(np.uint8)
        
        if isEncoding:
            self.encodingHeatMap = heatmap
        else:
            self.recognitionHeatMap = heatmap
        

class VTNet_Dataset(Dataset):

    def __init__(self, path='', scanpaths = [], rawdata = [], groups = [], subject = [],score = [],image = [], useHeatmap = False, IMAGE_DIR = '', imageMode = None):
        self.IMAGE_DIR = IMAGE_DIR
        if len(scanpaths)==0:

            self.scanpaths = []
            self.rawdata = []
            self.groups = []
            self.subject = []
            self.score = []
            self.image = []
            
            for file in [x for x in os.listdir(path) if x[-3:]=='pkl']:
                with open(path+file, 'rb+') as f:
                    if not useHeatmap:
                        self.scanpaths += list(map(lambda x: x.encodingScanPath if x.encoding is not None else None, pickle.load(f)))
                    else:
                        self.scanpaths += list(map(lambda x: x.encodingHeatMap if x.encoding is not None else None, pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.rawdata += list(map(lambda x: x.encoding[:,[0,3,4,5,6,7,8]] if x.encoding is not None else None,pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.groups += list(map(lambda x: (1 if x.group[0]=='c' else 0) if x.encoding is not None else None,pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.subject += list(map(lambda x: x.subject if x.encoding is not None else None,pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.image += list(map(lambda x: x.oldImage if x.encoding is not None else None,pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.score += list(map(lambda x: x.isCorrect if x.encoding is not None else None,pickle.load(f)))

            self.groups = np.array(self.groups)

            del file, f
            
            def new_len(x):
                return len(x) if hasattr(x, '__iter__') else 0

            max_len = max(map(new_len,self.rawdata))
            self.rawdata = np.array([np.pad(rd,((max_len-len(rd),0),(0,0)),constant_values=0) if hasattr(rd, '__iter__') else None for rd in self.rawdata], dtype=object)[self.groups!=None]
            
            self.rawdata = torch.tensor(np.array(list(self.rawdata)), dtype=torch.float32)
            
            self.rawdata[:,:,0] = self.rawdata[:,:,0]/1500 - 1
            self.rawdata[:,:,[1,4]] = (self.rawdata[:,:,[1,4]]-800)/350
            self.rawdata[:,:,[2,5]] = (self.rawdata[:,:,[2,5]]-450)/350
            
            self.subject = torch.tensor(np.array(self.subject)[self.groups!=None].astype(np.float16))
            self.scanpaths = np.array(self.scanpaths, dtype=object)[self.groups!=None]
            
            self.scanpaths=[pil_to_tensor(rd) for rd in self.scanpaths] if not useHeatmap else np.array(list(self.scanpaths),dtype=np.uint8)
            self.scanpaths = torch.stack(self.scanpaths, dim=0) if not useHeatmap else torch.unsqueeze(torch.tensor(self.scanpaths,dtype=torch.uint8),dim=1)
            
            self.score = torch.tensor(np.array(self.score)[self.groups!=None].astype(bool))
            self.image = np.array(self.image)[self.groups!=None].astype(str)
            self.groups = torch.tensor(self.groups[self.groups!=None].astype(int))

        else:
            self.scanpaths = scanpaths
            self.rawdata = rawdata
            self.groups = groups
            self.subject = subject
            self.score = score
            self.image = image
        
        if IMAGE_DIR != '':
            images = {}
            for im in os.listdir(f'{IMAGE_DIR}'):
                images[im] = torch.unsqueeze(read_image(IMAGE_DIR+im, mode =ImageReadMode.GRAY),dim=0) if imageMode=='BW' else torch.unsqueeze(read_image(IMAGE_DIR+im, mode =ImageReadMode.RGB),dim=0)
            i=0
            output_image = []
            for im in self.image:
                output_image += [images[im]]
            del images
            output_image = torch.cat(output_image, dim=0)
            self.scanpaths = self.scanpaths*output_image if imageMode == 'MIX' else torch.cat((self.scanpaths,output_image), dim=1)
            del output_image
        
        print(torch.sum(self.groups==0),torch.sum(self.groups==1))

    def __getitem__(self, index):
        if self.score!=[]:
            return self.rawdata[index], self.scanpaths[index], self.score[index],self.groups[index]
        else:
            return self.rawdata[index], self.scanpaths[index], self.groups[index]
    
    def __len__(self):
        return len(self.groups)
    
    def doubleStratifiedSplit(self, folds = 10):
        ind = np.arange(len(self.groups))

        sub_group = (self.groups[ind]*2-1)*self.subject[ind].type(torch.int8)
        
        sgkf = StratifiedGroupKFold(n_splits=folds, random_state=None, shuffle=False)
        
        output = []
        ind_folds = []
        
        for _, aux_ind in sgkf.split(ind, self.groups[ind], sub_group):
            ind_folds += [ind[aux_ind]]

        for i in range(folds):
            output += [(np.hstack([ind_folds[j] for j in range(folds) if j not in [i%folds,(i+1)%folds]]),ind_folds[i%folds],ind_folds[(i+1)%folds])]
        return output
