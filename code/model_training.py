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

DATASETS_DIR = ""

epochs = 100
batchsize = 32
cnnChannels = (8, 8)
cnnVectorSize = 16
rnnVectorSize = 32

COLUMNS = 'test_set,participant,group,mci_likelyhood_1,hc_likelyhood_1,mci_likelyhood_2,hc_likelyhood_2,mci_likelyhood_3,hc_likelyhood_3,mci_likelyhood_4,hc_likelyhood_4,mci_likelyhood_5,hc_likelyhood_5\n'

with open(f"{DATASETS_DIR}processed.pkl", "rb") as file:
    dataset = pickle.load(file)

dataset["heatmaps"] = torch.unsqueeze(
    torch.tensor(dataset["heatmaps"], dtype=torch.uint8), dim=1
)


dataset["eyeTracking"] = torch.tensor(dataset["eyeTracking"], dtype=torch.float32)


for i in range(len(dataset["participant"])):
    if dataset["participant"][i] < 100:
        dataset["eyeTracking"][i, :, [0, 1, 3, 4]] = (
            dataset["eyeTracking"][i, :, [0, 1, 3, 4]] / 400
        )
    else:
        dataset["eyeTracking"][i, :, [0, 1, 3, 4]] = (
            dataset["eyeTracking"][i, :, [0, 1, 3, 4]] / 350
        )



for appendix in ['_luz']:
    with open(f"{DATASETS_DIR}guides{appendix}.pkl", "rb") as file:
        guides = pickle.load(file)
        
    with open(f'results{appendix}.csv', "a") as f:
        f.write(COLUMNS)
        
    for k, (tt, ts) in enumerate(guides):
        test_set = VTNet_Dataset(
            scanpaths=dataset["heatmaps"][tt, :, :, :],
            rawdata=dataset["eyeTracking"][tt],
            subject=torch.tensor(dataset["participant"][tt]),
            groups=torch.tensor(dataset["group"][tt]),
        )

        mean_pupil_1 = torch.mean(test_set.rawdata[:, :, 2])
        std_pupil_1 = torch.std(test_set.rawdata[:, :, 2])

        mean_pupil_2 = torch.mean(test_set.rawdata[:, :, 5])
        std_pupil_2 = torch.std(test_set.rawdata[:, :, 5])

        test_set.rawdata[:, :, 2] = (test_set.rawdata[:, :, 2] - mean_pupil_1) / std_pupil_1
        test_set.rawdata[:, :, 5] = (test_set.rawdata[:, :, 5] - mean_pupil_2) / std_pupil_2

        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        models = []
        graphs = []
        for n in range(len(ts)):
            # Load Data
            train_set = VTNet_Dataset(
                scanpaths=dataset["heatmaps"][ts[n], :, :, :],
                rawdata=dataset["eyeTracking"][ts[n]],
                subject=torch.tensor(dataset["participant"][ts[n]]),
                groups=torch.tensor(dataset["group"][ts[n]]),
            )

            mean_pupil_1 = torch.mean(train_set.rawdata[:, :, 2])
            std_pupil_1 = torch.std(train_set.rawdata[:, :, 2])

            mean_pupil_2 = torch.mean(train_set.rawdata[:, :, 5])
            std_pupil_2 = torch.std(train_set.rawdata[:, :, 5])

            train_set.rawdata[:, :, 2] = (
                train_set.rawdata[:, :, 2] - mean_pupil_1
            ) / std_pupil_1
            train_set.rawdata[:, :, 5] = (
                train_set.rawdata[:, :, 5] - mean_pupil_2
            ) / std_pupil_2

            trainloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)

            del train_set

            # Create Model
            models += [
                VETNet(
                    timeseries_size=dataset["eyeTracking"][0][0].shape,
                    scanpath_size=dataset["heatmaps"].shape[1:],
                    vector_shape=(rnnVectorSize, cnnVectorSize),
                    cnn_shape=cnnChannels,
                ).to(device)
            ]

            # Set CallBacks and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(models[-1].parameters(), lr=1e-5)

            lr_tracker = ReduceLROnPlateau(5, 0.5, mode="min", minimum_lr=1e-6)
            earlystop_tracker = EarlyStopping(10, mode="min")

            # Train Model
            graphs += [
                models[-1].train(
                    optimizer=optimizer,
                    criterion=criterion,
                    train_loader=trainloader,
                    epochs=epochs,
                    lr_tracker=lr_tracker,
                    earlystop_tracker=earlystop_tracker,
                    device=device,
                )
            ]

        # Test Results per subject
        subjects = torch.unique(test_set.subject)


        with torch.no_grad():
            csvs = []
            for j, subject in enumerate(subjects):
                guide = test_set.subject == subject
                inputs, labels = test_set[guide]
                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
                labels = labels.to(device)

                csv = torch.transpose(torch.stack([torch.ones(torch.sum(guide).cpu()).cpu()*k,test_set.subject[guide].cpu(),labels.cpu()]),0,1)
                
                for model in models:
                    outputs = model(*inputs)
                    csv = torch.hstack([csv, outputs.cpu()])
                
                csvs += [csv]
        
        csvs = torch.concat(csvs)
        csvs = csvs.cpu().numpy()
        
        with open(f'results{appendix}.csv', "ab") as f:
            np.savetxt(f, csvs, delimiter=',')
        


        print(
            strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            appendix,
            rnnVectorSize,
            cnnChannels,
            cnnVectorSize,
            k,
        )   
