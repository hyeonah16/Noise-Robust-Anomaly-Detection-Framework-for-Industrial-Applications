import torch.nn.functional as F

from homura import lr_scheduler, optim, reporters
from homura.trainers import SupervisedTrainer as Trainer
from homura.vision import DATASET_REGISTRY
from senet.baseline import resnet20
from senet.se_resnet import se_resnet20

import sys, os, time, glob, time, pdb, cv2
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import re
from pathlib import Path

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as Data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from senet.se_resnet import unet
from senet.se_resnet import UNet

def load_paths():
    normaldatapath = glob.glob("data/normal/**/*.jpg",recursive=True)
    normaldatapath = [str(Path(p).as_posix()) for p in normaldatapath]
    abnormaldatapath = glob.glob("data/abnormal/**/*.jpg", recursive=True)
    abnormaldatapath = [str(Path(p).as_posix()) for p in abnormaldatapath]

    with open('data/metadata_valid.json', 'r') as f:
        validdatajson = json.load(f)
    
    validdatalist = [*validdatajson]

    validpath = []
    count = 0
    temp = []

    for path in normaldatapath:
        if path.split('/')[2] in validdatalist:
            temp.append(path)
            count += 1
            
        if count == 8:
            temp = sorted(temp, key=lambda s: int(re.compile(r'\d+').search(s.split('/')[3][:-4]).group()))
            validpath.extend(temp)
            temp = []
            count = 0
        
    print(len(validpath))
        
    with open('data/metadata_test.json', 'r') as f:
        testdatajson = json.load(f)
        
    testdatalist = [*testdatajson]

    testpath = []
    count = 0
    temp = []

    for path in normaldatapath:
        if path.split('/')[2] in testdatalist:
            temp.append(path)
            count += 1
            
        if count == 16:
            temp = sorted(temp, key=lambda s: int(re.compile(r'\d+').search(s.split('/')[3][:-4]).group()))
            testpath.extend(temp)
            temp = []
            count = 0
            
    for path in abnormaldatapath:
        matching = [s for s in abnormaldatapath if path.split('/')[2] in s]
        if path.split('/')[2] in testdatalist:
            temp.append(path)
            count += 1
            
        if count == len(matching):
            temp = sorted(temp, key=lambda s: int(re.compile(r'\d+').search(s.split('/')[3][:-4]).group()))
            testpath.extend(temp)
            temp = []
            count = 0

    return validpath, testpath
        
class Custom_Dataset(Data.Dataset):
    
    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x, mode='RGB')
            x = self.transform(x)
            y = Image.fromarray(y, mode='RGB')
            y = self.transform(y)
            
        return x, y
    
def Custom_loader(args):

    transform = transforms.Compose([transforms.ToTensor()])
    
    train_x = [cv2.imread(img) for img in train_x_path]
    print(len(train_x))

    train_y = [cv2.imread(img) for img in train_y_path]
    print(len(train_y))
    
    data_train = Custom_Dataset(train_x, train_y, transform)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    
    valid_x = [cv2.imread(img) for img in valid_x_path]
    print(len(valid_x))

    valid_y = [cv2.imread(img) for img in valid_y_path]
    print(len(valid_y))
    
    data_valid = Custom_Dataset(valid_x, valid_y, transform)
    valid_loader = DataLoader(data_valid, batch_size=1, 
                                  shuffle=False, num_workers=0)

    test_x = [cv2.imread(img) for img in test_x_path]
    print(len(test_x))

    test_y = [cv2.imread(img) for img in test_y_path]
    print(len(test_y))
    
    data_test = Custom_Dataset(test_x, test_y, transform)
    test_loader = DataLoader(data_test, batch_size=1, 
                                  shuffle=False, num_workers=0)    

    return train_loader, valid_loader, test_loader

class FeatureExtractor:
    def __init__(self, model):
        
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model = self.embed_hooks(self.model)
    
    def forward(self, x):
        # x = 1, 3, 240, 320
        def concat_embeddings(*xs):
            zs = [torch.nn.functional.interpolate(x, size=(12,16)) for x in xs] # zs[0] = 1, 512, 6, 8
            return torch.cat(zs, dim=1)
        
        self.features = []
        x = x.to(self.device)
        _ = self.model(x)
        
        features = [feature.cpu() for feature in self.features] # feature = 1, 520, 30, 40
        
        embedding = concat_embeddings(*features)
        
        return embedding
        
    def transform(self, dataloader, description="Extracting...", **kwargs):
        def flatten_NHW(tensor):
            return tensor.permute(0,2,3,1).flatten(start_dim = 1, end_dim=3)
        
        if isinstance(dataloader, torch.utils.data.DataLoader):
            embeddings_list = list()
            for x in dataloader:
                embedding = self.forward(x[0])
                embeddings_list.append(flatten_NHW(embedding).cpu().numpy())
            return np.concatenate(embeddings_list, axis=0)
                
        elif isinstance(dataloader, torch.Tensor):
            embedding = self.forward(dataloader)
            return flatten_NHW(embedding).cpu().numpy()
            
    def embed_hooks(self, model):
        def hook(module, input, output):
            self.features.append(output)

        model.down_path[3].block[4].register_forward_hook(hook) # 2048
        
        return model

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # load data
    validpath, testpath = load_paths()
    transform = transforms.Compose([transforms.ToTensor()])

    validation = [cv2.imread(img) for img in validpath]
    data_valid = Custom_Dataset(validation, validation, transform)
    valid_loader = DataLoader(data_valid, batch_size=1, shuffle=False, num_workers=0)

    test = [cv2.imread(img) for img in testpath]
    data_test = Custom_Dataset(test, test, transform)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0)

    # load feature extractor
    model = unet(n_classes = 2, depth = 4, padding = True).to(device)
    parameter = torch.load('output/trained_model.pth')
    model.load_state_dict(parameter)
    model.eval()
    model.to(device)
    extractor = FeatureExtractor(model)

    # extract embedding vectors
    embeddings = extractor.transform(valid_loader, batch_size=1, num_workers=0)
    embeddings_test = extractor.transform(test_loader, batch_size=1, num_workers=0)

    # calcuate threshold
    valid_diff_list = []
    threshold_100 = 0
    for i in range(len(validpath)):
        if i % 8 != 7:
            t1 = embeddings[i]
            t2 = embeddings[i+1]
            diff = np.mean(((t1-t2)**2)**0.5)
            valid_diff_list.append(diff)
            if diff > threshold_100:
                threshold_100 = diff

    # inference
    test_score = [[None] * 2 for _ in range(len(testpath))]
    thr = threshold_100
    abn_list = os.listdir('data/abnormal')
    wrong = False
    correct = False
    nwn = 0
    acn = 0
    normal = 0
    abnormal = 0
    k = 0
    for i in range(len(testpath)-7):
        if testpath[i].split('/')[2] == testpath[i+7].split('/')[2]:
            for j in range(7):
                t1 = embeddings_test[i+j]
                t2 = embeddings_test[i+j+1]
                diff = np.mean(((t1-t2)**2)**0.5)
                if testpath[i+j].split('/')[2] not in abn_list:
                    if diff > thr:
                        wrong = True
                elif testpath[i+j].split('/')[2] in abn_list:
                    if diff > thr:
                        correct = True

            if testpath[i].split('/')[2] not in abn_list:
                normal += 1
                if wrong:
                    nwn += 1
                    wrong = False
            else:
                abnormal += 1
                if correct:
                    acn += 1
                    correct = False

    precision = acn/(nwn+acn)*100
    recall = acn/abnormal*100
    f1_score = 2*precision*recall/(precision+recall)
    accuracy = (acn+normal-nwn)/(normal+abnormal)*100

if __name__ == "__main__":
    main()