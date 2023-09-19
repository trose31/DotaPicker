"""This file defines the TextDataset object such as it can properly
interact with pytorch. """

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize

class CustomTextDataset(Dataset):    
    def __init__(self, txt, labels, transform = None, target_transform = None):        
        self.text = self.fetchtxt(txt)
        self.labels = self.fetchLabels(labels)
        self.transform = transform
        self.target_transform = target_transform

    def fetchtxt(self, file):
        f = open(file,'r')
        lines = f.read().splitlines()
        output = [[int(i) for i in line.replace('[','').replace(']','').split(', ')] for line in lines] #Converts the string representation of an array to actual array, fix later
        f.close
        return output

    def fetchLabels(self, file):
        f = open(file,'r')
        lines = f.read().splitlines()
        output = [int(i) for i in lines]
        f.close
        return output
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        
        if self.transform:
            text = torch.FloatTensor(text)

        if self.target_transform:
            label = self.target_transform(label)
            
        output = text, label
        return output
