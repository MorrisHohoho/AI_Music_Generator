import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    def __init__(self,data_dir, split_method,max_seq=512):
        self.data_dir = data_dir
        self.max_seq = max_seq
        self.vocab_len = len(pickle.load(
            open(os.path.join(data_dir,'char2id.pkl'),'rb')))
        self.x,self.y = split_method(pickle.load(open(os.path.join(self.data_dir,'songs.pkl'),'rb')),
                                 self.max_seq)

    def __len__(self):
        return len(self.y) 
    
    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]
        
        return one_hot_encode(x,self.vocab_len), one_hot_encode(y,self.vocab_len)

def one_hot_encode(data,vocab_len):
    encoded = torch.zeros(len(data),vocab_len)
    for i,symbol in np.ndenumerate(data):
        encoded[i][int(symbol)] = 1
    return encoded
