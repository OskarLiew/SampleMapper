import pickle

import torch
from torch.utils.data import Dataset


class SoundsDataset(Dataset):
    
    def __init__(self, spectrogram_path, transform=None):
        self.transform = transform
        with spectrogram_path.open("rb") as fp:
            self.sounds = pickle.load(fp)
        
    def __len__(self):
        return len(self.sounds)
    
    def __getitem__(self, idx):
        spec = self.sounds[idx].T

        if self.transform:
            spec = self.transform(spec)

        return spec


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, spec):
        # Transpose spectrogram and add an extra dimension
        return torch.from_numpy(spec)

    
class LimitLength(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, length):
        self.length = length
    
    def __call__(self, spec):
        # Pad with zeros if smaller than length
        if spec.shape[0] < self.length:
            spec_ = spec
            spec = torch.zeros(self.length, spec.shape[1])
            spec[:spec_.shape[0], :] = spec_
        else:
            spec = spec[:self.length, :]
            
        return spec
    
    
class Scale(object):
    """Normalize input to mean 0 standard deviation 1"""
    
    def __call__(self, x):
        return x / 80