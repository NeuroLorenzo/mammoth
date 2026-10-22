import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.validation import get_train_val
from utils.conf import base_path


class MyCIFAR10_feats(Dataset):
    """
    Dataset PyTorch custom per feature 512-d.
    Restituisce direttamente oggetti torch.Tensor (Float32 e Long) 
    evitando qualsiasi conversione o dipendenza da PIL/torchvision.
    """
    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None, download: bool = False):
        self.root = root
        self.train = train
        self.transform = None
        self.target_transform = target_transform

        filename = "train_feats.npz" if train else "test_feats.npz"
        file_path = os.path.join(root, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")

        loaded = np.load(file_path)
        
        # Manteniamo array NumPy float32 e int64
        self.data = loaded["features"].astype(np.float32)
        self.targets = loaded["labels"].astype(np.int64)

    def _check_integrity(self):
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = torch.from_numpy(self.data[index])
        target = int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Restituisce la tupla a 3 elementi (inputs, labels, not_aug_inputs) richiesta da Mammoth
        return img, target, img


class SequentialCIFAR10Feats(ContinualDataset):
    """
    ContinualDataset wrapper per Mammoth su feature 512-d.
    """
    NAME = 'seq-cifar10-feats'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = 10
    
    TRANSFORM = None
    TEST_TRANSFORM = None

    def get_data_loaders(self):
        # Gestione sicura del parametro num_workers se e' None
        if getattr(self.args, 'num_workers', None) is None:
            self.args.num_workers = 4

        data_folder = os.path.join(base_path(), 'cifar10_feats')
        
        train_dataset = MyCIFAR10_feats(data_folder, train=True)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(
                train_dataset, None, self.NAME
            )
        else:
            test_dataset = MyCIFAR10_feats(data_folder, train=False)
        # test_dataset = MyCIFAR10_feats(data_folder, train=False)

        # store_masked_loaders applica correttamente il mascheramento per i 5 task Class-IL
        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)
        return train_loader, test_loader

    @staticmethod
    def get_loss():
        return nn.CrossEntropyLoss()

    @staticmethod
    def get_backbone():
        return nn.Identity()

    @staticmethod
    def get_transform():
        return lambda x: x

    @staticmethod
    def get_test_transform():
        return lambda x: x
    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None