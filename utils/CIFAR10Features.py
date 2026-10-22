import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CIFAR10Features(Dataset):
    """
    Dataset custom per caricare le feature 512-d pre-estratte dai file .npz
    al posto delle immagini originali di CIFAR-10.
    """
    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Scegli il file corretto in base allo split
        filename = "train_feats.npz" if train else "test_feats.npz"
        file_path = os.path.join(root, "cifar10_feats", filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {file_path} non trovato! Assicurati di aver eseguito save_feats.py"
            )

        # Carica il file .npz
        data = np.load(file_path)
        
        # Converti le feature in Tensor Float32 e le etichette in LongTensor
        self.data = torch.from_numpy(data["features"]).float()  # (N, 512)
        self.targets = torch.from_numpy(data["labels"]).long()   # (N,)

    def __getitem__(self, index):
        feature, target = self.data[index], self.targets[index]

        if self.transform is not None:
            feature = self.transform(feature)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feature, target

    def __len__(self):
        return len(self.data)