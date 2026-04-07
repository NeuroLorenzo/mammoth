# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST

from datasets.transforms.permutation import Permutation
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from utils.conf import base_path
from datasets.utils import set_default_from_args

class PermLatMNIST(MNIST):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, args=None) -> None:
        super(PermLatMNIST, self).__init__(root, train=True if train else False, 
                                           transform=transform,
                                           target_transform=target_transform, 
                                           download=download)
        # Recuperiamo T dagli argomenti (es. 100 o 300)
        self.T = getattr(args, 'n_t', 100) if args else 100

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        # 1. Recupero dati originali
        img, target = self.data[index], int(self.targets[index])

        # 2. Trasformazione in PIL e eventuale Normalizzazione/Augmentation
        img_pil = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img_tensor = self.transform(img_pil) # Risultato tipico: [1, 28, 28] in [0, 1]
        else:
            img_tensor = img.float() / 255.0
        print('img_tensor:',img_tensor)
        # Appiattiamo l'immagine: [1, 28, 28] -> [784]
        flat_img = img_tensor.view(-1)

        # 3. LATENCY ENCODING
        # Idea: tempo di spike t_i = T * (1 - pixel_intensity)
        # Un pixel a 1.0 (bianco) -> t = 0
        # Un pixel a 0.1 (grigio scuro) -> t = T * 0.9
        
        # Creiamo il tensore di output: [T, 784] inizializzato a zero
        latency_img = torch.zeros((self.T, flat_img.size(0)))

        # Calcoliamo il bin temporale per ogni pixel
        # Usiamo clamp per evitare indici fuori limite se l'intensità è esattamente 0
        spike_times = ((1.0 - flat_img) * (self.T - 1)).long()
        
        # Mettiamo a 1 (spike) solo i pixel che hanno un'intensità minima (es. > 0.01)
        # per evitare che il "nero" faccia scattare spike all'ultimo step
        mask = flat_img > 0.01 
        
        # Popoliamo il tensore: per ogni pixel (dim 1), mettiamo 1 al tempo calcolato (dim 0)
        # Usiamo scatter_ per efficienza
        indices = spike_times[mask].unsqueeze(0) # [1, num_spikes]
        latency_img.scatter_(0, indices, 1.0)
        print('img',latency_img.shape)
        print('target',target.shape)
        # Restituiamo (Input Temporale, Target, Immagine Originale per DER/Buffer)
        return latency_img, target, img_tensor

class PermutedLatMNIST(ContinualDataset):
    """Permuted MNIST Dataset.

    Creates a dataset composed by a sequence of tasks, each containing a
    different permutation of the pixels of the MNIST dataset.

    Args:
        NAME (str): name of the dataset
        SETTING (str): setting of the experiment
        N_CLASSES_PER_TASK (int): number of classes in each task
        N_TASKS (int): number of tasks
        SIZE (tuple): size of the images
    """

    NAME = 'perm-lat-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20
    SIZE = (28, 28)
    def __init__(self, args):
        super().__init__(args)
        self.args=args
    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = transforms.Compose((transforms.ToTensor(), Permutation(np.prod(PermutedLatMNIST.SIZE))))

        train_dataset = PermLatMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform,args=self.args)
        test_dataset = PermLatMNIST(base_path() + 'MNIST',
                             train=False, download=True, transform=transform,args=self.args)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"
        # return "mnistmlp"

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @set_default_from_args('batch_size')
    def get_batch_size(self) -> int:
        return 128

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 1

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = MNIST(base_path() + 'MNIST', train=True, download=True).classes
        classes = [c.split('-')[1].strip() for c in classes]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
