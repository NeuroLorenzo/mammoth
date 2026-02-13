# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST,VisionDataset

from datasets.transforms.permutation import Permutation
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from utils.conf import base_path
from datasets.utils import set_default_from_args

class NMNIST(VisionDataset):
    """
    Overrides the MNIST dataset to change the getitem function.
    """

    def __init__(self, root, transform=None, train=True,
                 target_transform=None, download=False) -> None:
        # Call parent constructor
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        
        pixels_blocklist = np.loadtxt(root+"/../NMNIST_pixels_blocklist.txt")

        self.pixels_dict = {
            "n_x": 34,  # number of pixels in horizontal direction
            "n_y": 34,  # number of pixels in vertical direction
            "n_polarity": 2,  # number of pixels in the dimension coding for polarity
        }

        self.pixels_dict["n_total"] = self.pixels_dict["n_x"] * self.pixels_dict["n_y"] * self.pixels_dict["n_polarity"]  # total number of pixels
        self.pixels_dict["active"] = sorted(set(range(self.pixels_dict["n_total"])) - set(pixels_blocklist))  # active pixels
        self.pixels_dict["n_active"] = len(self.pixels_dict["active"])  # number of active pixels


        print(root)
        print(type(root))
        self.root = root
        self.train = train
        self.target_transform = target_transform
        self.download = download
        self.transform = transform
        self.n_out=10 # number of targets
        self.dt=0.1 # size of integration timestep
        self.seq_len=30 # number of integration timesteps ($check if it's time o #timesteps=time/dt) 
        # self.seq_len=300
        self.sample_paths, self.targets = self._get_all_sample_paths_with_labels()
        self.active_indices = {p: i for i, p in enumerate(self.pixels_dict["active"])}

        self.data = self._load_data()
        self.data = np.stack(self._load_data(), axis=0)
        # self.targets = np.array(self.targets)
        self.indexes = np.arange(len(self.targets))
    
    def _load_data(self):
                return [self.load_image(path, self.pixels_dict) for path in self.sample_paths]
    
    def _get_all_sample_paths_with_labels(self):
        sample_paths = []
        targets = []

        for trg in range(10):
            label_dir = os.path.join(self.root, str(trg))
            all_files = sorted(os.listdir(label_dir))

            for sample in all_files:
                sample_paths.append(os.path.join(label_dir, sample))
                targets.append(trg)

        return sample_paths, targets
    
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()

        img, target = self.data[index], self.targets[index]
        # target = np.tile(self.targets[index],self.seq_len)

        # img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float()    # tensor (T, n_in)


        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = torch.tensor(target).long()   # scalar class label

        return img, target, img
    
    def load_image(self,file_path, pixels_dict, n_time_bins=300):
        with open(file_path, "rb") as f:
            byte_array = np.asarray([x for x in f.read()])
        #   load DVS like events of the current image. Each event has 5 items: x,y,p+t1,t2,t3 each 8 bit 

        n_byte_columns = 5
        byte_columns = [byte_array[column::n_byte_columns] for column in range(n_byte_columns)]

        x_coords = byte_columns[0]
        y_coords = byte_columns[1]

        #   third byte of the event contains both time and polarity info. 
        #   polarity is stored in the most significant bit (7)
        polarities = byte_columns[2] >> 7 # 7 shifts to the left of a byte keep only a single binary bit  
        mask_22_bit = 0x7FFFFF 
        #   t1 contains the high bits ot the timestamp
        #   t2 the middle ones
        #   t3 the lowest 8 bits of the timestamp
        #   [t1|t2|t3] is a 24 bit timestamp, where the most significant bit is still the polarity
        times = (byte_columns[2] << 16 | byte_columns[3] << 8 | byte_columns[4]) & mask_22_bit
        time_max = 336040

        # rescale continuous time to present the whole registration in the seq_len video (be careful of saturation! $$)
        # this is a binning procudere (type=int)
        times = np.around(times * self.seq_len / time_max).astype(int)

        pixels = polarities * pixels_dict["n_x"] * pixels_dict["n_y"] + y_coords * pixels_dict["n_x"] + x_coords
      

        # Convert spike times to binary tensor
        n_active = len(pixels_dict["active"])
        tensor = np.zeros((self.seq_len, n_active), dtype=np.float32) # shape of seq_len, n_active
        # tensor = np.zeros((n_time_bins, n_active), dtype=np.float32) # shape of n_time_bins (!=seq_len?), n_active


        
        for p, t in zip(pixels, times):
            # if p in active_indices and 0 <= t < n_time_bins:
            if p in self.active_indices and 0 <= t < self.seq_len:
                tensor[t, self.active_indices[p]] = 1.0

        return tensor
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
    
    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    


class NeuromorphicMNIST(ContinualDataset):
    """Neuromorphic MNIST Dataset.

    Creates a dataset composed by a sequence of tasks, each containing a
    different neuromorphic video of the MNIST dataset.

    Args:
        NAME (str): name of the dataset
        SETTING (str): setting of the experiment
        N_CLASSES_PER_TASK (int): number of classes in each task
        N_TASKS (int): number of tasks
        SIZE (tuple): size of the images
    """

    NAME = 'neuro-mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    SIZE = (30, 1156)
    def __init__(self, args):
        super().__init__(args)
        self.train_path=base_path() + 'NMNIST/Train' #args.train_path # $$ I need to find a way to adjust this!
        self.test_path=base_path() + 'NMNIST/Test'
        # self.pixels_dict=args.pixels_dict

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        # transform = transforms.Compose((transforms.ToTensor(), Permutation(np.prod(NeuromorphicMNIST.SIZE))))
        
        
        train_dataset = NMNIST(str(self.train_path), train=True, download=False, transform=None)
        # train_dataset = NMNIST(str(self.train_path), train=True, download=True, transform=transforms.ToTensor())
        test_dataset = NMNIST(str(self.test_path), train=False, download=False, transform=None)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "srnn_spiking"

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
        return F.binary_cross_entropy

    @set_default_from_args('batch_size')
    def get_batch_size(self) -> int:
        return 5

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
