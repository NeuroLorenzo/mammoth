"""
This module contains a version of the reservoir buffer that uses a ring buffer strategy instead of reservoir.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import numpy as np
import torch
from torchvision import transforms

from utils.augmentations import apply_transform



def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size

class FeatsBuffer:
    """
    The memory buffer for features retrieval.
    """

    def __init__(self, buffer_size,p_n=0.,p_t=0., n_tasks=1, device="cpu"):
        self.buffer_size = buffer_size
        self.buffer_portion_size = buffer_size // n_tasks
        self.device = device
        self.task_number = 0
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.p_n=p_n
        self.p_t=p_t
    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

        self.filled_space = torch.zeros((self.buffer_size), dtype=torch.bool, device=self.device)  # initialize filled space

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = ring(self.num_seen_examples, self.buffer_portion_size, self.task_number)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                self.filled_space[index] = True
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms = None, device=None) -> Tuple:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple with the requested items
        """
        target_device = self.device if device is None else device
        populated_portion_length = self.filled_space.sum().item()

        if size > populated_portion_length:
            size = populated_portion_length

        choice = torch.from_numpy(np.random.choice(populated_portion_length, size=size, replace=False))
        if transform is None:
            def transform(x): return x

        # ret_tuple = (apply_transform(self.examples[choice], transform=transform).to(target_device),)
        feats = apply_transform(self.examples[choice], transform=transform).to(target_device)
        feats=self.apply_time_feature_mask(feats,self.p_t,self.p_n)
        ret_tuple=(feats,)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice].to(target_device),)
        return ret_tuple

    def apply_time_feature_mask(self,tensor: torch.Tensor, p_t: float = 0.2, p_n: float = 0.2) -> torch.Tensor:
        """
        Apply stocastic mask to the data tensor.  
        For each example in the batch, select a percentage p_n of features 
        and set to 0 a percentage p_t of their timesteps.
        """
        if p_t <= 0.0 or p_n <= 0.0:
            # print('NEGATIVE MASKING PERCENTAGES', p_t,p_n)
            return tensor

        # Applying the masking only on the returned data, not on the stored buffer data
        masked_tensor = tensor.clone()
        B, T, N_feats = masked_tensor.shape

        n_time_zero = int(T * p_t)
        n_feats_zero = int(N_feats * p_n)

        if n_time_zero == 0 or n_feats_zero == 0:
            return masked_tensor

        # Generation of independent masks for each batch
        for b in range(B):
            # 1. Random feature selections (p_n %)
            feat_idx = torch.randperm(N_feats, device=masked_tensor.device)[:n_feats_zero]

            # 2. Random timesteps selections (p_t %)
            time_idx = torch.randperm(T, device=masked_tensor.device)[:n_time_zero]

            # 3. set "spike trains" to zero
            masked_tensor[b, time_idx.unsqueeze(1), feat_idx] = 0.0

        return masked_tensor
        
    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == self.task_number == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None, device=None) -> Tuple:
        """
        Return all the items in the memory buffer.

        Args:
            transform: the transformation to be applied (data augmentation)
            device: the device to be used

        Returns:
            a tuple with all the items in the memory buffer
        """
        target_device = self.device if device is None else device
        if transform is None:
            def transform(x): return x
        ret_tuple = (apply_transform(self.examples[choice], transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr.to(target_device),)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
        self.filled_space[:] = False
