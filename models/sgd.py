"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
import os
import torch

class Sgd(ContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Sgd, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.losses=[]

    def end_task(self, dataset):
        """
        This function is called at the end of each task.
        """
        stats = {
            'task_id': self.current_task,
            'losses': self.losses
        }
        self.losses=[]  
    
        
        folder='losses_resnet'
        os.makedirs(folder, exist_ok=True)
        torch.save(stats, f'{folder}/stats_task_{self.current_task}.pt')
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.losses.append(loss.item())

        self.opt.step()

        return loss.item()


