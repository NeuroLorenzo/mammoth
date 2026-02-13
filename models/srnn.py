# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
import torch

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class srnn(ContinualModel):
    """Continual learning via spiking Recurrent Neural Network"""
    NAME = 'srnn'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        print('USING PARSER: ', parser)
        # parser.add_argument('--alpha', type=float, required=True,
        #                     help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(srnn, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        
        self.net.eval()
        # Build targets exactly like before
        if self.args.classif:
            targets = (
                torch.zeros(labels.shape, device=self.device)
                .unsqueeze(-1)
                .expand(-1, -1, self.args.n_out)
                .scatter(2, labels.unsqueeze(-1), 1.0)
                .permute(1, 0, 2)
            )
        else:
            targets = labels.permute(1, 0, 2)

        # Reset optimizer state
        self.opt.zero_grad()
        # Forward pass 
        outputs = self.net(inputs,do_training=True, yt=targets) #also implement grads_batch to compute updated weights
        # Apply weight updates
        self.opt.step()
        # Compute loss only for logging
        loss = self.loss(outputs.permute(2,0,1), targets)

            
        return loss.item()
