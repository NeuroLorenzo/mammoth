# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
import torch
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer

import torch.nn as nn

class Der(ContinualModel):
    """Continual learning via Dark Experience Replay."""
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        return parser
    def end_task(self, dataset):
        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(dataset.train_loader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()

    
    def get_penalty_grads(self):
        return self.args.e_lambda * 2 * self.fish * (self.net.get_params().data - self.checkpoint)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Der, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        
        self.checkpoint = None
        self.logsoft= nn.LogSoftmax(dim=1)
        self.fish = None

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        tot_loss = 0

        outputs = self.net(inputs)
        # if self.checkpoint is not None:
        #     self.net.set_grads(self.get_penalty_grads())
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += loss_mse

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
