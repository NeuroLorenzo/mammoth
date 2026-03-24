# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
import torch

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer
import matplotlib.pyplot as plt

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
        self.checkpoint=None
        self.fig, self.ax_list = plt.subplots(2, sharex=True)
        self.print_sparsity=True
    
    def end_task(self, dataset):
        if self.args.use_ewc:
            print('COMPUTING THE FISHER!')
            # Inizializziamo le Fisher per ogni layer (stessa forma dei pesi)
            self.fish = {
                'w_in': torch.zeros_like(self.net.w_in),
                'w_rec': torch.zeros_like(self.net.w_rec),
                'w_out': torch.zeros_like(self.net.w_out)
            }
            self.checkpoint = {
                'w_in': self.net.w_in.data.clone(),
                'w_rec': self.net.w_rec.data.clone(),
                'w_out': self.net.w_out.data.clone()
            }

            device = self.device
            self.net.eval() # Importante per non sporcare le tracce con dropout/batchnorm

            # Iteriamo sul dataset della task appena finita
            for j, data in enumerate(dataset.train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                _,T,_ = inputs.shape

                if self.args.classif:
                    targets=F.one_hot(labels, num_classes=self.args.n_out).float().unsqueeze(0).expand(T, -1, -1)
                else:
                    targets = labels.permute(1, 0, 2)
                

                # 2. Forward pass che calcola le tracce e l'errore (usa la tua logica e-prop)
                # Assumiamo che questa funzione popoli internamente self.w_in.grad etc.
                # ma ATTENZIONE: qui non vogliamo fare l'update dei pesi!
                start_id,end_id=self.dataset.get_offsets()
                outputs = self.net(inputs,do_training=False, yt=targets) #also implement grads_batch to compute updated weights
                Temp=5.
                # outputs=outputs/Temp
                outputs=outputs.permute(2,0,1)
                _ = self.net.grads_batch(inputs, outputs[:,:,start_id:end_id], targets[:,:,start_id:end_id],start_id,end_id,self.loss)

                # 3. Accumulo della Fisher: usiamo il quadrato del gradiente calcolato
                # Dividiamo per il numero di batch alla fine
                self.fish['w_in']  += (self.net.w_in.grad.data  / self.net.lr_layer[0])**2
                self.fish['w_rec'] += (self.net.w_rec.grad.data / self.net.lr_layer[1])**2
                self.fish['w_out'] += (self.net.w_out.grad.data / self.net.lr_layer[2])**2

                # 4. Fondamentale: azzerare i gradienti per il prossimo batch della Fisher
                self.net.w_in.grad.zero_()
                self.net.w_rec.grad.zero_()
                self.net.w_out.grad.zero_()
                
            # Media della Fisher
            for name in self.fish:
                self.fish[name] /= len(dataset.train_loader)

        if self.print_sparsity:
            ax = self.ax_list[0]
            ax.clear()
            ax.plot(self.net.fire_sparsities)
        plt.savefig(f'task_{j}_sparsity.png', dpi=300, bbox_inches='tight')
        print('COMPUTED')

        
    def get_penalty_grads_map(self):
        grads = {}
        for name in ['w_in', 'w_rec', 'w_out']:
            # Recuperiamo il peso attuale (es. self.w_in)
            weight = getattr(self.net, name)
            
            # Calcolo derivata EWC: 2 * lambda * Fisher * (θ - θ_old)
            grads[name] = self.args.e_lambda * 2 * self.fish[name] * (weight.data - self.checkpoint[name])
        return grads

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        
        # self.net.eval()
        _,T,_ = inputs.shape
        # Build targets exactly like before
        if self.args.classif:
            targets=F.one_hot(labels, num_classes=self.args.n_out).float().unsqueeze(0).expand(T, -1, -1)
        else:
            targets = labels.permute(1, 0, 2)

        # Reset optimizer state
        self.opt.zero_grad()
        do_training=True
        # Forward pass 
        outputs = self.net(inputs,do_training=do_training, yt=targets) #also implement grads_batch to compute updated weights
        outputs=outputs.permute(2,0,1)
        Temp=5.
        # outputs=outputs/Temp
        start_id,end_id=self.dataset.get_offsets()
        
        # print('training: ',self.training, self.net.training)
        # -------------------------
        # Custom learning rule
        # -------------------------
        if labels is not None:
            if self.checkpoint is not None: # Updated only after the first iteration of ewc
                penalty_grads=self.get_penalty_grads_map()
                self.net.grads_batch(inputs, outputs[:,:,start_id:end_id], targets[:,:,start_id:end_id],start_id,end_id,self.loss,penalty_grads)
            else:
                self.net.grads_batch(inputs, outputs[:,:,start_id:end_id], targets[:,:,start_id:end_id],start_id,end_id,self.loss)
                
             # Apply weight updates
            self.opt.step()
            # Compute loss only for logging
            loss = self.loss(outputs, targets) # computing the loss on the whole outputs instead of only the current ones
            return loss.item()

        
        else:
            # return self.vo[-1,:,:]
            print('output values:',self.net.vo[-1,:,:], 'labels: ',labels)
            return F.softmax(self.net.vo[-1,:,:], dim=1)
            return F.softmax(self.vo.mean(dim=0), dim=1)
            return yo.mean(dim=0)
        
