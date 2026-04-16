
from torch.nn import functional as F
import torch
import os
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer
import matplotlib.pyplot as plt

import logging
from typing import Iterator
import torch.optim as optim

from line_profiler import profile

from utils import optimizers_utils
class srnn(ContinualModel):
    """Continual learning via spiking Recurrent Neural Network"""
    NAME = 'srnn'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    AVAIL_OPTIMS = ['sgd', 'adam', 'adamw','adam_meta','sgd_meta']
    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        print('USING PARSER: ', parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(srnn, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.checkpoint=None
        # self.fig, self.ax_list = plt.subplots(2, sharex=True)
        self.print_sparsity=False
        self.fish = None
        self.batch_id = 0
        self.last_firing_rates=[]
        self.last_labels=[]
        self.last_outputs=[]
        self.loss_terms=[]

    
    def get_optimizer(self, params: Iterator[torch.Tensor] = None, lr=None) -> optim.Optimizer:
        """
        Returns the optimizer to be used for training.

        Default: SGD.

        Args:
            params: the parameters to be optimized. If None, the default specified by `get_parameters` is used.
            lr: the learning rate. If None, the default specified by the command line arguments is used.

        Returns:
            the optimizer
        """
        meta=optimizers_utils.meta_generate(self.args,self.net)
        params = params if params is not None else self.get_parameters()
        if params is None:
            logging.info("No parameters to optimize.")
            return None
        params = list(params)
        if len(params) == 0:
            logging.info("No parameters to optimize.")
            return None

        lr = lr if lr is not None else self.args.lr
        
        all_supported_optims = {}
        # check if optimizer is in torch.optim or in custom optimizers
        for optim_name in dir(optim):
            if optim_name.lower() in self.AVAIL_OPTIMS:
                
                all_supported_optims[optim_name.lower()] = optim_name
        for o in dir(optimizers_utils):
            if not o.startswith("__"):
                all_supported_optims[o.lower()] = getattr(optimizers_utils, o)
        # supported_optims = {optim_name.lower(): optim_name for optim_name in dir(optim) if optim_name.lower() in self.AVAIL_OPTIMS}
        opt = None
        if self.args.optimizer.lower() in all_supported_optims:
            if self.args.optimizer.lower() == 'sgd':
                opt = getattr(optim, all_supported_optims[self.args.optimizer.lower()])(params, lr=lr,
                                                                                    weight_decay=self.args.optim_wd,
                                                                                    momentum=self.args.optim_mom,
                                                                                    nesterov=self.args.optim_nesterov)
            elif self.args.optimizer.lower() == 'adam' or self.args.optimizer.lower() == 'adamw':
                opt = getattr(optim, all_supported_optims[self.args.optimizer.lower()])(params, lr=lr,
                                                                                    weight_decay=self.args.optim_wd,)
                
            elif self.args.optimizer.lower() == 'sgd_meta':
                opt = all_supported_optims[self.args.optimizer.lower()](params, lr=lr,
                                                                                    meta_func=self.args.meta_func,
                                                                                    # weight_decay=self.args.optim_wd,
                                                                                    # momentum=self.args.optim_mom,
                                                                                    # nesterov=self.args.optim_nesterov,
                                                                                    meta=meta)
            elif self.args.optimizer.lower() == 'adam_meta':
                opt = all_supported_optims[self.args.optimizer.lower()](params, lr=lr, 
                                             meta_func=self.args.meta_func,
                                            #  meta_params=self.meta_dict,
                                             meta=meta)

        if opt is None:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
        return opt
    
        
    @profile
    def end_task(self, dataset):
        if self.args.use_ewc:
            
            # fisher layer wise initialization
            fish = {
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
                for ex, lab in zip(inputs, labels):
                    self.opt.zero_grad()
                    ex = ex.unsqueeze(0)
                    if self.args.classif:
                        targets=F.one_hot(lab, num_classes=self.args.n_out).float().unsqueeze(0).expand(T, -1, -1)
                        if self.args.lr_win is None or self.args.lr_win <= 0:
                            self.args.lr_win = self.args.seq_len_train
                        if self.args.lr_win < self.args.seq_len_train:
                            targets[:(self.args.seq_len_train - self.args.lr_win), :, :] = 0.0
                    else:
                        targets = lab.permute(1, 0, 2)
                    
                    outputs = self.net(ex,do_training=False, yt=targets) #also implement grads_batch to compute updated weights
                    avg_outputs = torch.mean(outputs, dim=2) 
                    # outputs is the softmax of the logits, convert it to logsoftmax
                    log_probs = torch.log(avg_outputs + 1e-12)

                    # compute Negative Log Likelihood Loss
                    loss = -F.nll_loss(log_probs, lab.view(-1))
                    exp_cond_prob = torch.mean(torch.exp(loss.detach().clone())) # $$ mean non utile poichè è il singolo esempio
                    loss = torch.mean(loss) # $$ mean non utile poichè è il singolo esempio

                    outputs=outputs.permute(2,0,1)
                    _ = self.net.grads_batch(ex, outputs, targets[-self.args.lr_win:,:,:],start_id=None,end_id=None,loss=self.loss,logit_reg=self.args.logit_reg)



                    # Scale gradients by lr_layer which is manually applied in grads_batch
                    fish['w_in']  += exp_cond_prob*(self.net.w_in.grad.data  / self.net.lr_layer[0])**2
                    fish['w_rec'] += exp_cond_prob*(self.net.w_rec.grad.data / self.net.lr_layer[1])**2
                    fish['w_out'] += exp_cond_prob*(self.net.w_out.grad.data / self.net.lr_layer[2])**2

                    # Reset gradients
                    self.net.w_in.grad.zero_()
                    self.net.w_rec.grad.zero_()
                    self.net.w_out.grad.zero_()

            # Fish average
            for name in fish:
                fish[name] /= (len(dataset.train_loader)* self.args.batch_size)

            if self.fish is None:
                self.fish = fish
            else:
                for name in fish:
                    self.fish[name] *= self.args.gamma
                    self.fish[name]  += fish[name] 

        
            
            stats = {
                'task_id': self.current_task,
                'fisher': {k: v.cpu() for k, v in self.fish.items()},
                'weights': {k: v.cpu() for k, v in self.checkpoint.items()},
                'firing_rates': self.last_firing_rates, # Assicurati di salvarli in observe
                'labels': self.last_labels,
                'losses': self.loss_terms,
                'outputs': self.last_outputs
            }
            self.last_firing_rates=[]
            self.last_labels=[]
            self.last_outputs=[]
            self.loss_terms=[]
            os.makedirs('debug_data', exist_ok=True)
            torch.save(stats, f'debug_data/stats_task_{self.current_task}.pt')

        
        
        
        self.batch_id=0
            
        if self.args.use_metapl:
            checkpoint = {
                'w_in': self.net.w_in.data.clone(),
                'w_rec': self.net.w_rec.data.clone(),
                'w_out': self.net.w_out.data.clone()
            }
            stats = {
                'task_id': self.current_task,
                'weights': {k: v.cpu() for k, v in checkpoint.items()},
                'firing_rates': self.last_firing_rates, # Assicurati di salvarli in observe
                'labels': self.last_labels,
                'losses': self.loss_terms,
                'outputs': self.last_outputs
            }
            self.last_firing_rates=[]
            self.last_labels=[]
            self.last_outputs=[]
            self.loss_terms=[]
            os.makedirs('debug_data_metapl', exist_ok=True)
            torch.save(stats, f'debug_data_metapl/stats_task_{self.current_task}.pt')
    @profile
    def get_penalty_grads_map(self):
        grads = {}
        for name in ['w_in', 'w_rec', 'w_out']:
            # get weights
            weight = getattr(self.net, name)
            
            # EWC derivative: 2 * lambda * Fisher * (θ - θ_old)
            grads[name] = self.args.e_lambda * 2 * self.fish[name] * (weight.data - self.checkpoint[name])
        return grads

    @profile
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        
        
        _,T,_ = inputs.shape
        # Build one-hot encoded targets 
        if self.args.classif:
            targets=F.one_hot(labels, num_classes=self.args.n_out).float().unsqueeze(0).expand(self.args.lr_win, -1, -1)
            
        else:
            targets = labels.permute(1, 0, 2)

        # Reset optimizer state
        self.opt.zero_grad()
        do_training=True
        # Forward pass 
        outputs = self.net(inputs,do_training=do_training, yt=targets,lr_win=self.args.lr_win) #also implement grads_batch to compute updated weights
        outputs=outputs.permute(2,0,1)
        
        
        # Custom learning rule
        if labels is not None:
            log_data_flag= ((self.batch_id%20) ==0) #save data to study the network every 20 batches
            if self.checkpoint is not None: # Updated only after the first iteration of ewc
                penalty_grads=self.get_penalty_grads_map()
                self.net.grads_batch(inputs, outputs, targets,start_id=None,end_id=None,loss=self.loss,penalties=penalty_grads,logit_reg=self.args.logit_reg,log_data_flag=log_data_flag)
            else: 
                self.net.grads_batch(inputs, outputs, targets,start_id=None,end_id=None,loss=self.loss,use_metapl=self.args.use_metapl,logit_reg=self.args.logit_reg,log_data_flag=log_data_flag)
                
             # Apply weight updates
            self.opt.step()
            
            # Compute loss only for logging
            loss = self.loss(outputs, targets) # computing the loss on the whole outputs instead of only the current ones
                
            if self.args.use_buffer and not self.buffer.is_empty():
                buf_inputs, buf_labels,_ = self.buffer.get_data(
                    self.args.minibatch_size, transform=None, device=self.device)
                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_labels)
                loss += loss_mse
            
                if self.checkpoint is not None: # Updated only after the first iteration of ewc
                    penalty_grads=self.get_penalty_grads_map()
                    self.net.grads_batch(buf_inputs, buf_outputs, buf_labels,start_id=None,end_id=None,loss=self.loss,penalties=penalty_grads,fr_reg=self.args.fr_reg,logit_reg=self.args.logit_reg,log_data_flag=log_data_flag)
                else:
                    self.net.grads_batch(buf_inputs, buf_outputs, buf_labels,start_id=None,end_id=None,loss=self.loss,fr_reg=self.args.fr_reg,logit_reg=self.args.logit_reg,log_data_flag=log_data_flag)
            
            self.batch_id +=1
            if log_data_flag:
                self.last_firing_rates.append(torch.mean(self.net.z, dim=(0)).cpu()) # mean in time, fr of shape [n_b,n_rec]
                self.last_labels.append(labels.cpu())

                final_logits = self.net.vo[-self.net.lr_win:,:,:] 
                avg_logits = torch.mean(final_logits, dim=0)
                self.last_outputs.append(avg_logits.cpu())
                self.loss_terms.append(self.net.batch_logs)
            return loss.item()


        
        else:
            return F.softmax(self.net.vo[-1,:,:], dim=1)
            return F.softmax(self.vo.mean(dim=0), dim=1)
            return yo.mean(dim=0)
        
