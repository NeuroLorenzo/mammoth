"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel

from utils.args import ArgumentParser, add_rehearsal_args
import torch
import os
from line_profiler import profile
from torch.nn import functional as F
from utils.buffer import Buffer
from utils.feature_buffer import FeatsBuffer
import numpy as np

class Sgd_snn(ContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    NAME = 'sgd_snn'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        print('USING PARSER: ', parser)
        return parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Sgd_snn, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.checkpoint=None
        self.print_sparsity=False
        self.fish = None
        self.batch_id = 0
        self.last_firing_rates=[]
        self.last_labels=[]
        self.last_outputs=[]
        self.loss_terms=[]
        self.feat_buff = FeatsBuffer(self.args.buffer_size,p_n=self.args.pn_featbuff,p_t=self.args.pt_featbuff,n_tasks=5)
        self.net.eval()
        self.errors=[]
        self.losses=[]
        # 2. Congela tutti i parametri della ResNet
        for param in self.net.parameters():
            param.requires_grad = False

    @profile
    def end_task(self, dataset):
        if self.args.use_buff:
            buf_labels_l=[]
            for i in range(10):
                buf_feats, buf_targets = self.feat_buff.get_data(self.args.batch_size, device=self.device)
                if buf_targets is not None:
                    # Convert temporal one-hot back to standard class integers (0, 1, 2, 3...)
                    # If target shape is [lr_win, batch, num_classes], aggregate across time then argmax:
                    if buf_targets.ndim == 3:
                        class_ids = buf_targets[:,0,:].argmax(dim=1)
                    else:
                        class_ids = buf_targets
                    
                    buf_labels_l.append(class_ids.cpu().numpy())

            if len(buf_labels_l) > 0:
                buf_labels_l = np.concatenate(buf_labels_l)
                print("Buffer Class Distribution:", np.unique(buf_labels_l, return_counts=True))

        # if self.checkpoint is not None: # Updated only after the first iteration
        #     for i in range(10*self.feat_buff.task_number):
        #         buf_feats,buf_labels=self.feat_buff.get_data(self.args.batch_size,  device=self.device)
        #         # buf_labels=buf_labels.permute(1,0,2)
        #         outputs = self.net(buf_feats, yt=buf_labels,returnt='feats_only') #also implement grads_batch to compute updated weights
        #         outputs=outputs.permute(1,0,2)
        #         # buf_feats=buf_feats.permute(1,0,2)
        #         buf_labels=buf_labels.permute(1,0,2)
        #         self.log_data_flag= False

        #         penalty_grads=self.get_penalty_grads_map()
        #         self.net.grads_batch(buf_feats, outputs, buf_labels,start_id=None,end_id=None,loss=self.loss,penalties=penalty_grads,fr_reg=self.args.fr_reg,logit_reg=self.args.logit_reg,self.log_data_flag=self.log_data_flag)
                    
        #             # Apply weight updates
        #         self.opt.step()

            self.feat_buff.task_number+=1
            self.net.first_task=False
        
        

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
                
                # _,T,_ = inputs.shape
                T=self.args.n_t
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
                    
                    outputs,sfeats = self.net(ex, yt=targets,returnt = "spikes_feats") #also implement grads_batch to compute updated weights
                    # outputs,sfeats = self.net(ex, yt=targets) #also implement grads_batch to compute updated weights
                    
                    avg_outputs = torch.mean(outputs, dim=2) 
                    # outputs is the softmax of the logits, convert it to logsoftmax
                    log_probs = torch.log(avg_outputs + 1e-12)

                    # compute Negative Log Likelihood Loss
                    loss = -F.nll_loss(log_probs, lab.view(-1))
                    exp_cond_prob = torch.mean(torch.exp(loss.detach().clone())) # $$ mean non utile poichè è il singolo esempio
                    loss = torch.mean(loss) # $$ mean non utile poichè è il singolo esempio

                    outputs=outputs.permute(2,0,1)
                    _ = self.net.grads_batch(sfeats, outputs, targets[-self.args.lr_win:,:,:],start_id=None,end_id=None,loss=self.loss,logit_reg=self.args.logit_reg)



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
                'grads': self.loss_terms,
                'outputs': self.last_outputs,
                'errors': self.errors,
                'losses': self.losses
            }
            self.losses=[]  
            self.errors=[]
            self.last_firing_rates=[]
            self.last_labels=[]
            self.last_outputs=[]
            self.loss_terms=[]
            folder='debug_data_metapl_su'
            os.makedirs(folder, exist_ok=True)
            torch.save(stats, f'{folder}/stats_task_{self.current_task}.pt')

        
        
        else:
            self.checkpoint = {
                            'w_in': self.net.w_in.data.clone(),
                            'w_rec': self.net.w_rec.data.clone(),
                            'w_out': self.net.w_out.data.clone()
                        }
        self.batch_id=0
            
        # if self.args.use_metapl:
            
        #     if self.args.filter_most_active:
        #         index_sorted=self.net.z_mean_acc.argsort()
        #         most_active_ids=index_sorted[-self.args.n_active_filter:]
        #         print(f'10 most firing neurons: {most_active_ids} with rates: {self.net.z_mean_acc[most_active_ids]}')
        #         self.net.z_mean_acc=None
        #         if self.current_task>8:
        #             self.net.mask_z=torch.zeros(self.net.n_rec, dtype=torch.bool)
        #             self.net.mask_z[most_active_ids]=True
        #     checkpoint = {
        #         'w_in': self.net.w_in.data.clone(),
        #         'w_rec': self.net.w_rec.data.clone(),
        #         'w_out': self.net.w_out.data.clone()
        #     }
        #     stats = {
        #         'task_id': self.current_task,
        #         'weights': {k: v.cpu() for k, v in checkpoint.items()},
        #         'firing_rates': self.last_firing_rates, # Assicurati di salvarli in observe
        #         'labels': self.last_labels,
        #         'losses': self.loss_terms,
        #         'outputs': self.last_outputs
        #     }
        #     self.last_firing_rates=[]
        #     self.last_labels=[]
        #     self.last_outputs=[]
        #     self.loss_terms=[]
        #     folder='debug_data_metapl_fullds'
        #     os.makedirs(folder, exist_ok=True)
        #     torch.save(stats, f'{folder}/stats_task_{self.current_task}.pt')


        if self.args.weight_align and (self.current_task > 0): # Non farlo per la prima task
            with torch.no_grad():

                # 1. Compute the norm of the weights of previous tasks (:start_id)
                start_id,end_id=self.dataset.get_offsets()
                print('start_id ', start_id, ', end_id', end_id)
                old_weights = self.net.w_out[:start_id, :]
                norm_old = torch.norm(old_weights, p=2, dim=1).mean()
                
                # 2. Compute the norm of the weights of the current task 
                new_weights = self.net.w_out[start_id:end_id, :]
                norm_new = torch.norm(new_weights, p=2, dim=1).mean()
                
                # 3. Computing scaling factor
                gamma = norm_old / norm_new
                
                # 4. balance the weights of the current task
                self.net.w_out[start_id:end_id, :] *= gamma
                
                print(f"--- Weight Aligning Applicato ---")
                print(f"Norma Vecchia: {norm_old:.4f}, Norma Nuova: {norm_new:.4f}, Gamma: {gamma:.4f}")
        if len(self.last_outputs)>0:
            stats = {
                    'task_id': self.current_task,
                    'weights': {k: v.cpu() for k, v in self.checkpoint.items()},
                    'firing_rates': self.last_firing_rates, # Assicurati di salvarli in observe
                    'labels': self.last_labels,
                    'grads': self.loss_terms,
                    'outputs': self.last_outputs,
                    'errors': self.errors,
                    'losses': self.losses
            }
            self.losses=[]  
            self.errors=[]
            self.last_firing_rates=[]
            self.last_labels=[]
            self.last_outputs=[]
            self.loss_terms=[]
            
            folder='debug_data_nologits'
            os.makedirs(folder, exist_ok=True)
            torch.save(stats, f'{folder}/stats_task_{self.current_task}.pt')
        
    @profile
    def get_penalty_grads_map(self):
        grads = {}
        for name in ['w_in', 'w_rec', 'w_out']:
            # get weights
            weight = getattr(self.net, name)
            
            # EWC derivative: 2 * lambda * Fisher * (θ - θ_old)
            grads[name] = self.args.e_lambda * 2 * self.fish[name] * (weight.data - self.checkpoint[name])
        return grads
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        if self.args.classif:
            targets=F.one_hot(labels, num_classes=self.args.n_out).float().unsqueeze(0).expand(self.args.lr_win, -1, -1)
            
        else:
            targets = labels.permute(1, 0, 2)
        self.opt.zero_grad()
        # outputs = self.net(inputs)
        # sup_inputs, sup_labels = self.sup_virtual_batch.get_data(self.args.batch_size, transform=self.transform, device=self.device)

        outputs,sfeats = self.net(inputs, yt=targets,lr_win=self.args.lr_win,returnt = "spikes_feats") #also implement grads_batch to compute updated weights
        outputs=outputs.permute(2,0,1)

        if self.args.use_buff:
            ids_buff=np.random.randint(sfeats.shape[0],size=4) #pick 4 examples per batch (25%)

            self.feat_buff.add_data(sfeats[ids_buff], targets.permute(1,0,2)[ids_buff])

        start_id,end_id=self.dataset.get_offsets()

        # Custom learning rule
        if labels is not None:
            self.log_data_flag= ((self.batch_id%20) ==0) #save data to study the network every 20 batches
            # self.log_data_flag= False
            if self.fish is not None: # Updated only after the first iteration of ewc
                penalty_grads=self.get_penalty_grads_map()
                self.net.grads_batch(sfeats, outputs, targets,start_id=start_id,end_id=end_id,loss=self.loss,penalties=penalty_grads,fr_reg=self.args.fr_reg,logit_reg=self.args.logit_reg,log_data_flag=self.log_data_flag)
            else: 
                err=self.net.grads_batch(sfeats, outputs, targets,start_id=start_id,end_id=end_id,loss=self.loss,use_metapl=self.args.use_metapl,fr_reg=self.args.fr_reg,logit_reg=self.args.logit_reg,log_data_flag=self.log_data_flag)


            self.errors.append(err)
            # Apply weight updates
            self.opt.step()
            
            # Compute loss only for logging
            loss = self.loss(outputs, targets) # computing the loss on the whole outputs instead of only the current ones
            self.losses.append(loss.item())
            

            
            if self.checkpoint is not None and self.args.use_buff: # Updated only after the first iteration of ewc
                self.opt.zero_grad()
                
                # using previous data
                buf_feats,buf_labels=self.feat_buff.get_data(4,  device=self.device) # 4 works fine
                # buf_feats,buf_labels=self.feat_buff.get_data(int(12-end_id),  device=self.device) # 4 works fine
                outputs = self.net(buf_feats, yt=buf_labels,returnt='feats_only') #also implement grads_batch to compute updated weights
                outputs=outputs.permute(1,0,2)
                buf_labels=buf_labels.permute(1,0,2)
                if self.checkpoint is not None: # Updated only after the first iteration of ewc
                    self.net.grads_batch(buf_feats, outputs, buf_labels,start_id=0,end_id=end_id,loss=self.loss,use_metapl=self.args.use_metapl,fr_reg=self.args.fr_reg,logit_reg=self.args.logit_reg,log_data_flag=self.log_data_flag)            
                    # self.net.grads_batch(buf_feats, outputs, buf_labels,start_id=None,end_id=None,loss=self.loss,use_metapl=self.args.use_metapl,fr_reg=self.args.fr_reg,logit_reg=self.args.logit_reg,log_data_flag=self.log_data_flag)            
                self.opt.step()
            # self.opt.step()
            
            
            self.batch_id +=1
            if self.log_data_flag:
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
        
        loss = self.loss(outputs, labels)
        # loss.backward()
        self.opt.step()
        print('loss',loss)
        return loss.item()
