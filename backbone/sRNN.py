# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import logging

from backbone import MammothBackbone, num_flat_features, register_backbone, xavier
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class BaseSRNN(MammothBackbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    

    def __init__(self, n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, gamma, dt, classif,keep_sparsity,sparsity,model='LIF' , w_init_gain=(0.5,0.1,0.5), lr_layer=(0.05,0.05,1.0), t_crop=100, visualize=False, visualize_light=True) -> None:    
        
        """
        Instantiates the layers of the network.

        Args:
            input_size: the size of the input data
            output_size: the size of the output
        """
        super(BaseSRNN, self).__init__()

        self.load_traces=False


        print('keep_sparsity',keep_sparsity)
        print('sparsity',sparsity)
        self.n_in     = n_in
        self.n_rec    = n_rec
        self.n_out    = n_out
        self.n_t      = n_t
        self.thr      = thr
        self.dt       = dt
        self.alpha    = np.exp(-dt/tau_m)
        self.kappa    = np.exp(-dt/tau_o)
        self.gamma    = gamma
        self.b_o      = b_o
        self.model    = model
        self.classif  = classif
        self.lr_layer = lr_layer
        self.t_crop   = t_crop  
        self.visu     = visualize
        self.visu_l   = visualize_light
        self.keep_sparsity=keep_sparsity
        
        #Parameters
        self.w_in  = nn.Parameter(torch.Tensor(n_rec, n_in ))
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))
        self.reg_term = torch.zeros(self.n_rec).to(self.device)
        self.B_out = torch.Tensor(n_out, n_rec).to(self.device)
        self.reset_parameters(w_init_gain,sparsity)
        # Precompute the convolution kernels
        alpha_conv = torch.tensor([self.alpha ** (n_t - i - 1) for i in range(n_t)],
                                       dtype=torch.float32, device=self.device).view(1, 1, -1)
        kappa_conv = torch.tensor([self.kappa ** (n_t - i - 1) for i in range(n_t)],
                                       dtype=torch.float32, device=self.device).view(1, 1, -1)
        self.register_buffer("alpha_conv", alpha_conv)
        self.register_buffer("kappa_conv", kappa_conv)
        #Visualization
        if self.visu:
            plt.ion()
            self.fig, self.ax_list = plt.subplots(2+self.n_out+5, sharex=True)

    def create_mask(self, shape, sparsity_level):
        """
        Generate a binary mask with the given sparsity level.
        sparsity_level = fraction of weights set to zero.
        """
        print('sparsity level',sparsity_level)
        return torch.tensor(
            np.random.choice([0, 1], size=shape, p=[sparsity_level, 1 - sparsity_level]),
            dtype=torch.float32
        )
    def apply_masks(self):
        
        # save masks for later use (important!)
        self.mask_in = self.mask_in.to(self.w_in.device)
        self.mask_rec = self.mask_rec.to(self.w_rec.device)
        self.mask_out = self.mask_out.to(self.w_out.device)
        with torch.no_grad():
            self.w_in *= self.mask_in
            self.w_rec *= self.mask_rec
            self.w_out *= self.mask_out


    def reset_parameters(self, gain,sparsity) -> None:
        
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0]*self.w_in.data
        mask_in = self.create_mask(self.w_in.shape, sparsity[0])
        self.w_in.data *= mask_in

        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1]*self.w_rec.data
        mask_rec = self.create_mask(self.w_rec.shape, sparsity[1])
        self.w_rec.data *= mask_rec

        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2]*self.w_out.data
        mask_out = self.create_mask(self.w_out.shape, sparsity[2])
        self.w_out.data *= mask_out

        
        # save masks for later use (important!)
        self.mask_in = self.create_mask(self.w_in.shape, sparsity[0])
        self.mask_rec = self.create_mask(self.w_rec.shape, sparsity[1])
        self.mask_out = self.create_mask(self.w_out.shape, sparsity[2])
        
        self.apply_masks()

    def init_net(self, n_b, n_t, n_in, n_rec, n_out):
        # -------------------------
        # Network state variables
        # -------------------------
        self.v  = torch.zeros(n_t, n_b, n_rec, device=self.device)
        self.vo = torch.zeros(n_t, n_b, n_out, device=self.device)
        self.z  = torch.zeros(n_t, n_b, n_rec, device=self.device)

        # -------------------------
        # Manual gradient buffers
        # (do NOT reassign if they already exist)
        # -------------------------
        if self.w_in.grad is None:
            self.w_in.grad = torch.zeros_like(self.w_in)
        else:
            self.w_in.grad.zero_()

        if self.w_rec.grad is None:
            self.w_rec.grad = torch.zeros_like(self.w_rec)
        else:
            self.w_rec.grad.zero_()

        if self.w_out.grad is None:
            self.w_out.grad = torch.zeros_like(self.w_out)
        else:
            self.w_out.grad.zero_()


    def forward(self, x,do_training=False, yt=None, returnt="out"):
        """
        x: (B, T, n_in) 
        yt: (T, B, n_out) or None
        """
        B,T,_ = x.shape
        assert T == self.n_t

        self.init_net(B, T, self.n_in, self.n_rec, self.n_out)    # Network reset
         
        # Kill self-connections
        self.w_rec.data *= (1 - torch.eye(self.n_rec, device=self.device))

        for t in range(T - 1):
            self.v[t+1] = (
                self.alpha * self.v[t]
                + self.z[t] @ self.w_rec.T
                + x[:,t,:] @ self.w_in.T
                - self.z[t] * self.thr
            )

            self.z[t+1] = (self.v[t+1] > self.thr).float()

            self.vo[t+1] = (
                self.kappa * self.vo[t]
                + self.z[t+1] @ self.w_out.T
                + self.b_o
            )

        if self.classif:
            yo = F.softmax(self.vo, dim=2)
        else:
            yo = self.vo

        # -------------------------
        # Custom learning rule
        # -------------------------
        if do_training and yt is not None:
            self.grads_batch(x, yo, yt)
            if self.keep_sparsity:
                    self.apply_masks()
        
        else:
            return yo.mean(dim=0)

        if returnt == "out":
            return yo.permute(1,2,0)
        elif returnt == "full":
            return yo, self.z, self.v
        else:
            raise NotImplementedError

    def grads_batch(self, x, yo, yt):   
        # Surrogate derivatives
        h = self.gamma*torch.max(torch.zeros_like(self.v), 1-torch.abs((self.v-self.thr)/self.thr))
        B,T,_=x.shape
        # Input and recurrent eligibility vectors for the 'LIF' model (vectorized computation, model-dependent)
        assert self.model == "LIF", "Nice try, but model " + self.model + " is not supported. ;-)"
        self.trace_in    = F.conv1d(     x.permute(0,2,1), self.alpha_conv.expand(self.n_in ,-1,-1), padding=self.n_t, groups=self.n_in )[:,:,1:self.n_t+1].unsqueeze(1).expand(-1,self.n_rec,-1,-1)  #B, n_rec, n_in , n_t 
        self.trace_in    = torch.einsum('tbr,brit->brit', h, self.trace_in )                                                                                                                          #B, n_rec, n_in , n_t 
        self.trace_rec   = F.conv1d(self.z.permute(1,2,0), self.alpha_conv.expand(self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec)[:,:, :self.n_t  ].unsqueeze(1).expand(-1,self.n_rec,-1,-1)  #B, n_rec, n_rec, n_t
        self.trace_rec   = torch.einsum('tbr,brit->brit', h, self.trace_rec)                                                                                                                          #B, n_rec, n_rec, n_t    
        self.trace_reg   = self.trace_rec

        # Output eligibility vector (vectorized computation, model-dependent)
        self.trace_out  = F.conv1d(self.z.permute(1,2,0), self.kappa_conv.expand(self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec)[:,:,1:self.n_t+1]  #B, n_rec, n_t

        # Eligibility traces
        self.trace_in     = F.conv1d(   self.trace_in.reshape(B,self.n_in *self.n_rec,self.n_t), self.kappa_conv.expand(self.n_in *self.n_rec,-1,-1), padding=self.n_t, groups=self.n_in *self.n_rec)[:,:,1:self.n_t+1].reshape(B,self.n_rec,self.n_in ,self.n_t)   #B, n_rec, n_in , n_t  
        self.trace_rec    = F.conv1d(  self.trace_rec.reshape(B,self.n_rec*self.n_rec,self.n_t), self.kappa_conv.expand(self.n_rec*self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec*self.n_rec)[:,:,1:self.n_t+1].reshape(B,self.n_rec,self.n_rec,self.n_t)   #B, n_rec, n_rec, n_t
        # yt = yt.unsqueeze(1).repeat(1, T, 1).permute(1, 0, 2)
        # Learning signals
        err = yo - yt
        L = torch.einsum('tbo,or->brt', err, self.w_out)
        
        # Update network visualization
        if self.visu:
            self.update_plot(x, self.z, yo, yt, L, self.trace_reg, self.trace_in,self.trace_rec, self.trace_out)
        
        # Compute network updates taking only the timesteps where the target is present
        if self.t_crop != 0:
            L         =          L[:,:,-self.t_crop:]
            err       =        err[-self.t_crop:,:,:]
            self.trace_in  =   self.trace_in[:,:,:,-self.t_crop:]
            self.trace_rec =  self.trace_rec[:,:,:,-self.t_crop:]
            self.trace_out =  self.trace_out[:,:,-self.t_crop:]
        
        # Weight gradient updates
        self.w_in.grad  += self.lr_layer[0]*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_in ,-1) * self.trace_in , dim=(0,3)) 
        self.w_rec.grad += self.lr_layer[1]*torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_rec,-1) * self.trace_rec, dim=(0,3))
        self.w_out.grad += self.lr_layer[2]*torch.einsum('tbo,brt->or', err, self.trace_out)
    
    def update_plot(self, x, z, yo, yt, L, trace_reg, trace_in, trace_rec, trace_out):
        """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""
    
        # Clear the axis to print new plots
        for k in range(self.ax_list.shape[0]):
            ax = self.ax_list[k]
            ax.clear()
    
        # Plot input signals
        for k, spike_ref in enumerate(zip(['In spikes','Rec spikes'],[x,z])):
            spikes = spike_ref[1][:,0,:].cpu().numpy()
            ax = self.ax_list[k]
    
            ax.imshow(spikes.T, aspect='auto', cmap='hot_r', interpolation="none")
            ax.set_xlim([0, self.n_t])
            ax.set_ylabel(spike_ref[0])
    
        for i in range(self.n_out):
            ax = self.ax_list[i + 2]
            if self.classif:
                ax.set_ylim([-0.05, 1.05])
            ax.set_ylabel('Output '+str(i))
    
            ax.plot(np.arange(self.n_t), yo[:,0,i].cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            if self.t_crop != 0:
                ax.plot(np.arange(self.n_t)[-self.t_crop:], yt[-self.t_crop:,0,i].cpu().numpy(), linestyle='solid', label='Target', alpha=0.8)
            else:
                ax.plot(np.arange(self.n_t), yt[:,0,i].cpu().numpy(), linestyle='solid' , label='Target', alpha=0.8)
    
            ax.set_xlim([0, self.n_t])
    
        for i in range(5):
            ax = self.ax_list[i + 2 + self.n_out]
            ax.set_ylabel("Trace reg" if i==0 else "Traces out" if i==1 else "Traces rec" if i==2 else "Traces in" if i==3 else "Learning sigs")
            
            if i==0:
                if self.visu_l:
                    ax.plot(np.arange(self.n_t), trace_reg[0,:,0,:].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
                else:
                    ax.plot(np.arange(self.n_t), trace_reg[0,:,:,:].reshape(self.n_rec*self.n_rec,self.n_t).T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            elif i<4:
                if self.visu_l:
                    ax.plot(np.arange(self.n_t), trace_out[0,:,:].T.cpu().numpy() if i==1 \
                                            else trace_rec[0,:,0,:].T.cpu().numpy() if i==2 \
                                            else trace_in[0,:,0,:].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
                else:
                    ax.plot(np.arange(self.n_t), trace_out[0,:,:].T.cpu().numpy() if i==1 \
                                        else trace_rec[0,:,:,:].reshape(self.n_rec*self.n_rec,self.n_t).T.cpu().numpy() if i==2 \
                                        else trace_in[0,:,:,:].reshape(self.n_rec*self.n_in,self.n_t).T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            elif self.t_crop != 0:
                ax.plot(np.arange(self.n_t)[-self.t_crop:], L[0,:,-self.t_crop:].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            else:
                ax.plot(np.arange(self.n_t), L[0,:,:].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
        
        ax.set_xlim([0, self.n_t])
        ax.set_xlabel('Time in ms')
        
        # Short wait time to draw with interactive python
        plt.draw()
        plt.pause(0.1)

    def get_eligibility(self):
        return self.trace_in,self.trace_rec,self.trace_out
    


@register_backbone("srnn_spiking")
def srnn_spiking(
    num_classes: int,
    srnn_hidden_size: int,
    n_in: int,
    n_out:int,
    n_t: int,
    thr: float,
    tau_m: float,
    tau_o: float,
    b_o: float,
    gamma: float,
    dt: float,
    classif: bool,
    keep_sparsity: bool,
    sparsity: tuple,
    ) -> BaseSRNN:

        return BaseSRNN(
            n_in=n_in,
            n_rec=srnn_hidden_size,
            n_out=n_out,
            n_t=n_t,
            thr=thr,
            tau_m=tau_m,
            tau_o=tau_o,
            b_o=b_o,
            gamma=gamma,
            dt=dt,
            classif=classif,
            keep_sparsity=keep_sparsity,
            sparsity=sparsity,
        )
