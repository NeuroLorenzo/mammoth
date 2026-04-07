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

    

    def __init__(self, n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, gamma, dt, classif,keep_sparsity,sparsity,lr_layer,f_target,c_reg,lr_win,xi,model='LIF' , w_init_gain=(0.5,0.1,0.5), t_crop=100, visualize=False, visualize_light=True) -> None:    
        
        """
        Instantiates the layers of the network.

        Args:
            input_size: the size of the input data
            output_size: the size of the output
        """
        super(BaseSRNN, self).__init__()

        self.load_traces=False


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
        # self.lr_layer = (0.005,0.005,0.005)
        # self.lr_layer = lr_layer
        self.t_crop   = t_crop  
        self.visu     = visualize
        self.visu_count    = 0
        self.visu_l   = visualize_light
        self.keep_sparsity=keep_sparsity
        self.fire_sparsities=[]
        self.f_target=f_target
        self.c_reg=c_reg
        self.lr_win=lr_win
        
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

        # Nel tuo __init__
        self.xi = xi  # Forza della metaplasticità
        self.u_decay = 0.99  # Coeff di persistenza della memoria sinaptica

        self.u_in = torch.zeros_like(self.w_in).to(self.device)
        self.u_rec = torch.zeros_like(self.w_rec).to(self.device)
        self.u_out = torch.zeros_like(self.w_out).to(self.device)

    def create_mask(self, shape, sparsity_level):
        """
        Generate a binary mask with the given sparsity level.
        sparsity_level = fraction of weights set to zero.
        """
        # print('sparsity level',sparsity_level)
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

        

    def forward(self, x,do_training=False, yt=None, returnt="out",lr_win=0):
        """
        x: (B, T, n_in) 
        yt: (T, B, n_out) or None
        """
        if self.keep_sparsity:
            self.apply_masks()
        in_shape=x.shape
        if len(in_shape)==2:
            x = x.unsqueeze(0)
        B,T,N = x.shape
        assert T >0, f"Temporal Consistency Error: data len T ({T}) different from the simulation times n_t ({self.n_t})"
        # assert T == self.n_t, f"Temporal Consistency Error: data len T ({T}) different from the simulation times n_t ({self.n_t})"

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
        fire_sparsity=torch.count_nonzero(self.z)/(T*B*N)
        self.fire_sparsities.append(100 -100*fire_sparsity.cpu())
        
        
        if self.classif:
            yo = F.softmax(self.vo[-self.lr_win:,:,:], dim=2)
        else:
            yo = self.vo

        if yt is None:
            # print('output values:',self.vo[T-1,:,:]) 
            final_logits = self.vo[-self.lr_win:,:,:] 
        
            # Facciamo la media lungo la dimensione temporale (0)
            # Result shape: [Batch, n_out]
            avg_logits = torch.mean(final_logits, dim=0)
            return F.softmax(avg_logits, dim=1)
            return F.softmax(self.vo[T-1,:,:], dim=1)
            return F.softmax(self.vo.mean(dim=0), dim=1)
            return yo.mean(dim=0)

        
        
        if returnt == "out":
            # return yo
            return yo.permute(1,2,0)
        elif returnt == "full":
            return yo, self.z, self.v
        else:
            raise NotImplementedError

    def grads_batch(self, x, yo, yt,start_id,end_id,loss=None,penalties=None,fr_reg=False,use_metapl=False,logit_reg=False):   
        with torch.no_grad():
            
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
            T=5
            err = yo - yt
            # if loss:
                # err_ce = loss(yo,yt)
                # print('yo-yt: ',err,'; CE: ',err_ce)
            # err=err[:,:,start_id:end_id]
            
            if start_id is not None:
                L = torch.einsum('tbo,or->brt', err, self.w_out[start_id:end_id,:])
            else:
                L = torch.einsum('tbo,or->brt', err, self.w_out)
                

            # Update network visualization
            self.visu_count+=1
            if self.visu_count%100000 ==0 :
                print('uuuuuuuuuu save!---------------------------------------')
                visual_data = {
                        "input_spikes": x, # [T, 2, 34, 34]
                        "z_states": self.z,        # [T, 100]
                        "vo_states": yo,        # [T, 100]
                        "t_in":self.trace_in,
                        "t_rec":self.trace_rec,
                        "t_out":self.trace_out,
                        "w_in":self.w_in.data,
                        "w_rec":self.w_rec.data,
                        "w_out":self.w_out.data,
                    }
                torch.save(visual_data, f'sim_output_{self.visu_count}.pt')
                # self.update_plot(x, self.z, yo, yt, L, self.trace_reg, self.trace_in,self.trace_rec, self.trace_out)
            
            # Compute network updates taking only the timesteps where the target is present
            # if self.t_crop != 0:
            #     L         =          L[:,:,-self.t_crop:]
            #     err       =        err[-self.t_crop:,:,:]
            #     self.trace_in  =   self.trace_in[:,:,:,-self.t_crop:]
            #     self.trace_rec =  self.trace_rec[:,:,:,-self.t_crop:]
            #     self.trace_out =  self.trace_out[:,:,-self.t_crop:]
            
            # Weight gradient updates
           
            g_in =  torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_in ,-1) * self.trace_in[:,:,:,-yo.shape[0]:] , dim=(0,3))
            g_rec = torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_rec,-1) * self.trace_rec[:,:,:,-yo.shape[0]:], dim=(0,3))
            g_out_chunk = torch.einsum('tbo,brt->or', err, self.trace_out[:,:,-yo.shape[0]:])
            
            if fr_reg:
                fr_mean = torch.mean(self.z, dim=(0, 1))
                reg_error = (self.f_target - fr_mean)
                
                reg_grad_in = torch.einsum('j,bjit->ji', reg_error, self.trace_in) / (T * B)
                reg_grad_rec = torch.einsum('j,bjit->ji', reg_error, self.trace_rec) / (T * B)

                
                # Aggiornamento gradienti
                g_in += self.c_reg * reg_grad_in
                g_rec += self.c_reg * reg_grad_rec

            if logit_reg:
                if start_id is not None:
                    g_out_chunk[start_id:end_id,:]+=0.1*self.w_out.data[start_id:end_id,:]
                else:
                    g_out_chunk+=0.1*self.w_out.data
            if penalties:
                g_in+=penalties['w_in']
                g_rec+=penalties['w_rec']
                # self.w_out.grad+=penalties['w_out']+0.1*self.w_out.data
                if start_id is not None:
                    g_out_chunk[start_id:end_id,:]+=penalties['w_out'][start_id:end_id,:] 
                else:
                    g_out_chunk+=penalties['w_out'] 
                    
                # self.w_out.grad+=0.001*self.w_out.data #L2 regularization
            
            if use_metapl:
                tmp_in = torch.exp(-self.xi * torch.abs(self.w_in))
                tmp_rec = torch.exp(-self.xi * torch.abs(self.w_rec))
                tmp_out = torch.exp(-self.xi * torch.abs(self.w_out))
                
                condition_in = (torch.sign(self.w_in) * g_in > 0.0)
                condition_rec = (torch.sign(self.w_rec) * g_rec > 0.0)
                condition_out = (torch.sign(self.w_out) * g_out_chunk > 0.0)

                # 4. Modula il gradiente
                mod_g_in = torch.where(condition_in, g_in * tmp_in, g_in)
                mod_g_rec = torch.where(condition_rec, g_rec * tmp_rec, g_rec)
                mod_g_out = torch.where(condition_out, g_out_chunk * tmp_out, g_out_chunk)

                # self.u_in = self.u_decay * self.u_in + (1 - self.u_decay) * torch.abs(g_in)
                # self.u_rec = self.u_decay * self.u_rec + (1 - self.u_decay) * torch.abs(g_rec)
                # if start_id is not None:
                    # self.u_out[start_id:end_id,:] = self.u_decay * self.u_out[start_id:end_id,:] + (1 - self.u_decay) * torch.abs(g_out_chunk)
                # else:
                    # self.u_out=self.u_decay * self.u_out + (1 - self.u_decay) * torch.abs(g_out_chunk)
                self.w_in.grad += self.lr_layer[0] *mod_g_in
                self.w_rec.grad += self.lr_layer[1] *mod_g_rec
                self.w_out.grad +=  self.lr_layer[2] *mod_g_out

                # self.w_in.grad  += g_in/ (1 + self.xi * self.u_in)
                # self.w_rec.grad += g_rec/ (1 + self.xi * self.u_in)
                # if start_id is not None:
                    # mod_g_out = g_out_chunk / (1 + self.xi * self.u_out[start_id:end_id,:])
                    # self.w_out.grad[start_id:end_id,:] += mod_g_out
                # else:
                    # mod_g_out = g_out_chunk / (1 + self.xi * self.u_out)
                    # self.w_out.grad+=mod_g_out
            else:
                self.w_in.grad  += self.lr_layer[0] *g_in
                self.w_rec.grad += self.lr_layer[1] *g_rec
                self.w_out.grad +=  self.lr_layer[2] *g_out_chunk
            
            return err

                
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
    lr_layer: tuple,
    f_target: int,
    c_reg:  float,
    lr_win: int,
    xi: float,
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
            lr_layer=lr_layer,
            f_target=f_target,
            c_reg=c_reg,
            lr_win=lr_win,
            xi=xi,
        )
