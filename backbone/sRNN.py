import torch
import torch.nn as nn
import logging

from backbone import MammothBackbone, num_flat_features, register_backbone, xavier
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from line_profiler import profile

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

        self.batch_logs={}
    def create_mask(self, shape, sparsity_level):
        """
        Generate a binary mask with the given sparsity level.
        sparsity_level = fraction of weights set to zero.
        """
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

        
        # save masks for later use
        self.mask_in = self.create_mask(self.w_in.shape, sparsity[0])
        self.mask_rec = self.create_mask(self.w_rec.shape, sparsity[1])
        self.mask_out = self.create_mask(self.w_out.shape, sparsity[2])
        
        self.apply_masks()

    @profile
    def init_net(self, n_b, n_t, n_in, n_rec, n_out):
        # Reset Network state variables
        self.v  = torch.zeros(n_t, n_b, n_rec, device=self.device)
        self.vo = torch.zeros(n_t, n_b, n_out, device=self.device)
        self.z  = torch.zeros(n_t, n_b, n_rec, device=self.device)

        # reset gradient buffers
        
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

    @profile
    def integrate_odes(self,T,x):
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
        
    @profile
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

        self.integrate_odes(T,x)

        fire_sparsity=torch.count_nonzero(self.z)/(T*B*N)
        self.fire_sparsities.append(100 -100*fire_sparsity.cpu())
        
        
        if self.classif:
            yo = F.softmax(self.vo[-self.lr_win:,:,:], dim=2)
        else:
            yo = self.vo

        if yt is None:
            final_logits = self.vo[-self.lr_win:,:,:] 
        
            # Mean over time dimension result shape: [Batch, n_out]
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
    
    @profile
    def compute_traces(self,x,h,B):
        self.trace_in    = F.conv1d(     x.permute(0,2,1), self.alpha_conv.expand(self.n_in ,-1,-1), padding=self.n_t, groups=self.n_in )[:,:,1:self.n_t+1].unsqueeze(1).expand(-1,self.n_rec,-1,-1)  #B, n_rec, n_in , n_t 
        self.trace_in    = torch.einsum('tbr,brit->brit', h, self.trace_in )                                                                                                                          #B, n_rec, n_in , n_t 
        self.trace_rec   = F.conv1d(self.z.permute(1,2,0), self.alpha_conv.expand(self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec)[:,:, :self.n_t  ].unsqueeze(1).expand(-1,self.n_rec,-1,-1)  #B, n_rec, n_rec, n_t
        self.trace_rec   = torch.einsum('tbr,brit->brit', h, self.trace_rec)                                                                                                                          #B, n_rec, n_rec, n_t    
        self.trace_reg   = self.trace_rec
        self.trace_out  = F.conv1d(self.z.permute(1,2,0), self.kappa_conv.expand(self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec)[:,:,1:self.n_t+1]  #B, n_rec, n_t
        self.trace_in     = F.conv1d(   self.trace_in.reshape(B,self.n_in *self.n_rec,self.n_t), self.kappa_conv.expand(self.n_in *self.n_rec,-1,-1), padding=self.n_t, groups=self.n_in *self.n_rec)[:,:,1:self.n_t+1].reshape(B,self.n_rec,self.n_in ,self.n_t)   #B, n_rec, n_in , n_t  
        self.trace_rec    = F.conv1d(  self.trace_rec.reshape(B,self.n_rec*self.n_rec,self.n_t), self.kappa_conv.expand(self.n_rec*self.n_rec,-1,-1), padding=self.n_t, groups=self.n_rec*self.n_rec)[:,:,1:self.n_t+1].reshape(B,self.n_rec,self.n_rec,self.n_t)   #B, n_rec, n_rec, n_t
    
    @profile
    def compute_grads(self,L,err,yo,T):
        g_in =  torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_in ,-1) * self.trace_in[:,:,:,-yo.shape[0]:] , dim=(0,3))/T
        g_rec = torch.sum(L.unsqueeze(2).expand(-1,-1,self.n_rec,-1) * self.trace_rec[:,:,:,-yo.shape[0]:], dim=(0,3))/T
        g_out_chunk = torch.einsum('tbo,brt->or', err, self.trace_out[:,:,-yo.shape[0]:])
        return g_in,g_rec,g_out_chunk
    
    @profile
    def print_penalities(self,g_in,g_rec,g_out,penalties):
        print(f"g_in mean: {g_in.abs().mean().item()}; penalty_in mean {penalties['w_in'].abs().mean()}")
        print(f"g_rec mean: {g_rec.abs().mean().item()}; penalty_rec mean {penalties['w_rec'].abs().mean()}")
        print(f"g_out mean: {g_out.abs().mean().item()}; penalty_out mean {penalties['w_out'].abs().mean()}")
    
  
    @profile
    def compute_fr_reg(self,T,B,log_data_flag):
        fr_mean = torch.mean(self.z, dim=(0, 1))
        reg_error = (self.f_target - fr_mean)
        
        reg_grad_in = torch.einsum('j,bjit->ji', reg_error, self.trace_in) / (T * B)
        reg_grad_rec = torch.einsum('j,bjit->ji', reg_error, self.trace_rec) / (T * B)

        
        # Update gradients
        g_in += self.c_reg * reg_grad_in
        g_rec += self.c_reg * reg_grad_rec
        if log_data_flag:
            self.batch_logs['fr_reg'] = {}

            self.batch_logs['fr_reg']['g_in']=self.c_reg * reg_grad_in.cpu()
            self.batch_logs['fr_reg']['g_rec']=self.c_reg * reg_grad_rec.cpu()


    @profile
    def compute_metapl(self,g_in,g_rec,g_out_chunk,log_data_flag):
        tmp_in = torch.exp(-self.xi * torch.abs(self.w_in))
        tmp_rec = torch.exp(-self.xi * torch.abs(self.w_rec))
        tmp_out = torch.exp(-self.xi * torch.abs(self.w_out))
        
        # Symmetric Metaplasticity
        mod_g_in=tmp_in*g_in
        mod_g_rec=tmp_rec*g_rec
        mod_g_out=tmp_out*g_out_chunk

        # # Asymmetric Metaplasticity

        # condition_in = (torch.sign(self.w_in) * g_in > 0.0)
        # condition_rec = (torch.sign(self.w_rec) * g_rec > 0.0)
        # condition_out = (torch.sign(self.w_out) * g_out_chunk > 0.0)

        # mod_g_in = torch.where(condition_in, g_in * tmp_in, g_in)
        # mod_g_rec = torch.where(condition_rec, g_rec * tmp_rec, g_rec)
        # mod_g_out = torch.where(condition_out, g_out_chunk * tmp_out, g_out_chunk)

        

        # Thresholded metaplasticity 
        # threshold = 0.1
        # condition_important = (torch.abs(self.w_rec) > threshold)
        # # Se importante, moltiplica il gradiente per 0.01, altrimenti lascialo normale (1.0)
        # freno = torch.where(condition_important, 0.01, 1.0)


        self.w_in.grad += self.lr_layer[0] *mod_g_in
        self.w_rec.grad += self.lr_layer[1] *mod_g_rec
        self.w_out.grad +=  self.lr_layer[2] *mod_g_out
        
        if log_data_flag:
            self.batch_logs['metapl_reduction'] = {}
            avg_reduction = mod_g_in.norm() / (g_in.norm() + 1e-8)
            self.batch_logs['metapl_reduction']['g_in'] = avg_reduction.item()
            avg_reduction = mod_g_rec.norm() / (g_rec.norm() + 1e-8)
            self.batch_logs['metapl_reduction']['g_rec'] = avg_reduction.item()
            avg_reduction = mod_g_out.norm() / (g_out_chunk.norm() + 1e-8)
            self.batch_logs['metapl_reduction']['g_out'] = avg_reduction.item()

                
    @profile
    def grads_batch(self, x, yo, yt,start_id,end_id,loss=None,penalties=None,fr_reg=False,use_metapl=False,logit_reg=False,log_data_flag=False):   
        with torch.no_grad():
            
            # Surrogate derivatives
            h = self.gamma*torch.max(torch.zeros_like(self.v), 1-torch.abs((self.v-self.thr)/self.thr))
            B,T,_=x.shape

            # Input and recurrent eligibility vectors for the 'LIF' model (vectorized computation, model-dependent)
            assert self.model == "LIF", "Nice try, but model " + self.model + " is not supported. ;-)"
            self.compute_traces(x,h,B)
            

            # Learning signals
            err = yo - yt
            
            if start_id is not None:
                L = torch.einsum('tbo,or->brt', err, self.w_out[start_id:end_id,:])
            else:
                L = torch.einsum('tbo,or->brt', err, self.w_out)
                

            
            # Weight gradient updates
            g_in,g_rec,g_out_chunk=self.compute_grads(L,err,yo,T)
            
            if log_data_flag:
                self.batch_logs={}
                self.batch_logs['main_grad'] = {}
                self.batch_logs['main_grad']['g_in']=g_in.cpu()
                self.batch_logs['main_grad']['g_rec']=g_rec.cpu()
                self.batch_logs['main_grad']['g_out']=g_out_chunk.cpu()
            
            if fr_reg:
                self.compute_fr_reg(T,B,log_data_flag)
                
            if logit_reg:
                if start_id is not None:
                    g_out_chunk[start_id:end_id,:]+=0.1*self.w_out.data[start_id:end_id,:]
                else:
                    g_out_chunk+=0.1*self.w_out.data
                
                if log_data_flag:
                    self.batch_logs['l_reg'] = 0.1*self.w_out.data.cpu()

            if penalties:
                g_in+=penalties['w_in']
                g_rec+=penalties['w_rec']
                penalties['w_out']=penalties['w_out']/1000 # scaling down wout penalties 
                if start_id is not None:
                    g_out_chunk[start_id:end_id,:]+=penalties['w_out'][start_id:end_id,:] 
                else:
                    g_out_chunk+=penalties['w_out'] 

                if log_data_flag:
                    self.batch_logs['ewc'] = {}
                    self.batch_logs['ewc']['w_in'] =penalties['w_in']
                    self.batch_logs['ewc']['w_rec'] =penalties['w_rec']
                    self.batch_logs['ewc']['w_out'] =penalties['w_out']
                    
                # self.w_out.grad+=0.001*self.w_out.data #L2 regularization
            
            if use_metapl:
                self.compute_metapl()
            else:
                self.w_in.grad  += self.lr_layer[0] *g_in
                self.w_rec.grad += self.lr_layer[1] *g_rec
                self.w_out.grad +=  self.lr_layer[2] *g_out_chunk
            
                
            return err

          

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
