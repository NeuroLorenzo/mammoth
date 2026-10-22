# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from random import seed
from typing import List,Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu
import numpy as np
from backbone import MammothBackbone, register_backbone
import os
from line_profiler import profile

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.

    Args:
        in_planes: number of input channels
        out_planes: number of output channels
        stride: stride of the convolution

    Returns:
        convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.

        Args:
            in_planes: the number of input channels
            planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.return_prerelu = False
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.return_prerelu:
            self.prerelu = out.clone()

        out = relu(out)
        return out



class PreActBlock(nn.Module):
    """
    Pre-activation version of the BasicBlock.
    From: ``CNLL: A Semi-supervised Approach For Continual Noisy Label Learning''
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, initial_conv_k=3) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
            initial_conv_k: the kernel size of the initial convolution
        """
        super(ResNet, self).__init__()
        self.return_prerelu = False
        self.device = "cpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        if initial_conv_k != 3:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(3, nf * 1 * block.expansion, kernel_size=initial_conv_k, stride=2, padding=3, bias=False)
        else:
            self.conv1 = conv3x3(3, nf * 1 * block.expansion)
        self.bn1 = nn.BatchNorm2d(nf * 1 * block.expansion)
        if len(num_blocks) == 4:
            self.feature_dim = nf * 8 * block.expansion
            self.layer1 = self._make_layer(block, nf * 1 * block.expansion, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, nf * 2 * block.expansion, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, nf * 4 * block.expansion, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, self.feature_dim, num_blocks[3], stride=2)
        else:
            self.feature_dim = nf * 4 * block.expansion
            self.layer1 = self._make_layer(block, nf * 1 * block.expansion, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, nf * 2 * block.expansion, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, self.feature_dim, num_blocks[2], stride=2)
            self.layer3._modules[f"{num_blocks[2]-1}"].is_last = True
            self.layer4 = nn.Identity()
            self.layer4.return_prerelu = False

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'full'] but got {}".format(returnt))

@register_backbone("preact-resnet18")
def preact_resnet18(num_classes: int, num_filters: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        ResNet network
    """
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes, num_filters)

@register_backbone("resnet18")
def resnet18(num_classes: int, num_filters: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_filters)


@register_backbone("resnet18_7x7_pt")
def resnet18_7x7(num_classes: int) -> ResNet:
    """
    Instantiates a ResNet18 network with a 7x7 initial convolution and pretrained weights.

    Args:
        num_classes: number of output classes

    Returns:
        ResNet network
    """
    from torchvision.models import ResNet18_Weights # type: ignore[import-untyped]
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, 64, initial_conv_k=7)
    pretrain_weights = ResNet18_Weights.DEFAULT
    st = pretrain_weights.get_state_dict(progress=True, check_hash=True)
    for k in list(st.keys()):
        if 'downsample' in k:
            st[k.replace('downsample', 'shortcut')] = st.pop(k)
        elif k.startswith('fc'):
            st.pop(k)
    missing, unexp = net.load_state_dict(st, strict=False)
    assert len([k for k in missing if not k.startswith('classifier')]) == 0, missing
    assert len(unexp) == 0, unexp
    return net


@register_backbone("reduced-resnet18")
def reduced_resnet18(num_classes: int) -> ResNet:
    """
    Instantiates a ResNet18 network with a third of the parameters, as used in `Gradient Episodic Memory for Continual Learning`

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, 20)


@register_backbone("resnet34")
def resnet34(num_classes: int, num_filters: int = 64) -> ResNet:
    """
    Instantiates a ResNet34 network.

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        ResNet network
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, num_filters)


@register_backbone("resnet18_spr")
def resnet18_spr(num_classes: int) -> ResNet:
    """
    Instantiates a ResNet18 network as used in the original `SPR <https://arxiv.org/abs/2110.07735>`_ paper.

    Args:
        num_classes: number of output classes

    Returns:
        ResNet network
    """

    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, 64, initial_conv_k=7)  # spr uses 7x7 conv even for 32x32 images



@register_backbone("resnet18cus") #decorator, per riferirsi alla classe
def resnet18cus( srnn_hidden_size: int,
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
    syn_scale: bool,
    sparsity: tuple,
    lr_layer: tuple,
    f_target: int,
    c_reg:  float,
    lr_win: int,
    net_state: str,
    xi: float,num_classes: int, num_filters: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        ResNet network
    """
    return RESNETCUS( n_in, srnn_hidden_size, n_out, n_t, thr, tau_m, tau_o, b_o, 
                 gamma, dt,classif,keep_sparsity,syn_scale,sparsity,lr_layer,
                 f_target,c_reg,lr_win,xi,BasicBlock, [2, 2, 2, 2], num_classes, num_filters,net_state=net_state)


class RESNETCUS(ResNet):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self,  n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, 
                 gamma, dt,classif,keep_sparsity,syn_scale,sparsity,lr_layer,
                 f_target,c_reg,lr_win,xi, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, initial_conv_k=3,
                  model='LIF' , w_init_gain=(0.5,0.1,0.5), net_state=None, t_crop=100,
                    visualize=False, visualize_light=True) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
            initial_conv_k: the kernel size of the initial convolution
        """
        super(ResNet, self).__init__()
        self.return_prerelu = False
        self.device = "cpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        if initial_conv_k != 3: # skipped
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(3, nf * 1 * block.expansion, kernel_size=initial_conv_k, stride=2, padding=3, bias=False)
        else:
            self.conv1 = conv3x3(3, nf * 1 * block.expansion)
        self.bn1 = nn.BatchNorm2d(nf * 1 * block.expansion)
        if len(num_blocks) == 4: # num_block= [2,2,2,2]
            self.feature_dim = nf * 8 * block.expansion
            self.layer1 = self._make_layer(block, nf * 1 * block.expansion, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, nf * 2 * block.expansion, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, nf * 4 * block.expansion, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, self.feature_dim, num_blocks[3], stride=2)
        else:
            self.feature_dim = nf * 4 * block.expansion
            self.layer1 = self._make_layer(block, nf * 1 * block.expansion, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, nf * 2 * block.expansion, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, self.feature_dim, num_blocks[2], stride=2)
            self.layer3._modules[f"{num_blocks[2]-1}"].is_last = True
            self.layer4 = nn.Identity()
            self.layer4.return_prerelu = False

        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # load pretrained weights if provided
        if net_state is not None:
            self._load_net_state(net_state)

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
        self.syn_scale=syn_scale
        self.fire_sparsities=[]
        self.f_target=None
        # self.f_target=f_target
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
        
        # Nel tuo __init__
        self.xi = xi  # Forza della metaplasticità
        self.u_decay = 0.99  # Coeff di persistenza della memoria sinaptica

        self.u_in = torch.zeros_like(self.w_in).to(self.device)
        self.u_rec = torch.zeros_like(self.w_rec).to(self.device)
        self.u_out = torch.zeros_like(self.w_out).to(self.device)

        self.batch_logs={}
        self.z_mean_acc=None
        self.mask_z=None
        self.net_state=net_state
        self.first_task=True
    
    def _load_net_state(self, path: Union[str, os.PathLike]) -> None:
        cp = torch.load(path, map_location='cpu')
        state = cp.get('model', cp)   # handles both raw and Mammoth-wrapped checkpoints

        # strip the 'net.' prefix that Mammoth adds when wrapping inside ContinualModel
        PREFIX = 'net.'
        stripped = {
            (k[len(PREFIX):] if k.startswith(PREFIX) else k): v
            for k, v in state.items()
        }

        # strict=False: ignores keys not present in this subnetwork
        # (e.g. 'classifier' if you're using this as a feature extractor)
        missing, unexpected = self.load_state_dict(stripped, strict=False)

        if missing:
            print(f'[ResNet] missing keys  ({len(missing)}): {missing[:5]} ...')
        if unexpected:
            print(f'[ResNet] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...')
        print(f'[ResNet] weights loaded from {path}')

    def reset_parameters(self, gain,sparsity) -> None:
        
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0]*self.w_in.data
        mask_in = self.create_mask(self.w_in.shape, sparsity[0])
        self.w_in.data *= mask_in
        self.target_w_in=self.w_in.norm(p=1,dim=1,keepdim=True).to(self.device) 

        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1]*self.w_rec.data
        mask_rec = self.create_mask(self.w_rec.shape, sparsity[1])
        self.w_rec.data *= mask_rec
        self.target_w_rec=self.w_rec.norm(p=1,dim=1,keepdim=True).to(self.device)

        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2]*self.w_out.data
        mask_out = self.create_mask(self.w_out.shape, sparsity[2])
        self.w_out.data *= mask_out
        self.target_w_out=self.w_out.norm(p=1,dim=1,keepdim=True)

        
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

    def integrate_odes(self,T,x):
        for t in range(T - 1):
            
            self.v[t+1] = (
                self.alpha * self.v[t]
                + self.z[t] @ self.w_rec.T
                + x[:,t,:] @ self.w_in.T
                - self.z[t] * self.thr
            )

            self.z[t+1] = (self.v[t+1] > self.thr).float()
            if self.mask_z is not None:
                self.z[t+1][:,self.mask_z]=0 # remove 10 most firing neurons
            self.vo[t+1] = (
                self.kappa * self.vo[t]
                + self.z[t+1] @ self.w_out.T
                + self.b_o
            )
        
    @profile
    @torch.no_grad()
    def apply_homeostasis(self):
        eps=1e-8
        self.w_in.data*=self.target_w_in.to(self.w_in.device)/(self.w_in.norm(p=1,dim=1,keepdim=True)+ eps)
        self.w_rec.data*=self.target_w_rec.to(self.w_rec.device)/(self.w_rec.norm(p=1,dim=1,keepdim=True)+ eps)
        self.w_out.data*=self.target_w_out.to(self.w_out.device)/(self.w_out.norm(p=1,dim=1,keepdim=True)+ eps)


    @profile
    def apply_masks(self):
        
        # save masks for later use (important!)
        self.mask_in = self.mask_in.to(self.w_in.device)
        self.mask_rec = self.mask_rec.to(self.w_rec.device)
        self.mask_out = self.mask_out.to(self.w_out.device)
        with torch.no_grad():
            self.w_in *= self.mask_in
            self.w_rec *= self.mask_rec
            self.w_out *= self.mask_out

    def create_mask(self, shape, sparsity_level):
        """
        Generate a binary mask with the given sparsity level.
        sparsity_level = fraction of weights set to zero.
        """
        return torch.tensor(
            np.random.choice([0, 1], size=shape, p=[sparsity_level, 1 - sparsity_level]),
            dtype=torch.float32
        )

    def feature_to_spikes(self,features, T, max_val=10.0,seed=42):
        """
        Converte le feature in spike trains.
        Args:
            features: Tensor (B, 512) - Output della ResNet
            T: int - Numero di step temporali
            max_val: float - Il valore massimo osservato nelle feature per la normalizzazione
        Returns:
            spikes: Tensor (B, 512, T)
        """
        # 1. Normalizzazione (Portiamo i valori tra 0 e 1)
        # Usiamo un clamp per sicurezza se qualche valore supera max_val
        probs = torch.clamp(features / max_val, 0, 1)
        
        # 2. Espansione delle probabilità su T dimensioni
        # Da (B, 512) -> (B, 512, 1) -> (B, 512, T)
        probs = probs.unsqueeze(-1).repeat(1, 1, T)
        
        # 3. Generazione degli spike di Poisson
        # Generiamo numeri casuali uniformi [0, 1] e confrontiamo con le probabilità
        gen = torch.Generator(device=features.device)
        gen.manual_seed(seed)
        rand_vals = torch.rand(probs.shape, generator=gen, device=features.device, dtype=probs.dtype)
    
        # 5. Spikes
        spikes = (rand_vals < probs).float()

        return spikes

    def freeze_bn(self,m):
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    @profile
    def forward(self, x: torch.Tensor,  yt=None, returnt="out",lr_win=0) -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        self.apply(self.freeze_bn)
        with torch.no_grad():
            if x.shape[1] != 512:
                if returnt == "feats_only":
                    spikes=x
                else:
                    out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
                    if self.return_prerelu:
                        out_0_t = out_0.clone()
                    out_0 = relu(out_0)
                    if hasattr(self, 'maxpool'):
                        out_0 = self.maxpool(out_0)

                    out_1 = self.layer1(out_0)  # -> 64, 32, 32
                    out_2 = self.layer2(out_1)  # -> 128, 16, 16
                    out_3 = self.layer3(out_2)  # -> 256, 8, 8
                    out_4 = self.layer4(out_3)  # -> 512, 4, 4

                    feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
                    feature = feature.view(feature.size(0), -1)  # 512
                    
                    spikes=self.feature_to_spikes(feature,T=100,max_val=10).permute(0,2,1)
            else:
                spikes=self.feature_to_spikes(x,T=100,max_val=10).permute(0,2,1)
                
            if returnt == 'features':
                return feature

            """
            x: (B, T, n_in) 
            yt: (T, B, n_out) or None
            """
            if self.keep_sparsity:
                self.apply_masks()
            if self.syn_scale:
                self.apply_homeostasis()
            in_shape=x.shape
            if len(in_shape)==2:
                x = x.unsqueeze(0)
            B,T,N = spikes.shape
            assert T >0, f"Temporal Consistency Error: data len T ({T}) different from the simulation times n_t ({self.n_t})"
            # assert T == self.n_t, f"Temporal Consistency Error: data len T ({T}) different from the simulation times n_t ({self.n_t})"

            self.init_net(B, T, self.n_in, self.n_rec, self.n_out)    # Network reset
            
            # Kill self-connections
            self.w_rec.data *= (1 - torch.eye(self.n_rec, device=self.device))
            # feats=resnet(x)
            # feats_spike=convspikes(feats)
            # self.integrate_odes(T,feats_spike)
            self.integrate_odes(T,spikes)

            fire_sparsity=torch.count_nonzero(self.z)/(T*B*N)
            if self.z_mean_acc is None:
                
                self.z_mean_acc=self.z.mean(axis=(0,1))
            else:
                self.z_mean_acc+=self.z.mean(axis=(0,1))
            

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

            
            if returnt == "feats_only":
                # return yo
                return yo.permute(1,2,0)
            if returnt == "spikes_feats":
                # return yo
                return yo.permute(1,2,0),spikes
            
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
    def compute_fr_reg(self, T, B, log_data_flag, g_in, g_rec):
        # --- TASK 0: Costruzione / Accumulo del Target ---
        if self.first_task:
            # Firing rate medio del batch corrente in Hz (shape: [num_neurons])
            current_fr = torch.mean(self.z, dim=(0, 1)) * 1000.0 
            
            if self.f_target is not None:
                # Media mobile esponenziale (EMA) per aggiornare il target ad ogni batch
                alpha = 0.05  # Peso del batch corrente
                self.f_target = (1.0 - alpha) * self.f_target + alpha * current_fr.detach()
            else:
                self.f_target = current_fr.detach().clone()

                
            return g_in, g_rec
            

        # --- TASK > 0: Calcolo Regolarizzazione via e-prop ---
        else:
            # Firing rate corrente del batch (Hz)
            fr_mean = torch.mean(self.z, dim=(0, 1)) * 1000.0
            
            # 1. Maschera sui neuroni che erano significativamente attivi nel Task 0
            threshold_active = 0.1  # soglia in Hz
            active_mask = (self.f_target > threshold_active).float()  # [num_neurons]
            
            # Se nessun neurone è attivo, restituisci i gradienti inalterati
            if active_mask.sum() == 0:
                return g_in, g_rec

            # 2. Normalizzazione degli errori (Z-score) per non far dominare le alte frequenze
            target_mean = self.f_target.mean()
            target_std = self.f_target.std() + 1e-6
            
            # Errore normalizzato: dL/d(fr) = (fr_mean - f_target) / (std^2)
            # Nota il segno: (fr_mean - f_target) per ridurre l'errore!
            raw_error = (fr_mean - self.f_target)
            reg_error = (raw_error / (target_std ** 2)) * active_mask  # Applichiamo la maschera
            
            # 3. Proiezione dell'errore sulle Eligibility Traces (e-prop)
            reg_grad_in = torch.einsum('j, bjit -> ji', reg_error, self.trace_in) / (T * B)
            reg_grad_rec = torch.einsum('j, bjit -> ji', reg_error, self.trace_rec) / (T * B)

            # 4. Sottrarre il gradiente della loss (Discesa del gradiente: theta = theta - lr * grad)
            # Se c_reg è positivo, sommiamo il gradiente se e-prop lavora sui gradienti diretti
            g_in += self.c_reg * reg_grad_in
            g_rec += self.c_reg * reg_grad_rec

            # Logging
            if log_data_flag:
                if 'fr_reg' not in self.batch_logs:
                    self.batch_logs['fr_reg'] = {}
                self.batch_logs['fr_reg']['g_in'] = (self.c_reg * reg_grad_in).detach().cpu()
                self.batch_logs['fr_reg']['g_rec'] = (self.c_reg * reg_grad_rec).detach().cpu()

            return g_in, g_rec

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
        self.apply(self.freeze_bn)
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
                L = torch.einsum('tbo,or->brt', err[:,:,start_id:end_id], self.w_out[start_id:end_id,:])
            else:
                # err=0.5*err
                L = 0.5*torch.einsum('tbo,or->brt', err, self.w_out)
                

            
            # Weight gradient updates
            g_in,g_rec,g_out_chunk=self.compute_grads(L,err,yo,T)
            
            if log_data_flag:
                self.batch_logs={}
                self.batch_logs['main_grad'] = {}
                self.batch_logs['main_grad']['g_in']=g_in.cpu()
                self.batch_logs['main_grad']['g_rec']=g_rec.cpu()
                self.batch_logs['main_grad']['g_out']=g_out_chunk.cpu()
            
            if fr_reg:
                g_in,g_rec=self.compute_fr_reg(T,B,log_data_flag,g_in,g_rec)
                
            if logit_reg:
                if start_id is not None:
                    g_out_chunk[start_id:end_id,:]-=1e-8*self.w_out.data[start_id:end_id,:]
                else:
                    g_out_chunk-=1e-8*self.w_out.data
                
                if log_data_flag:
                    self.batch_logs['l_reg'] = -1e-8*self.w_out.data.cpu()

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
                self.compute_metapl(g_in,g_rec,g_out_chunk,log_data_flag)
            else:
                self.w_in.grad  += self.lr_layer[0] *g_in
                self.w_rec.grad += self.lr_layer[1] *g_rec
                self.w_out.grad +=  self.lr_layer[2] *g_out_chunk
            
                
            return err

          

    def get_eligibility(self):
        return self.trace_in,self.trace_rec,self.trace_out
