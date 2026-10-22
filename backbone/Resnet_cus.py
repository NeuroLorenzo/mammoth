# import torch
# import torch.nn as nn
# import logging
# import math


# from torch.nn.functional import avg_pool2d, relu


# from backbone import MammothBackbone, num_flat_features, register_backbone, xavier,ResNet,BasicBlock
# import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn.functional as F
# from line_profiler import profile


# @register_backbone("ressrnn_spiking")
# def ressrnn_spiking(
#     num_classes: int,
#     srnn_hidden_size: int,
#     n_in: int,
#     n_out:int,
#     n_t: int,
#     thr: float,
#     tau_m: float,
#     tau_o: float,
#     b_o: float,
#     gamma: float,
#     dt: float,
#     classif: bool,
#     keep_sparsity: bool,
#     syn_scale: bool,
#     sparsity: tuple,
#     lr_layer: tuple,
#     f_target: int,
#     c_reg:  float,
#     lr_win: int,
#     xi: float,
#     ) -> BaseRESSRNN:

#         return BaseRESSRNN(
#             n_in=n_in,
#             n_rec=srnn_hidden_size,
#             n_out=n_out,
#             n_t=n_t,
#             thr=thr,
#             tau_m=tau_m,
#             tau_o=tau_o,
#             b_o=b_o,
#             gamma=gamma,
#             dt=dt,
#             classif=classif,
#             keep_sparsity=keep_sparsity,
#             syn_scale=syn_scale,
#             sparsity=sparsity,
#             lr_layer=lr_layer,
#             f_target=f_target,
#             c_reg=c_reg,
#             lr_win=lr_win,
#             xi=xi,
#         )
