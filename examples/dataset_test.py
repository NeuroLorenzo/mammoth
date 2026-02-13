import mammoth.datasets
from mammoth.utils.conf import base_path
import torchvision.transforms as transforms
from mammoth.datasets.transforms.permutation import Permutation
import numpy as np
import datasets.perm_mnist as pm

import datasets.neuro_mnist as nm
import cProfile
import pstats

# transform = transforms.Compose((transforms.ToTensor(), Permutation(np.prod(pm.PermutedMNIST.SIZE))))
transform=None

profiler = cProfile.Profile()
profiler.enable()
# train_dataset_n = nm.NMNIST(base_path() + 'NMNIST/Test',selected_targets=[0,1],
train_dataset_n = nm.NMNIST(base_path() + 'NMNIST/Test', train=True, download=True, transform=transform)

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumtime")  # sort by cumulative time
stats.print_stats(20)        # top 20 functions



sample_id=1000
labels=[]
for i in range(train_dataset_n.__len__()):
    labels.append(train_dataset_n[i][1][0])
# labels=np.concatenate(labels)
print('labels=',np.unique(labels))
print(type(train_dataset_n))
print(train_dataset_n)
print(train_dataset_n[sample_id])
# print(train_dataset_n[sample_id][1].shape)


# transform = transforms.Compose((transforms.ToTensor(), Permutation(np.prod(pm.PermutedMNIST.SIZE))))

train_dataset = pm.MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
labels=[]

for i in range(train_dataset.__len__()):
    labels.append(train_dataset[i][1])

print('labels=',np.unique(labels))
print(type(train_dataset))

print(train_dataset)

print(train_dataset[sample_id])