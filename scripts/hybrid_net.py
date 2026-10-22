import torch 
import matplotlib.pyplot as plt

# Caricamento e preparazione pesi
cp = torch.load('./checkpoints/resnet18_scifar10_sgd_joint_inference.pt', map_location='cpu')
state = cp.get('model', cp)
kernels = state['net.conv1.weight'].detach().cpu()

# Creazione figura
fig, axs = plt.subplots(8, 8, figsize=(20, 20))

# Variabile per memorizzare l'ultimo oggetto mappabile
im = None 

for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        # Permute fondamentale: da (C, H, W) a (H, W, C)
        img = kernels[idx].permute(1, 2, 0)
        
        # Normalizzazione
        img_min, img_max = img.min(), img.max()
        img_normalized = (img - img_min) / (img_max - img_min)
        
        # Salviamo il risultato di imshow in 'im'
        im = axs[i][j].imshow(img_normalized.numpy(), interpolation='nearest')
        axs[i][j].axis('off')

# Regolazione spazio per far posto alla colorbar
fig.subplots_adjust(right=0.85) 
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax)

plt.show()