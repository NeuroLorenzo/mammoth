
import datasets.NMNIST



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
def reconstruct_image(input_1196, pixels_blocklist):
    """
    input_1196: tensore di un singolo timestep [1196]
    pixels_blocklist: caricata dal tuo file .txt
    """
    # 1. Creiamo il vettore "full" (tutte le polarità e pixel: 2312 elementi)
    full_vector = np.zeros(34 * 34 * 2)
    
    # 2. Identifichiamo gli indici "active" (quelli NON nella blocklist)
    all_indices = np.arange(34 * 34 * 2)
    # active_indices sono quelli che la tua rete usa effettivamente
    active_mask = np.isin(all_indices, pixels_blocklist, invert=True)
    
    # 3. Riempiamo il vettore full con i dati della rete
    full_vector[active_mask] = input_1196.cpu().numpy()
    
    # 4. Dividiamo per polarità
    pol_neg = full_vector[:34*34].reshape(34, 34)
    pol_pos = full_vector[34*34:].reshape(34, 34)
    
    # 5. Creiamo l'immagine RGB per Matplotlib
    rgb_img = np.zeros((34, 34, 3))
    rgb_img[..., 0] = pol_neg  # Rosso per polarità -1
    rgb_img[..., 2] = pol_pos  # Blu per polarità +1
    
    return rgb_img


class SpatialPermutation2D(object):
    def __init__(self, perm):
        self.perm = perm
        print('permutation:',perm)

    def __call__(self, sample):
        # sample ha shape (T, N_pixels) -> es. (300, 1156)
        # Permutiamo solo la seconda dimensione (i neuroni)
        # print('new dataset loaded:',torch.unique(sample),sample.shape)
        return sample[:, self.perm]
    


def update_neurons_in(x,split_idx,in_scatter):
    
    # 2. Dividiamo il vettore nei due blocchi di neuroni "esistenti"
    act_n = x[:split_idx] # attività neuroni negativi (quelli nello scatter)
    act_p = x[split_idx:] # attività neuroni positivi (quelli nello scatter)
    
    # 3. Creiamo i colori solo per i neuroni presenti nello scatter
    # Se hai usato un unico in_scatter con concatenate:
    # combined_activity = np.concatenate([act_n.cpu(), act_p.cpu()])
    
    # Applichiamo il filtro ::10 se lo hai usato nella definizione dello scatter!
    # Nota: se nello scatter hai usato [::10], devi farlo anche qui
    # activity_subset = combined_activity[::10]
    
    colors_n = ['blue' if val > 0 else 'black' for val in act_n.cpu()[::10]]
    colors_p = ['red' if val > 0 else 'black' for val in act_p.cpu()[::10]]
    colors=colors_n+colors_p
    in_scatter.set_facecolors(colors)
    in_scatter.set_edgecolors(colors)

def update_connections(ax, pos_in, pos_hid, tracce_matrice):
    """
    pos_in: array [N_in, 2] con le coordinate (x,y)
    pos_hid: array [N_hid, 2] con le coordinate (x,y)
    tracce_matrice: matrice [N_in, N_hid] con i valori delle tracce
    """
    linee = []
    colori = []
    
    # Esempio: colleghiamo solo le connessioni con traccia > 0.1 per non esplodere
    for i in range(len(pos_in)):
        for j in range(len(pos_hid)):
            if tracce_matrice[i, j] > 0.1:
                linee.append([pos_in[i], pos_hid[j]])
                colori.append(cm.magma(tracce_matrice[i, j]))

    lc = LineCollection(linee, colors=colori, linewidths=1)
    ax.add_collection(lc)

# Funzione per ottenere la maschera del top 10% dei pesi
def get_top_10_mask(weights,p):
    threshold = np.percentile(np.abs(weights), 100-p)
    return np.abs(weights) >= threshold
def create_segments(pos_start, pos_end, mask):
    segments = []
    # pos_start: [N1, 2], pos_end: [N2, 2]
    # mask: [N2, N1] (attenzione alla convenzione degli indici nei tuoi pesi)
    indices_end, indices_start = np.where(mask)
    for i_s, i_e in zip(indices_start, indices_end):
        segments.append([pos_start[i_s], pos_end[i_e]])
    return segments, indices_start, indices_end

def create_viz(all_data, blocklist,T=30):
    fig = plt.figure(figsize=(35, 15))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])


    
    x=all_data['input_spikes']
   
    t=0
    
    # 1. INPUT IMAGE (NMNIST)
    image=reconstruct_image(x[0,t,:],blocklist)
    ax_img = fig.add_subplot(gs[0])
    img_display = ax_img.imshow(image) # RGB
    ax_img.set_title("Input Events")

    # 2. INPUT NEURONS (Scatter columns)
    # n_in_n=(34*34)-np.count_nonzero(blocklist<(34*34))
    # n_in_p=(34*34)-np.count_nonzero(blocklist>=(34*34))
    # ax_in = fig.add_subplot(gs[1])
    # in_pos_x = np.concatenate([np.zeros(n_in_p), np.ones(n_in_n)])
    # in_pos_y = np.concatenate([np.arange(n_in_p), np.arange(n_in_n)])
    # in_scatter = ax_in.scatter(in_pos_x[::10], in_pos_y[::10], c='black', s=5)
    
    def update(t):
        # Update Immagine NMNIST (Rosso/Blu)
        image=reconstruct_image(x[0,t,:],blocklist)
        img_display.set_data(image)

        
        
        return img_display
        # return img_display, hid_scatter, out_scatter

    ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    plt.show()
    ani.save('srnn_activity2.gif', writer='ffmpeg')




SIZE = (300, 1196)
perm = torch.randperm(SIZE[1])
transform = transforms.Compose([
        torch.from_numpy, 
        SpatialPermutation2D(perm)
        # SpatialPermutation(perm)

    ])

pixels_blocklist = np.loadtxt("../data/NMNIST/NMNIST_pixels_blocklist.txt")

img=reconstruct_image(perm, pixels_blocklist)
img.shape
plt.imshow(img/1196)




train_dataset = NMNIST(str(self.train_path), train=True, download=False, transform=transform,dt=self.dt,seq_len=self.seq_len_train)
x=transform(data)

create_viz(x,pixels_blocklist)