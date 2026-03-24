import os
import torch
import numpy as np
from tqdm import tqdm

def preprocess_nmnist(input_dir, output_dir, n_t=30, shape=(2, 34, 34)):
    """
    input_dir: percorso alla cartella NMNIST_10p (con i .bin)
    output_dir: dove salvare i .pt
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Trova tutti i file .bin
    bin_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.bin'):
                bin_files.append(os.path.join(root, f))

    print(f"Trovati {len(bin_files)} file da processare.")

    for bin_path in tqdm(bin_files):
        # 1. Caricamento e conversione (usa la tua funzione esistente)
        # Supponiamo che restituisca un tensore [T, C, H, W]
        spike_tensor = convert_bin_to_tensor(bin_path, n_t=n_t) 
        
        # 2. Ottimizzazione memoria: converti in uint8 (0 e 1)
        spike_tensor = spike_tensor.to(torch.uint8)

        # 3. Costruzione path di output
        # Mantiene la struttura delle sottocartelle (es. Train/0/file.bin -> Train/0/file.pt)
        relative_path = os.path.relpath(bin_path, input_dir)
        target_path = os.path.join(output_dir, relative_path.replace('.bin', '.pt'))
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # 4. Salvataggio
        torch.save(spike_tensor, target_path)

# Nota: dovrai incollare qui la tua funzione 'convert_bin_to_tensor'