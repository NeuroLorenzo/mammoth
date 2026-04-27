import os
import shutil
import random

def downsample_dataset(src_root, dest_root, ratio=0.1):
    """
    Copia una percentuale di file .bin mantenendo la struttura delle cartelle.
    :param src_root: Percorso della cartella originale (contiene Train e Test)
    :param dest_root: Percorso della nuova cartella di output
    :param ratio: Percentuale di file da mantenere (es. 0.1 = 10%)
    """
    # Estensioni supportate
    ext = ".bin"

    for root, dirs, files in os.walk(src_root):
        # Filtra solo i file .bin
        bin_files = [f for f in files if f.endswith(ext)]
        
        if bin_files:
            # Calcola quanti file mantenere
            num_to_keep = max(1, int(len(bin_files) * ratio))
            selected_files = random.sample(bin_files, num_to_keep)
            
            # Crea la struttura delle sottocartelle nella destinazione
            rel_path = os.path.relpath(root, src_root)
            target_dir = os.path.join(dest_root, rel_path)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copia i file selezionati
            for f in selected_files:
                shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))
                
    print(f"Downsampling completato! Dati salvati in: {dest_root}")

# --- CONFIGURAZIONE ---
SORGENTE = "./data/NMNIST"
PERCENTUALE = 0.5  # Modifica questo (0.1 = 10%, 0.5 = 50%, ecc.)
DESTINAZIONE = SORGENTE+f'_{PERCENTUALE}'
downsample_dataset(SORGENTE, DESTINAZIONE, PERCENTUALE)