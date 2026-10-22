import os
import torch

folder = "checkpoints"
new_weights = "sgd-snn_seq-cifar10_default_10000_1_20260729-111608_5b1d49f8_last.pt"
old_weights = "resnet18_scifar10_sgd_joint_inference.pt"

path_new = os.path.join(folder, new_weights)
path_old = os.path.join(folder, old_weights)


def load_state_dict(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    # Se il checkpoint contiene metadati (es. 'model_state_dict' o 'state_dict'), estraiamo solo i pesi
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


# 1. Caricamento pesi
dict_new = load_state_dict(path_new)
dict_old = load_state_dict(path_old)
# Keys da rimuovere
keys_to_remove = {
    "net.kappa_conv",
    "net.w_in",
    "net.w_rec",
    "net.alpha_conv",
    "net.w_out",
}
# 3. Rimuovi le chiavi
removed_count = 0
for key in keys_to_remove:
    if key in dict_new:
        del dict_new[key]
        print(f"✅ Rimuovendo: {key}")
        removed_count += 1
    else:
        print(f"⚠️ Chiave non trovata: {key}")
# 2. Controllo le chiavi (nomi dei layer)
keys_new = set(dict_new.keys())
keys_old = set(dict_old.keys())

if keys_new != keys_old:
    print("❌ La struttura dei modelli NON corrisponde (chiavi/layer differenti)!")
    missing_in_new = keys_old - keys_new
    missing_in_old = keys_new - keys_old
    if missing_in_new:
        print(f"  - Chiavi presenti solo in vecchio: {len(missing_in_new)}")
    if missing_in_old:
        print(f"  - Chiavi presenti solo in nuovo: {len(missing_in_old)},{missing_in_old}")
        
else:
    print("✅ Le strutture del modello e i nomi dei layer corrispondono perfettamente.")

    # 3. Confronto valore dei tensori
    are_equal = True
    mismatched_layers = []

    for key in dict_new.keys():
        tensor_new = dict_new[key]
        tensor_old = dict_old[key]

        # Verifica forma
        if tensor_new.shape != tensor_old.shape:
            print(
                f"❌ Shape differente per {key}: {tensor_new.shape} vs {tensor_old.shape}"
            )
            are_equal = False
            continue

        # Verifica valori numerici (supporta float e interi)
        if tensor_new.dtype.is_floating_point:
            if not torch.allclose(
                tensor_new, tensor_old, atol=1e-6, equal_nan=True
            ):
                are_equal = False
                mismatched_layers.append(key)
        else:
            if not torch.equal(tensor_new, tensor_old):
                are_equal = False
                mismatched_layers.append(key)

    if are_equal:
        print("🎉 I due set di pesi sono ESATTAMENTE IDENTICI!")
    else:
        print(
            f"❌ I pesi DIFFERISCONO in {len(mismatched_layers)} layer su {len(dict_new)}."
        )
        print("Esempi di layer con pesi diversi:", mismatched_layers[:5])