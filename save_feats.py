"""
Estrae le 512 features dal penultimo layer di ResNet-18 (Mammoth)
su seq-cifar10 e salva un dataset cifar10_feats.

Output:
    data/cifar10_feats/
        train_feats.npz   -> { 'features': (50000, 512), 'labels': (50000,) }
        test_feats.npz    -> {  'features': (10000, 512), 'labels': (10000,) }

Uso:
    python extract_cifar10_feats.py
    python extract_cifar10_feats.py --checkpoint ./checkpoints/altro.pt --batch_size 512
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ── Import backbone da Mammoth ────────────────────────────────────────────────
# Assicurati di lanciare lo script dalla root di mammoth, oppure aggiungi il
# path manualmente:
#   import sys; sys.path.insert(0, "/percorso/a/mammoth")
try:
    from backbone.ResNetBlock import resnet18cus
    MAMMOTH_BACKBONE = True
    print('[INFO] backbone.ResNet18 trovata. Uso Mammoth ResNet-18.')
except ImportError:
    MAMMOTH_BACKBONE = False
    print("[WARN] backbone.ResNet18 non trovata. Uso torchvision ResNet-18 come fallback.")
    from torchvision.models import resnet18 as tv_resnet18


# ── Normalizzazione CIFAR-10 (stessi valori usati da Mammoth/seq-cifar10) ────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2615)


def strip_prefix(state_dict: OrderedDict, prefix: str = "net.") -> OrderedDict:
    """Rimuove il prefisso 'net.' aggiunto da Mammoth quando wrappa la backbone."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new_sd[new_key] = v
    return new_sd


def load_backbone(checkpoint_path: str, num_classes: int = 10, device: str = "cpu"):
    """
    Carica la backbone ResNet-18 con i pesi dal checkpoint Mammoth.
    Restituisce il modello in eval mode.
    """
    print(f"  Caricamento checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location=device)

    # Mammoth può wrappare lo state_dict sotto 'model'
    state = cp.get("model", cp)
    state = strip_prefix(state, prefix="net.")

    if MAMMOTH_BACKBONE:
        net = resnet18cus( srnn_hidden_size=150,
            n_in=512,
            n_out=10,
            n_t=100,
            thr=0.6,
            tau_m=30,
            tau_o=100,
            b_o=0.0,
            gamma=0.5,
            dt=1.,
            classif=True,
            keep_sparsity=False,
            syn_scale=False,
            sparsity= [0.0,0.0,0.0],
            lr_layer=[0.5,0.5,0.5] ,
            f_target=1,
            c_reg =0.001,
            lr_win=10,
            net_state="./checkpoints/resnet18_scifar10_sgd_joint_inference.pt",
            xi=20,
            num_classes=10, 
            num_filters= 64)
    else:
        net = tv_resnet18(num_classes=num_classes)

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] missing keys  ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  [WARN] unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")

    net.to(device)
    net.eval()
    print("  Backbone caricata correttamente.")
    return net


def get_feature_extractor(net: nn.Module):
    """
    Restituisce una funzione che estrae le feature dal penultimo layer.
    - Se la backbone è Mammoth ResNet-18: usa net.features(x)  (metodo nativo)
    - Fallback torchvision: rimuove l'ultimo fc e usa avgpool output
    """
    if MAMMOTH_BACKBONE and hasattr(net, "features"):
        # Metodo nativo Mammoth: restituisce embedding 512-dim prima del classificatore
        # return lambda x: net.features(x)
        return lambda x, returnt='features': net(x, returnt=returnt)
    else:
        # Fallback: hook sull'avgpool di torchvision ResNet
        feature_holder = {}
        def hook(module, input, output):
            feature_holder["feat"] = output.flatten(1)
        net.avgpool.register_forward_hook(hook)
        def extractor(x, returnt='features'):
            net(x)   # forward completo, il hook cattura avgpool
            return feature_holder["feat"]
        return extractor


def extract_split(
    net: nn.Module,
    extractor,
    is_train: bool,
    batch_size: int,
    device: str,
    data_root: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Estrae features e labels per train o test set."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    dataset = datasets.CIFAR10(
        root=data_root,
        train=is_train,
        download=False,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device != "cpu"),
    )

    all_feats, all_labels = [], []

    with torch.no_grad():
        k=0
        for imgs, lbls in tqdm(loader, desc="  train" if is_train else "  test ", leave=False):
            imgs = imgs.to(device)
            feats = extractor(imgs, returnt='features')          # (B, 512)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(lbls.numpy())
            # k+=1
            # if k>20:
            #     break
    
    features = np.concatenate(all_feats,  axis=0)   # (N, 512)
    labels   = np.concatenate(all_labels, axis=0)   # (N,)
    print(f"  Estrazione completata: {features.shape[0]} campioni, {features.shape[1]} features, labels {np.unique(labels)}")
    return features, labels


def save_split(features: np.ndarray, labels: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, features=features, labels=labels)
    print(f"  Salvato: {path}  —  features {features.shape}, labels {labels.shape}")


def verify(path: str) -> None:
    """Quick sanity check sul file salvato."""
    data = np.load(path)
    f, l = data["features"], data["labels"]
    assert f.shape[1] == 512, f"Atteso 512 features, trovato {f.shape[1]}"
    assert f.shape[0] == l.shape[0], "Mismatch features/labels"
    classes, counts = np.unique(l, return_counts=True)
    print(f"  Verifica OK — {f.shape[0]} campioni, {len(classes)} classi: {dict(zip(classes.tolist(), counts.tolist()))}")


def main():
    parser = argparse.ArgumentParser(description="Estrai CIFAR-10 features da ResNet-18 Mammoth")
    parser.add_argument("--checkpoint",  default="./checkpoints/resnet18_scifar10_sgd_joint_inference.pt")
    parser.add_argument("--out_dir",     default="./data/cifar10_feats")
    parser.add_argument("--data_root",   default="./data")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  CIFAR-10 Feature Extraction — ResNet-18 (Mammoth)")
    print(f"{'='*55}")
    print(f"  Device     : {args.device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Output dir : {args.out_dir}")
    print(f"  Batch size : {args.batch_size}")
    print()

    # ── Carica rete ──────────────────────────────────────────────────────────
    net = load_backbone(args.checkpoint, args.num_classes, args.device)
    extractor = get_feature_extractor(net)

    # ── Train set ─────────────────────────────────────────────────────────────
    print("Estrazione train set...")
    tr_feats, tr_labels = extract_split(net, extractor, is_train=True,
                                        batch_size=args.batch_size,
                                        device=args.device, data_root=args.data_root)
    train_path = os.path.join(args.out_dir, "train_feats.npz")
    save_split(tr_feats, tr_labels, train_path)
    verify(train_path)

    # ── Test set ──────────────────────────────────────────────────────────────
    print("Estrazione test set...")
    te_feats, te_labels = extract_split(net, extractor, is_train=False,
                                        batch_size=args.batch_size,
                                        device=args.device, data_root=args.data_root)
    test_path = os.path.join(args.out_dir, "test_feats.npz")
    save_split(te_feats, te_labels, test_path)
    verify(test_path)

    print(f"\n✓ Fatto. Dataset cifar10_feats salvato in: {args.out_dir}\n")


if __name__ == "__main__":
    main()