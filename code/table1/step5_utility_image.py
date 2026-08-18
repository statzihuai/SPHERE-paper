#!/usr/bin/env python3
"""
Table 1, image rows: ImageNet-100 Top-1 on real vs SPHERE-rotated ResNet-50
features, linear probe and MLP, ten seeds each.

Labels are ±1 one-hot (correct class +1, every other class -1) and rotated
jointly with the features, which puts SPHERE in its mixed-angle regime: the ±1
columns are auto-detected and rotated by θ or π-θ with mix_prob=0.75. Both
angles are orthogonal, so the linear probe still satisfies W* = W and its
real/rotated gap is a numerical artefact rather than a loss.

The 5,000 held-out validation images are split 50/50 per class into a
validation half used for checkpoint selection and a test half used only for the
reported number, so the 2,500-image test set is untouched during training.

Note the two probes differ by more than depth: the linear head trains with SGD
at lr=0.1, the MLP with AdamW at 1e-3, and only the MLP has BatchNorm and
dropout. Features are L2-normalised row-wise before either.

    python3 step5_utility_image.py [--n-seeds 10]

Inputs   data/table1/image/imagenet100_{train,validation}_{features,labels}.npy  (step2)
Outputs  code/table1/image_utility.json
Needs    torch, scikit-learn, and the SPHERE implementation.
"""
import _common as C

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES = 100
EPOCHS, LR, BATCH = 30, 0.1, 256
MLP_HIDDEN, DROPOUT = 512, 0.3


class ShallowMLP(nn.Module):
    """Linear -> BatchNorm -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def pm1_onehot(labels, num_classes):
    y = np.full((len(labels), num_classes), -1.0, dtype=np.float32)
    y[np.arange(len(labels)), labels] = 1.0
    return y


def apply_sphere_joint(Z, labels, num_classes, k=C.K, seed=0):
    """Rotate [Z | y_pm1] as one matrix, k passes, and split the result back.

    The mixed-angle scheme is not requested explicitly here — SPHERE detects the
    ±1 label columns and switches to it on its own. step7 has no label columns
    to detect (it rotates the feature block alone) and therefore has to pass
    delta / mix_prob / force_mixed_angle by hand to land on the same rotation.
    """
    j = np.concatenate([Z, pm1_onehot(labels, num_classes)], axis=1)
    for i in range(k):
        np.random.seed(seed + i)
        j = sphere(j.copy(), theta=C.THETA).astype(np.float32)
    d = Z.shape[1]
    Z_sph, y_sph = j[:, :d], j[:, d:]
    err = np.abs(Z_sph.T @ Z_sph - Z.T @ Z).max()
    print(f"  SPHERE joint k={k} seed={seed}  Gram-err(Z)={err:.3e}", flush=True)
    return Z_sph, y_sph


def _l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def train_probe(train_feats, train_y_soft, val_feats, val_labels,
                test_feats, test_labels, device, seed, use_mlp, label):
    """MSE regression onto the rotated ±1 targets; report Top-1 of the best-val checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_loader = DataLoader(
        TensorDataset(torch.FloatTensor(_l2norm(train_feats)),
                      torch.FloatTensor(train_y_soft)),
        batch_size=BATCH, shuffle=True, num_workers=0)
    va_loader = DataLoader(
        TensorDataset(torch.FloatTensor(_l2norm(val_feats)),
                      torch.LongTensor(val_labels.astype(np.int64))),
        batch_size=BATCH, shuffle=False, num_workers=0)

    d = train_feats.shape[1]
    if use_mlp:
        model = ShallowMLP(d, MLP_HIDDEN, NUM_CLASSES).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        model = nn.Linear(d, NUM_CLASSES).to(device)
        optimiser = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val, best_state = -1.0, {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            criterion(model(x), y).backward()
            optimiser.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in va_loader:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += len(y)
            val_top1 = correct / total * 100
            if val_top1 > best_val:
                best_val = val_top1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    te_loader = DataLoader(
        TensorDataset(torch.FloatTensor(_l2norm(test_feats)),
                      torch.LongTensor(test_labels.astype(np.int64))),
        batch_size=BATCH, shuffle=False, num_workers=0)
    c1 = total = 0
    with torch.no_grad():
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            c1 += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    top1 = c1 / total * 100
    print(f"    [{label}] test Top-1={top1:.2f}%", flush=True)
    return top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=C.N_SEEDS)
    ap.add_argument("--in-dir", default=str(C.DATA / "image"))
    ap.add_argument("--out", default=str(C.OUT / "image_utility.json"))
    # Seeds are independent, so a run can be sharded across processes and the
    # per-seed lists merged afterwards.
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="explicit seeds to run; default range(--n-seeds)")
    ap.add_argument("--model-types", nargs="+", default=["linear", "mlp"],
                    choices=["linear", "mlp"])
    # Canonical device is CPU (see C.get_device). "auto"/"mps"/"cuda" are faster
    # but produce numbers that no other machine can match.
    ap.add_argument("--device", default=None, help="cpu (default) | auto | mps | cuda")
    args = ap.parse_args()
    seeds = args.seed_list if args.seed_list is not None else list(range(args.n_seeds))

    d = C.Path(args.in_dir)
    C.require(d / "imagenet100_train_features.npy", "run step2_image_features.py first")
    train_feats = np.load(d / "imagenet100_train_features.npy")
    train_labels = np.load(d / "imagenet100_train_labels.npy")
    all_val_feats = np.load(d / "imagenet100_validation_features.npy")
    all_val_labels = np.load(d / "imagenet100_validation_labels.npy")

    # Stratified 50/50 val/test split of the held-out set, fixed across seeds.
    rng = np.random.default_rng(seed=0)
    val_idx, test_idx = [], []
    for cls in np.unique(all_val_labels):
        ci = rng.permutation(np.where(all_val_labels == cls)[0])
        half = len(ci) // 2
        val_idx.extend(ci[:half].tolist())
        test_idx.extend(ci[half:].tolist())
    val_idx, test_idx = np.array(val_idx), np.array(test_idx)
    val_feats, val_labels = all_val_feats[val_idx], all_val_labels[val_idx]
    test_feats, test_labels = all_val_feats[test_idx], all_val_labels[test_idx]

    device = C.get_device(args.device)
    print(f"device={device} train={train_feats.shape} val={val_feats.shape} "
          f"test={test_feats.shape} seeds={seeds}", flush=True)

    y_real = pm1_onehot(train_labels, NUM_CLASSES)
    model_results = {}
    for model_type in args.model_types:
        use_mlp = model_type == "mlp"
        real = []
        for seed in seeds:
            real.append(train_probe(train_feats, y_real, val_feats, val_labels,
                                    test_feats, test_labels, device, seed, use_mlp,
                                    f"real-{model_type}[s{seed}]"))
        sph = []
        for seed in seeds:
            Z_sph, y_sph = apply_sphere_joint(train_feats, train_labels, NUM_CLASSES,
                                              k=C.K, seed=seed)
            sph.append(train_probe(Z_sph, y_sph, val_feats, val_labels,
                                   test_feats, test_labels, device, seed, use_mlp,
                                   f"sphere-{model_type}[s{seed}]"))
        model_results[model_type] = {"real": C.stat_leaf(real), "sphere": C.stat_leaf(sph)}
        print(f"[{model_type}] real={np.mean(real):.2f}% sphere={np.mean(sph):.2f}%", flush=True)

    C.write_json(args.out, {
        "dataset": "ImageNet-100 (ResNet-50 features)",
        "metric": "Top-1 accuracy (%)",
        "eval_set": f"test (n={len(test_feats)})",
        "protocol": "joint [Z | y_pm1] rotation of the TRAIN split only; val/test raw; "
                    "checkpoint on real val Top-1; MSE on rotated +/-1 targets",
        "k": C.K,
        "theta": float(C.THETA),
        "config": {"epochs": EPOCHS, "lr": LR, "batch": BATCH, "seeds": seeds,
                   "mlp_hidden": MLP_HIDDEN, "device": str(device)},
        "n_train": int(len(train_feats)), "n_val": int(len(val_feats)),
        "n_test": int(len(test_feats)), "feature_dim": int(train_feats.shape[1]),
        "model_results": model_results,
    })


if __name__ == "__main__":
    main()
