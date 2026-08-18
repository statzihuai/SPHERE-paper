#!/usr/bin/env python3
"""
Table 1, text rows: Dreaddit stress classification on real vs SPHERE-rotated
GTR embeddings, linear probe and MLP, ten seeds each.

The protocol has three properties that the numbers depend on:

  * SPHERE rotates the TRAIN split only. Validation and test stay raw. That is
    the deployment story being tested — a recipient trains on rotated data and
    is evaluated against real held-out records — and rotating the test split
    instead would scramble the row-label pairing and produce an artifact.
  * Labels are co-rotated with the features. [Z | y_pm1] is rotated as one
    matrix, so the probe trains with MSE against a continuous y*, not against
    the original binary label. For a linear model this is exactly why W* = W.
  * Checkpoint selection uses real validation AUC on binary labels, so the
    rotated and unrotated arms are selected by the same rule.

    python3 step4_utility_text.py [--n-seeds 10]

Inputs   data/table1/text/{emb_train_raw,emb_test_raw,labels_train,labels_test}.npy  (step1)
Outputs  code/table1/text_utility.json
Needs    torch, scikit-learn, and the SPHERE implementation. Runs on CPU by
         design — see README, "Reproducibility".
"""
import _common as C

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EPOCHS, LR, BATCH, VAL_FRAC, MLP_HIDDEN = 100, 1e-3, 64, 0.15, 128
DEVICE = torch.device("cpu")


class ShallowMLP(nn.Module):
    """Linear -> ReLU -> Linear."""

    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def apply_sphere_joint(Z, y_labels, k=C.K, seed=0):
    """Rotate [Z | y_pm1] as one matrix, k passes, and split the result back.

    Binary labels are re-coded 0 -> -1, 1 -> +1 before rotation. Each pass
    re-seeds at seed+i and rotates the already-rotated matrix, so k>1 compounds
    rather than restarting.
    """
    n = len(Z)
    y_col = (2 * y_labels.astype(np.float32) - 1).reshape(n, 1)
    j = np.concatenate([Z, y_col], axis=1)
    for i in range(k):
        np.random.seed(seed + i)
        j = sphere(j.copy(), theta=C.THETA).astype(np.float32)
    return j[:, :-1], j[:, -1]


def train_probe(train_Z, train_y_soft, val_Z, val_y, test_Z, test_y,
                seed, use_mlp, label):
    """MSE regression onto the continuous y*; report test AUC of the best-val checkpoint."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    d = train_Z.shape[1]
    model = (ShallowMLP(d, MLP_HIDDEN, 1) if use_mlp else nn.Linear(d, 1)).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_Z), torch.FloatTensor(train_y_soft)),
        batch_size=BATCH, shuffle=True)

    best_val_auc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            criterion(model(xb).squeeze(), yb).backward()
            optimiser.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                va = torch.sigmoid(model(torch.FloatTensor(val_Z).to(DEVICE))).squeeze().cpu().numpy()
            try:
                val_auc = roc_auc_score(val_y, va)
            except ValueError:
                val_auc = 0.5
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te = torch.sigmoid(model(torch.FloatTensor(test_Z).to(DEVICE))).squeeze().cpu().numpy()
    auc = float(roc_auc_score(test_y, te))
    acc = float(accuracy_score(test_y, (te >= 0.5).astype(int)))
    print(f"    [{label}] test AUC={auc:.4f} acc={acc:.4f}", flush=True)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=C.N_SEEDS)
    ap.add_argument("--in-dir", default=str(C.DATA / "text"))
    ap.add_argument("--out", default=str(C.OUT / "text_utility.json"))
    args = ap.parse_args()

    d = C.Path(args.in_dir)
    C.require(d / "emb_train_raw.npy", "run step1_text_embed.py first")
    Ztr = np.load(d / "emb_train_raw.npy").astype(np.float32)
    Zte = np.load(d / "emb_test_raw.npy").astype(np.float32)
    tr_y = np.load(d / "labels_train.npy")
    te_y = np.load(d / "labels_test.npy")

    # Fixed validation split, drawn once and shared by every seed and both arms,
    # so real and SPHERE are selected against exactly the same held-out rows.
    rng = np.random.default_rng(seed=0)
    n_val = max(1, int(len(Ztr) * VAL_FRAC))
    val_idx = rng.choice(len(Ztr), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(Ztr)), val_idx)
    tr_Z, tr_yb = Ztr[train_idx], tr_y[train_idx]
    va_Z, va_yb = Ztr[val_idx], tr_y[val_idx]
    print(f"train={len(tr_Z)} val={len(va_Z)} test={len(Zte)} d={tr_Z.shape[1]}", flush=True)

    model_results = {}
    for model_type, use_mlp in (("linear", False), ("mlp", True)):
        real = []
        for seed in range(args.n_seeds):
            real.append(train_probe(tr_Z, 2 * tr_yb.astype(np.float32) - 1,
                                    va_Z, va_yb, Zte, te_y, seed, use_mlp,
                                    f"real-{model_type}[s{seed}]"))
        sph = []
        for seed in range(args.n_seeds):
            Zk, yk = apply_sphere_joint(tr_Z, tr_yb, k=C.K, seed=seed)
            sph.append(train_probe(Zk, yk, va_Z, va_yb, Zte, te_y, seed, use_mlp,
                                   f"sphere-{model_type}[s{seed}]"))
        model_results[model_type] = {"real": C.stat_leaf(real), "sphere": C.stat_leaf(sph)}
        print(f"[{model_type}] real={np.mean(real):.4f} sphere={np.mean(sph):.4f}", flush=True)

    C.write_json(args.out, {
        "dataset": "Dreaddit",
        "embedding": "GTR-base 768-d (vec2text space)",
        "metric": "AUC-ROC",
        "eval_set": f"test (n={len(Zte)})",
        "protocol": "joint [Z | y_pm1] rotation of the TRAIN split only; "
                    "val/test raw; checkpoint on real val AUC; MSE on continuous y*",
        "k": C.K,
        "theta": float(C.THETA),
        "config": {"epochs": EPOCHS, "lr": LR, "batch": BATCH, "n_seeds": args.n_seeds,
                   "val_frac": VAL_FRAC, "mlp_hidden": MLP_HIDDEN, "device": str(DEVICE)},
        "n_train": int(len(tr_Z)), "n_val": int(len(va_Z)), "n_test": int(len(Zte)),
        "model_results": model_results,
    })


if __name__ == "__main__":
    main()
