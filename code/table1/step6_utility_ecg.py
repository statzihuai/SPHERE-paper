#!/usr/bin/env python3
"""
Table 1, ECG row: PTB myocardial-infarction detection on real vs SPHERE-rotated
waveforms with a temporal Transformer, ten seeds.

Each seed draws its own patient-level split (70/15/15, stratified on the
patient's worst label), so the ten runs vary the split as well as the
initialisation. That is why this row carries a visible SEM where the other two
do not: the test half is only about 60 recordings, and one or two borderline
patients move the AUC by several points. The published 4.4% real-vs-rotated gap
should be read against that spread, not as a systematic loss.

Waveforms are z-scored per lead using training statistics only, then flattened
to 12,000 dimensions and rotated jointly with the ±1 label before being
reshaped back to (n, 1000, 12). Each synthetic row is therefore a Givens
mixture of two real patients across all time-lead dimensions at once, which
preserves within-recording temporal structure.

    python3 step6_utility_ecg.py [--n-seeds 10]

Inputs   data/table1/ecg/{ptb_waveforms,ptb_labels,ptb_patient_ids}.npy  (step3)
Outputs  code/table1/ecg_utility.json
Needs    torch, scikit-learn, and the SPHERE implementation.
"""
import _common as C

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEQ_LEN, N_LEADS = 1000, 12
EPOCHS, LR, BATCH = 50, 1e-3, 32
VAL_FRAC, TEST_FRAC = 0.15, 0.15


class TemporalTransformerClassifier(nn.Module):
    """Pre-LN Transformer encoder, mean-pooled over time, single logit out.

    Linear(12 -> 128) + learned positional embedding, 2 layers, 4 heads,
    FFN 512, dropout 0.1.
    """

    def __init__(self, n_features=N_LEADS, d_model=128, nhead=4, num_layers=2,
                 seq_len=SEQ_LEN, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_emb
        h = self.encoder(h)
        return self.classifier(h.mean(dim=1)).squeeze(-1)


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def patient_split(waveforms, labels, patient_ids, seed):
    """Stratified split on patients, not recordings — several patients recur."""
    from sklearn.model_selection import train_test_split

    pids = list(np.unique(patient_ids))
    pid_label = {p: int(labels[patient_ids == p].max()) for p in pids}
    plabels = [pid_label[p] for p in pids]

    train_pids, test_pids = train_test_split(
        pids, test_size=TEST_FRAC, stratify=plabels, random_state=seed)
    train_pids, val_pids = train_test_split(
        train_pids, test_size=VAL_FRAC / (1.0 - TEST_FRAC),
        stratify=[pid_label[p] for p in train_pids], random_state=seed)

    tr = np.isin(patient_ids, list(train_pids))
    va = np.isin(patient_ids, list(val_pids))
    te = np.isin(patient_ids, list(test_pids))
    return (waveforms[tr], labels[tr], waveforms[va], labels[va],
            waveforms[te], labels[te])


def normalise(X_train, X_val, X_test):
    """Per-lead z-score from training statistics only."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (((X_train - mean) / std).astype(np.float32),
            ((X_val - mean) / std).astype(np.float32),
            ((X_test - mean) / std).astype(np.float32))


def apply_sphere_joint_ecg(X, y, k=C.K, seed=0):
    """Flatten (n, T, C) -> (n, T*C), rotate jointly with y_pm1, reshape back."""
    n, T, Ch = X.shape
    X_flat = X.reshape(n, T * Ch).astype(np.float32)
    y_col = (2 * y.astype(np.float32) - 1).reshape(n, 1)
    j = np.concatenate([X_flat, y_col], axis=1)
    for i in range(k):
        np.random.seed(seed + i)
        j = sphere(j.copy(), theta=C.THETA).astype(np.float32)
    err = np.abs(j[:, :-1].T @ j[:, :-1] - X_flat.T @ X_flat).max()
    print(f"  SPHERE joint k={k} seed={seed}  Gram-err(X)={err:.3e}", flush=True)
    return j[:, :-1].reshape(n, T, Ch), j[:, -1]


def train_transformer(X_train, y_train_soft, X_val, y_val, device, seed, label):
    """MSE regression onto the continuous y*; checkpoint on real val AUC."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TemporalTransformerClassifier().to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-5)

    tr_loader = DataLoader(ECGDataset(X_train, y_train_soft), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH, shuffle=False)

    best_val_auc, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()

        model.eval()
        probs, gts = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                probs.append(torch.sigmoid(model(xb.to(device)).cpu()).numpy())
                gts.append(yb.numpy())
        try:
            val_auc = roc_auc_score(np.concatenate(gts), np.concatenate(probs))
        except ValueError:
            val_auc = 0.5
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model, X_test, y_test, device, batch=64):
    from sklearn.metrics import roc_auc_score
    loader = DataLoader(ECGDataset(X_test, y_test), batch_size=batch, shuffle=False)
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.sigmoid(model(xb.to(device)).cpu()).numpy())
            gts.append(yb.numpy())
    return float(roc_auc_score(np.concatenate(gts), np.concatenate(probs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=C.N_SEEDS)
    ap.add_argument("--in-dir", default=str(C.DATA / "ecg"))
    ap.add_argument("--out", default=str(C.OUT / "ecg_utility.json"))
    # Seeds are independent, so a run can be sharded across processes and the
    # per-seed lists merged afterwards (merge_ecg_shards.py).
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="explicit seeds to run; default range(--n-seeds)")
    # Canonical device is CPU (see C.get_device). "auto"/"mps"/"cuda" are faster
    # but produce numbers that no other machine can match.
    ap.add_argument("--device", default=None, help="cpu (default) | auto | mps | cuda")
    args = ap.parse_args()
    seeds = args.seed_list if args.seed_list is not None else list(range(args.n_seeds))

    d = C.Path(args.in_dir)
    C.require(d / "ptb_waveforms.npy", "run step3_ecg_waveforms.py first")
    waveforms = np.load(d / "ptb_waveforms.npy")
    labels = np.load(d / "ptb_labels.npy")
    patient_ids = np.load(d / "ptb_patient_ids.npy", allow_pickle=True)

    device = C.get_device(args.device)
    n_mi, n_hc = int((labels == 1).sum()), int((labels == 0).sum())
    print(f"device={device} recordings={len(waveforms)} MI={n_mi} HC={n_hc} "
          f"seeds={seeds}", flush=True)

    real, sph, sizes = [], [], []
    for seed in seeds:
        X_tr, y_tr, X_va, y_va, X_te, y_te = patient_split(
            waveforms, labels, patient_ids, seed=seed)
        X_tr, X_va, X_te = normalise(X_tr, X_va, X_te)
        sizes.append((len(X_tr), len(X_va), len(X_te)))

        m = train_transformer(X_tr, 2 * y_tr.astype(np.float32) - 1, X_va, y_va,
                              device, seed, f"real[s{seed}]")
        auc = evaluate(m, X_te, y_te, device)
        print(f"  real s{seed}: AUC={auc:.4f}", flush=True)
        real.append(auc)
        del m

        X_sph, y_sph = apply_sphere_joint_ecg(X_tr, y_tr, k=C.K, seed=seed)
        m = train_transformer(X_sph, y_sph, X_va, y_va, device, seed, f"sphere[s{seed}]")
        auc = evaluate(m, X_te, y_te, device)
        print(f"  sphere s{seed}: AUC={auc:.4f}", flush=True)
        sph.append(auc)
        del m

    print(f"[transformer] real={np.mean(real):.4f} sphere={np.mean(sph):.4f}", flush=True)
    avg = lambda i: int(round(np.mean([s[i] for s in sizes])))

    C.write_json(args.out, {
        "dataset": "PTB Diagnostic ECG Database",
        "task": "Binary classification (MI vs HC)",
        "metric": "AUC-ROC",
        "eval_set": f"test (n~{avg(2)})",
        "protocol": "joint [X_flat | y_pm1] rotation of the TRAIN split only; val/test raw; "
                    "checkpoint on real val AUC; per-seed patient-level split",
        "k": C.K,
        "theta": float(C.THETA),
        "config": {"epochs": EPOCHS, "lr": LR, "batch": BATCH, "seeds": seeds,
                   "device": str(device)},
        "n_recordings": int(len(waveforms)), "n_mi": n_mi, "n_hc": n_hc,
        "avg_n_train": avg(0), "avg_n_val": avg(1), "avg_n_test": avg(2),
        "model_results": {"transformer": {"real": C.stat_leaf(real),
                                          "sphere": C.stat_leaf(sph)}},
    })


if __name__ == "__main__":
    main()
