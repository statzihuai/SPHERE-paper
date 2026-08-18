#!/usr/bin/env python3
"""
Extract ResNet-50 features for ImageNet-100 — the matrix Table 1's image row is
computed on, and the one its privacy column attacks.

ImageNet-100 here means the first 100 class ids of ImageNet-1K, streamed from
HuggingFace so that no 150 GB local copy is needed. Features are the 2048-d
avgpool output of torchvision's IMAGENET1K_V1 ResNet-50 with the classifier
head removed, under the standard Resize(256) / CenterCrop(224) / ImageNet
normalisation. The backbone is frozen; nothing here is trained.

Streaming order is part of the output, not an implementation detail. step7
reconstructs its 200-row privacy subset (10 images per class for classes 0-19)
by replaying the order of the saved validation labels, so the labels file must
stay in the order the shards were traversed. Do not sort it.

    python3 step2_image_features.py [--num-classes 100] [--batch-size 256]

Inputs   HuggingFace benjamin-paine/imagenet-1k-256x256 (train + validation)
Outputs  data/table1/image/imagenet100_train_features.npy       (~129k, 2048) float32
         data/table1/image/imagenet100_train_labels.npy         (~129k,)      int32
         data/table1/image/imagenet100_validation_features.npy  (5000, 2048)  float32
         data/table1/image/imagenet100_validation_labels.npy    (5000,)       int32
Needs    torch + torchvision + datasets; a GPU/MPS device makes this hours
         rather than a day. Existing outputs are reused, so the step is
         restartable per split.
"""
import _common as C

import argparse
import time

import numpy as np

HF_DATASET = "benjamin-paine/imagenet-1k-256x256"


def build_backbone(device):
    """ResNet-50 (IMAGENET1K_V1) with the fc head replaced by Identity."""
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()          # 2048-d avgpool output
    model.eval()
    return model.to(device)


def extract_split(backbone, device, num_classes, split, batch_size, out_dir):
    import torch
    import torchvision.transforms as T
    from datasets import load_dataset

    feat_path = out_dir / f"imagenet100_{split}_features.npy"
    label_path = out_dir / f"imagenet100_{split}_labels.npy"
    if feat_path.exists() and label_path.exists():
        print(f"  {split}: reusing {feat_path.name}", flush=True)
        return np.load(feat_path), np.load(label_path)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = load_dataset(HF_DATASET, split=split, streaming=True).filter(
        lambda ex: ex["label"] < num_classes)

    feat_list, label_list = [], []
    batch_imgs, batch_lbls = [], []
    n_done = 0
    t0 = time.perf_counter()

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            feats = backbone(torch.stack(batch_imgs).to(device)).cpu().numpy().astype(np.float32)
        feat_list.append(feats)
        label_list.extend(batch_lbls)
        batch_imgs.clear()
        batch_lbls.clear()

    # Exhaustive scan of the split, exactly as the original extractor does. An
    # early exit on the first out-of-range label would be wrong: the training
    # shards happen to be class-ordered, but the validation split is shuffled
    # across all 1,000 classes, so stopping early there collects almost nothing.
    for ex in ds:
        label = int(ex["label"])
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        batch_imgs.append(transform(img))
        batch_lbls.append(label)
        n_done += 1
        if len(batch_imgs) == batch_size:
            flush()
            if n_done % 5000 == 0:
                rate = n_done / (time.perf_counter() - t0)
                print(f"    {split} {n_done:>7} images ({rate:.0f} img/s)", flush=True)
    flush()

    features = np.concatenate(feat_list, axis=0)
    labels = np.array(label_list, dtype=np.int32)
    print(f"  {split}: {features.shape} in {time.perf_counter()-t0:.0f}s", flush=True)

    np.save(feat_path, features)
    np.save(label_path, labels)
    return features, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default=str(C.DATA / "image"))
    ap.add_argument("--device", default=None, help="cpu | auto | mps | cuda")
    args = ap.parse_args()

    out_dir = C.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = C.get_device(args.device)
    print(f"device={device}", flush=True)

    backbone = build_backbone(device)
    for split in ("train", "validation"):
        extract_split(backbone, device, args.num_classes, split,
                      args.batch_size, out_dir)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
