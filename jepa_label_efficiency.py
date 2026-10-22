"""
Label efficiency of frozen self-supervised encoders: I-JEPA vs MAE vs DINOv2.

Question: when the encoder is frozen and only a linear classifier is trained on
1%, 5%, 10% and 100% of the CIFAR-100 labels, how does a JEPA-pretrained
encoder compare with a reconstruction-pretrained (MAE) and a joint-embedding
(DINOv2) encoder?

HOW TO RUN (free Google Colab):
  1. Runtime > Change runtime type > T4 GPU
  2. In a cell:  !pip install -q transformers torchvision pandas
  3. Upload this file, then:
       !python jepa_label_efficiency.py --smoke      # 2-minute pipeline check
       !python jepa_label_efficiency.py              # full run
  4. Send back the files results/summary.csv, results/results.csv and results/meta.json

Full run time on a T4: roughly 30-60 minutes (feature extraction dominates).
Features are cached in results/, so a re-run skips extraction.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModel, IJepaModel, ViTMAEModel

# name -> (Hugging Face checkpoint, loader kind)
MODELS = {
    "I-JEPA ViT-H/14": ("facebook/ijepa_vith14_1k", "ijepa"),
    "MAE ViT-H/14": ("facebook/vit-mae-huge", "mae"),
    "DINOv2 ViT-L/14": ("facebook/dinov2-large", "dino"),
}
FRACTIONS = [0.01, 0.05, 0.10, 1.0]
SEEDS = [0, 1, 2]
PROBE_STEPS = 1500
PROBE_LR = 1e-3
PROBE_WD = 1e-4


class TransformedDataset(Dataset):
    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        return self.transform(img), y


def load_model(name, kind, device):
    if kind == "ijepa":
        model = IJepaModel.from_pretrained(name)
    elif kind == "mae":
        # mask_ratio=0 keeps every patch; mean pooling ignores the shuffled order
        model = ViTMAEModel.from_pretrained(name, mask_ratio=0.0)
    else:
        model = AutoModel.from_pretrained(name)
    return model.half().to(device).eval()


@torch.no_grad()
def extract(model, kind, loader, device):
    feats, labels = [], []
    for x, y in loader:
        out = model(pixel_values=x.to(device).half()).last_hidden_state
        if kind in ("mae", "dino"):
            out = out[:, 1:]  # drop the CLS token so all models use mean-pooled patch tokens
        feats.append(out.float().mean(dim=1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def get_features(label, name, kind, args, device, train_ds, test_ds):
    tag = label.replace(" ", "_").replace("/", "")
    path = os.path.join(args.out, f"feats_{tag}{'_smoke' if args.smoke else ''}.pt")
    if os.path.exists(path):
        return torch.load(path)
    proc = AutoImageProcessor.from_pretrained(name)
    tf = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(proc.image_mean, proc.image_std),
    ])
    model = load_model(name, kind, device)
    t0 = time.time()
    mk = lambda ds: DataLoader(TransformedDataset(ds, tf), batch_size=args.batch,
                               shuffle=False, num_workers=2)
    Xtr, ytr = extract(model, kind, mk(train_ds), device)
    Xte, yte = extract(model, kind, mk(test_ds), device)
    print(f"  extracted {label} in {time.time() - t0:.0f}s, feature dim {Xtr.shape[1]}")
    torch.save((Xtr, ytr, Xte, yte), path)
    del model
    torch.cuda.empty_cache()
    return Xtr, ytr, Xte, yte


def stratified_subset(y, fraction, seed):
    if fraction >= 1.0:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        members = np.where(y == c)[0]
        k = max(1, int(round(fraction * len(members))))
        idx.extend(rng.choice(members, k, replace=False))
    return np.array(idx)


def train_probe(Xtr, ytr, Xte, yte, seed, device):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = (t.to(device) for t in (Xtr, ytr, Xte, yte))
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    lin = nn.Linear(Xtr.shape[1], int(ytr.max()) + 1).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PROBE_STEPS)
    n, bs = len(Xtr), min(1024, len(Xtr))
    for _ in range(PROBE_STEPS):
        idx = torch.randint(0, n, (bs,), device=device)
        loss = F.cross_entropy(lin(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return (lin(Xte).argmax(1) == yte).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--smoke", action="store_true", help="tiny subset to test the pipeline")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found. Use a GPU runtime; CPU will be far too slow.")

    train_ds = datasets.CIFAR100("data", train=True, download=True)
    test_ds = datasets.CIFAR100("data", train=False, download=True)
    if args.smoke:
        train_ds = torch.utils.data.Subset(train_ds, range(0, 50000, 25))  # 2,000 images
        test_ds = torch.utils.data.Subset(test_ds, range(0, 10000, 20))    # 500 images

    rows = []
    for label, (name, kind) in MODELS.items():
        print(f"[{label}] {name}")
        try:
            Xtr, ytr, Xte, yte = get_features(label, name, kind, args, device, train_ds, test_ds)
        except Exception as e:  # missing checkpoint, out of memory, etc.
            print(f"  SKIPPED {label}: {e}")
            continue
        ytr_np = ytr.numpy()
        for frac in FRACTIONS:
            for seed in SEEDS:
                idx = stratified_subset(ytr_np, frac, seed)
                acc = train_probe(Xtr[idx], ytr[idx], Xte, yte, seed, device)
                rows.append(dict(model=label, label_fraction=frac, seed=seed,
                                 n_train=len(idx), top1=acc))
                print(f"  {int(frac * 100):>3}% labels seed {seed}: {acc:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    summary = (df.groupby(["model", "label_fraction"])["top1"]
                 .agg(["mean", "std"]).round(2).reset_index())
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\n", summary.to_string(index=False))

    meta = dict(
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        torch=torch.__version__,
        probe=dict(steps=PROBE_STEPS, lr=PROBE_LR, weight_decay=PROBE_WD),
        seeds=SEEDS, dataset="CIFAR-100 (upsampled to 224x224)", smoke=args.smoke,
        models={k: v[0] for k, v in MODELS.items()},
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
