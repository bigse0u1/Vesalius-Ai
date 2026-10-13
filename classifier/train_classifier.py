"""Handle classifier (7 tools + EMPTY) for Verify & Recover.

Reads the data written by the GUI's 분류기 tab (crops/<CLASS>/*.png + labels.jsonl) and
  1. runs 5-fold cross-validation grouped by arrangement (no arrangement in both train
     and test), for color and grayscale input  → accuracy, per-class recall, confusion matrix
  2. trains a final color model on all data and saves it for the GUI.

Usage:
  python classifier/train_classifier.py                      # CV (color + gray) + final model
  python classifier/train_classifier.py --skip-cv            # final model only
  python classifier/train_classifier.py --data ~/xlerobot_classifier_data --epochs 25
"""
import argparse, json, os, random, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

CLASSES = ["GRASPER", "BIPOLAR", "SCISSORS", "CLIPPER", "HOOK", "IRRIGATOR", "SPECIMEN_BAG", "EMPTY"]
INPUT_HW = (224, 112)  # crops are tall (~140×60 px); keep roughly that aspect
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_samples(root):
    """→ list of (crop_path, class_idx, arrangement_id, note)"""
    samples = []
    with open(os.path.join(root, "labels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            for label, rel in zip(r["arrangement"], r["crops"]):
                samples.append((os.path.join(root, rel), CLASSES.index(label), r["arrangement_id"], r.get("note", "")))
    return samples


def preprocess(bgr, gray=False, train=False, rng=None):
    """BGR crop → normalized CHW float tensor (same function is used by the GUI at inference)."""
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if train:
        h, w = img.shape[:2]
        # small shift/rotation/scale: tool nudged inside the slot, ROI not perfectly centered
        ang = rng.uniform(-6, 6); sc = rng.uniform(0.92, 1.08)
        tx, ty = rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.06, 0.06) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc); M[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # brightness/contrast only — no hue jitter, the band color is a real cue
        a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if gray:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.stack([g, g, g], -1)
    img = cv2.resize(img, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class CropDataset(Dataset):
    def __init__(self, samples, gray, train, seed=0):
        self.samples, self.gray, self.train = samples, gray, train
        self.rng = np.random.default_rng(seed)
        self.cache = {p: cv2.imread(p) for p, *_ in samples}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y, *_ = self.samples[i]
        return preprocess(self.cache[p], self.gray, self.train, self.rng), y


def make_model():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def train_model(train_samples, gray, epochs, device, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    ds = CropDataset(train_samples, gray, train=True, seed=seed)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    model = make_model().to(device)
    counts = np.bincount([y for _, y, *_ in train_samples], minlength=len(CLASSES)).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(CLASSES), device=device)  # EMPTY is rare
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, samples, gray, device):
    model.eval()
    dl = DataLoader(CropDataset(samples, gray, train=False), batch_size=64, num_workers=0)
    preds, confs = [], []
    for x, _ in dl:
        p = torch.softmax(model(x.to(device)), 1)
        c, k = p.max(1)
        preds += k.tolist(); confs += c.tolist()
    return np.array(preds), np.array(confs)


def grouped_folds(samples, k, seed):
    groups = sorted({g for _, _, g, _ in samples})
    random.Random(seed).shuffle(groups)
    return [set(groups[i::k]) for i in range(k)]


def run_cv(samples, gray, k, epochs, device, seed=0):
    folds = grouped_folds(samples, k, seed)
    y_true, y_pred, conf, notes = [], [], [], []
    for fi, test_groups in enumerate(folds):
        tr = [s for s in samples if s[2] not in test_groups]
        te = [s for s in samples if s[2] in test_groups]
        t0 = time.time()
        model = train_model(tr, gray, epochs, device, seed + fi)
        p, c = predict(model, te, gray, device)
        y_true += [s[1] for s in te]; y_pred += p.tolist(); conf += c.tolist(); notes += [s[3] for s in te]
        acc = (p == np.array([s[1] for s in te])).mean()
        print(f"  fold {fi+1}/{k}: train {len(tr)} / test {len(te)} crops ({len(test_groups)} arrangements)"
              f" → acc {acc*100:.1f}%  [{time.time()-t0:.0f}s]", flush=True)
    return np.array(y_true), np.array(y_pred), np.array(conf), notes


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def summarize(name, y_true, y_pred, conf, notes):
    n = len(y_true); k = int((y_true == y_pred).sum())
    lo, hi = wilson(k, n)
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    recall = {CLASSES[i]: (int(cm[i, i]), int(cm[i].sum())) for i in range(len(CLASSES))}
    dark = np.array(["어둡" in s for s in notes])
    res = {"condition": name, "n": n, "correct": k, "accuracy": 100 * k / n, "ci95": [lo, hi],
           "per_class_recall": recall, "confusion": cm.tolist(),
           "dark_light": {"n": int(dark.sum()), "correct": int((y_true[dark] == y_pred[dark]).sum())},
           "errors": [{"true": CLASSES[t], "pred": CLASSES[p], "conf": round(float(c), 3)}
                      for t, p, c in zip(y_true, y_pred, conf) if t != p]}
    print(f"\n[{name}] accuracy {k}/{n} = {100*k/n:.1f}% (95% CI {lo:.1f}–{hi:.1f}%)")
    print("  per-class recall: " + "  ".join(f"{c[:4]} {a}/{b}" for c, (a, b) in recall.items()))
    if dark.any():
        print(f"  dark lighting: {res['dark_light']['correct']}/{res['dark_light']['n']}")
    print("  confusion (rows = true, cols = pred):  " + " ".join(f"{c[:4]:>4}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"    {CLASSES[i][:12]:12s} " + " ".join(f"{v:4d}" for v in row))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/xlerobot_classifier_data"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(args.data)
    print(f"{len(samples)} crops, {len({s[2] for s in samples})} arrangements, device {device}")
    print("class counts: " + "  ".join(f"{c}:{n}" for c, n in zip(CLASSES, np.bincount([s[1] for s in samples], minlength=len(CLASSES)))))
    out_dir = os.path.join(args.data, "models"); os.makedirs(out_dir, exist_ok=True)

    if not args.skip_cv:
        results = []
        for name, gray in (("color", False), ("grayscale", True)):
            print(f"\n=== {args.folds}-fold CV grouped by arrangement — {name} ===")
            results.append(summarize(name, *run_cv(samples, gray, args.folds, args.epochs, device)))
        with open(os.path.join(out_dir, "cv_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nsaved {out_dir}/cv_results.json")

    print("\n=== final color model on all data ===")
    model = train_model(samples, gray=False, epochs=args.epochs, device=device, seed=0)
    path = os.path.join(out_dir, "handle_classifier_resnet18_color.pt")
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "input_hw": INPUT_HW,
                "gray": False, "n_train": len(samples)}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
