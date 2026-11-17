import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from metrics import RetinalLoss, compute_metrics, compute_pos_weight, print_metrics, split_train_val
from retfound_model import RETFoundNetwork

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_FILE = REPO_ROOT / "weights" / "RETFound_mae_natureCFP.pth"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TrainingImages(Dataset):
    def __init__(self, data, indices, augment):
        self.data = data
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.data.get_item(self.indices[i], augment=self.augment)


def make_optimizer(network, lr, layer_decay=0.65, weight_decay=0.05):
    num_layers = len(network.backbone.blocks)

    def layer_number(name):
        if name.startswith("backbone."):
            part = name[len("backbone."):]
            if part in ("cls_token", "pos_embed") or part.startswith("patch_embed"):
                return 0
            if part.startswith("blocks."):
                return int(part.split(".")[1]) + 1
        return num_layers + 1

    groups = {}
    for name, param in network.named_parameters():
        layer = layer_number(name)
        no_decay = param.ndim == 1 or name.endswith(("cls_token", "pos_embed"))
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * layer_decay ** (num_layers + 1 - layer),
                           "weight_decay": 0.0 if no_decay else weight_decay}
        groups[key]["params"].append(param)
    return torch.optim.AdamW(list(groups.values()))


def make_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs=1):
    warmup_steps = steps_per_epoch * warmup_epochs
    total_steps = steps_per_epoch * epochs

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    def __init__(self, network, loss_fn, optimizer, device, save_dir, scheduler=None):
        self.network = network.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.best_score = -1.0
        self.start_epoch = 1

    def train_one_epoch(self, loader):
        self.network.train()
        total_loss = 0.0
        for step, (images, grades, lesions) in enumerate(loader, start=1):
            images, grades, lesions = images.to(self.device), grades.to(self.device), lesions.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                lesion_logits, grade_logits = self.network(images)
            loss, _, _ = self.loss_fn(lesion_logits.float(), grade_logits.float(), lesions, grades)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            if step % 10 == 0 or step == len(loader):
                print(f"    batch {step}/{len(loader)}  loss {loss.item():.4f}")
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.network.eval()
        total_loss = 0.0
        all_scores, all_lesions, all_grade_preds, all_grades = [], [], [], []
        with torch.no_grad():
            for images, grades, lesions in loader:
                images, grades, lesions = images.to(self.device), grades.to(self.device), lesions.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    lesion_logits, grade_logits = self.network(images)
                lesion_logits, grade_logits = lesion_logits.float(), grade_logits.float()
                loss, _, _ = self.loss_fn(lesion_logits, grade_logits, lesions, grades)
                total_loss += loss.item()

                all_scores.append(torch.sigmoid(lesion_logits).cpu())
                all_lesions.append(lesions.cpu())
                all_grade_preds.append(grade_logits.argmax(dim=1).cpu())
                all_grades.append(grades.cpu())

        metrics = compute_metrics(torch.cat(all_scores), torch.cat(all_lesions),
                                  torch.cat(all_grade_preds), torch.cat(all_grades))
        metrics["loss"] = round(total_loss / len(loader), 4)
        return metrics

    def save_checkpoint(self, epoch):
        torch.save({
            "epoch": epoch,
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "best_score": self.best_score,
            "history": self.history,
        }, self.save_dir / "last.pt")

    def resume(self):
        path = self.save_dir / "last.pt"
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_score = checkpoint["best_score"]
        self.history = checkpoint["history"]
        self.start_epoch = checkpoint["epoch"] + 1
        print("Resumed from", path, "-> starting at epoch", self.start_epoch)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            start = time.time()
            train_loss = self.train_one_epoch(train_loader)
            val = self.evaluate(val_loader)
            minutes = (time.time() - start) / 60

            print(f"  train loss {train_loss:.4f} | val loss {val['loss']:.4f} | {minutes:.1f} min")
            print_metrics(val)

            self.history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val": val})
            score = round((val["mean_lesion_auc"] or 0.0) + val["grade_kappa"], 4)
            if score > self.best_score:
                self.best_score = score
                torch.save(self.network.state_dict(), self.save_dir / "best.pt")
                print(f"  new best (mean lesion AUC + kappa = {score}) -> saved best.pt")
            self.save_checkpoint(epoch)
            with open(self.save_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)


def build_trainer(data, train_idx, weights_path, save_dir, device, epochs, batch_size=16,
                  blr=5e-3, layer_decay=0.65, drop_path=0.2, smoothing=0.1, warmup_epochs=1,
                  num_workers=0):
    pos_weight, _ = compute_pos_weight(data, train_idx)
    train_loader = DataLoader(TrainingImages(data, train_idx, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    network = RETFoundNetwork(weights_path, drop_path=drop_path)
    lr = blr * batch_size / 256
    optimizer = make_optimizer(network, lr, layer_decay)
    scheduler = make_scheduler(optimizer, len(train_loader), epochs, warmup_epochs)
    print(f"learning rate (top layer): {lr:.2e} | layer decay {layer_decay} | drop path {drop_path}"
          f" | label smoothing {smoothing} | warm-up {warmup_epochs} epoch(s) | batch {batch_size}")
    trainer = Trainer(network, RetinalLoss(pos_weight, label_smoothing=smoothing), optimizer, device,
                      save_dir, scheduler)
    return trainer, train_loader


if __name__ == "__main__":
    from dataset import MMRDRDataset

    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    data = MMRDRDataset("cfp", "train")
    train_idx, val_idx = split_train_val(data)
    train_idx, val_idx = train_idx[:200], val_idx[:50]
    print("smoke test images -> train:", len(train_idx), "| validation:", len(val_idx))

    trainer, train_loader = build_trainer(data, train_idx, WEIGHTS_FILE, REPO_ROOT / "checkpoints" / "smoke_test",
                                          device, epochs=3, batch_size=8)
    val_loader = DataLoader(TrainingImages(data, val_idx, augment=False), batch_size=8)
    trainer.fit(train_loader, val_loader, epochs=3)

    print("\ntrain loss per epoch:", [h["train_loss"] for h in trainer.history])
