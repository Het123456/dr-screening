"""Exercise 8: the training loop."""

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from metrics import RetinalLoss, compute_metrics, compute_pos_weight, print_metrics, split_train_val
from retfound_model import RETFoundNetwork

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_FILE = REPO_ROOT / "weights" / "RETFound_mae_natureCFP.pth"


def pick_device():
    # Use the fastest hardware available: NVIDIA GPU (Colab/Kaggle), Apple GPU (Mac), else CPU
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TrainingImages(Dataset):
    # Connects our MMRDRDataset to PyTorch's DataLoader, which groups images into batches.
    # It only uses the image numbers in `indices`, so train and validation stay separate.
    def __init__(self, data, indices, augment):
        self.data = data
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.data.get_item(self.indices[i], augment=self.augment)


class Trainer:
    def __init__(self, network, loss_fn, optimizer, device, save_dir):
        self.network = network.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = []        # one entry per epoch, saved to history.json
        self.best_score = -1.0   # best validation mean lesion AUC so far
        self.start_epoch = 1

    def train_one_epoch(self, loader):
        self.network.train()  # switch on training behaviour (dropout etc.)
        total_loss = 0.0
        for step, (images, grades, lesions) in enumerate(loader, start=1):
            images, grades, lesions = images.to(self.device), grades.to(self.device), lesions.to(self.device)

            lesion_logits, grade_logits = self.network(images)
            loss, _, _ = self.loss_fn(lesion_logits, grade_logits, lesions, grades)

            self.optimizer.zero_grad()   # forget the previous batch's gradients
            loss.backward()              # work out how to change each weight
            # Stop one bad batch from making a huge jump (rare lesions have big weights)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()        # change the weights a little

            total_loss += loss.item()
            if step % 10 == 0 or step == len(loader):
                print(f"    batch {step}/{len(loader)}  loss {loss.item():.4f}")
        return total_loss / len(loader)

    def evaluate(self, loader):
        # Run the model on images it does not learn from and score it
        self.network.eval()
        total_loss = 0.0
        all_scores, all_lesions, all_grade_preds, all_grades = [], [], [], []
        with torch.no_grad():  # no learning here, so skip the gradient bookkeeping
            for images, grades, lesions in loader:
                images, grades, lesions = images.to(self.device), grades.to(self.device), lesions.to(self.device)
                lesion_logits, grade_logits = self.network(images)
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
        # last.pt: everything needed to carry on later (Colab can disconnect mid-training)
        torch.save({
            "epoch": epoch,
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "history": self.history,
        }, self.save_dir / "last.pt")

    def resume(self):
        # Carry on from last.pt if it exists
        path = self.save_dir / "last.pt"
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
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
            score = val["mean_lesion_auc"] or 0.0
            if score > self.best_score:
                # best.pt: just the model weights from the best epoch; this is what the app uses
                self.best_score = score
                torch.save(self.network.state_dict(), self.save_dir / "best.pt")
                print(f"  new best mean lesion AUC {score} -> saved best.pt")
            self.save_checkpoint(epoch)
            with open(self.save_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)


def make_optimizer(network, backbone_lr, head_lr):
    # The backbone already knows a lot about retinas, so it gets small changes.
    # The two heads start from random numbers, so they get bigger changes.
    backbone_params = list(network.backbone.parameters())
    head_params = list(network.lesion_head.parameters()) + list(network.grade_head.parameters())
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=0.05)


if __name__ == "__main__":
    # Smoke test: a tiny run on the Mac to prove the model CAN learn (loss goes down).
    # The scores mean nothing with this few images; the real run happens on Colab/Kaggle.
    from dataset import MMRDRDataset

    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    data = MMRDRDataset("cfp", "train")
    train_idx, val_idx = split_train_val(data)
    train_idx, val_idx = train_idx[:200], val_idx[:50]   # tiny slice for the smoke test
    print("smoke test images -> train:", len(train_idx), "| validation:", len(val_idx))

    pos_weight, _ = compute_pos_weight(data, train_idx)
    train_loader = DataLoader(TrainingImages(data, train_idx, augment=True), batch_size=8, shuffle=True)
    val_loader = DataLoader(TrainingImages(data, val_idx, augment=False), batch_size=8)

    network = RETFoundNetwork(WEIGHTS_FILE)
    trainer = Trainer(
        network,
        RetinalLoss(pos_weight),
        make_optimizer(network, backbone_lr=2e-5, head_lr=1e-3),
        device,
        save_dir=REPO_ROOT / "checkpoints" / "smoke_test",
    )
    trainer.fit(train_loader, val_loader, epochs=3)

    print("\ntrain loss per epoch:", [h["train_loss"] for h in trainer.history])
