import random

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

LESION_NAMES = [
    "microaneurysm",
    "hard exudate",
    "intraretinal hemorrhage",
    "VB/IRMA",
    "neovascularization",
    "vitreous hemorrhage",
    "retinal detachment",
]


def split_train_val(dataset, val_fraction=0.1, seed=42):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    val_size = int(len(indices) * val_fraction)
    return indices[val_size:], indices[:val_size]


def compute_pos_weight(dataset, indices):
    positives = [0] * len(LESION_NAMES)
    for i in indices:
        for j, value in enumerate(dataset[i].lesions):
            positives[j] += value
    total = len(indices)
    weights = [(total - p) / max(p, 1) for p in positives]
    return torch.tensor(weights, dtype=torch.float32), positives


class RetinalLoss(nn.Module):
    def __init__(self, pos_weight, grade_loss_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.lesion_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.grade_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.grade_loss_weight = grade_loss_weight

    def forward(self, lesion_logits, grade_logits, lesion_targets, grade_targets):
        lesion_part = self.lesion_loss(lesion_logits, lesion_targets)
        grade_part = self.grade_loss(grade_logits, grade_targets)
        total = lesion_part + self.grade_loss_weight * grade_part
        return total, lesion_part.item(), grade_part.item()


def compute_metrics(lesion_scores, lesion_targets, grade_preds, grade_targets):
    lesion_scores = torch.as_tensor(lesion_scores).tolist()
    lesion_targets = torch.as_tensor(lesion_targets).tolist()
    grade_preds = torch.as_tensor(grade_preds).tolist()
    grade_targets = torch.as_tensor(grade_targets).tolist()

    lesion_auc = {}
    for j, name in enumerate(LESION_NAMES):
        y_true = [row[j] for row in lesion_targets]
        y_score = [row[j] for row in lesion_scores]
        if len(set(y_true)) < 2:
            lesion_auc[name] = None
        else:
            lesion_auc[name] = round(roc_auc_score(y_true, y_score), 4)

    valid_aucs = [a for a in lesion_auc.values() if a is not None]
    mean_auc = round(sum(valid_aucs) / len(valid_aucs), 4) if valid_aucs else None

    return {
        "lesion_auc": lesion_auc,
        "mean_lesion_auc": mean_auc,
        "grade_accuracy": round(accuracy_score(grade_targets, grade_preds), 4),
        "grade_kappa": round(cohen_kappa_score(grade_targets, grade_preds, weights="quadratic"), 4),
    }


def print_metrics(metrics):
    for name, auc in metrics["lesion_auc"].items():
        print(f"  {name:<25} AUC: {auc}")
    print("  mean lesion AUC:", metrics["mean_lesion_auc"])
    print("  grade accuracy: ", metrics["grade_accuracy"])
    print("  grade kappa:    ", metrics["grade_kappa"])


if __name__ == "__main__":
    from dataset import MMRDRDataset

    torch.manual_seed(0)
    data = MMRDRDataset("cfp", "train")

    train_idx, val_idx = split_train_val(data)
    print("CFP train images:", len(data), "-> train:", len(train_idx), "| validation:", len(val_idx))
    print("overlap between train and validation:", len(set(train_idx) & set(val_idx)))

    pos_weight, positives = compute_pos_weight(data, train_idx)
    print("\nlesion                    positives   pos_weight")
    for name, p, w in zip(LESION_NAMES, positives, pos_weight.tolist()):
        print(f"  {name:<25} {p:>6}   {w:>8.2f}")

    batch = [data.get_item(i) for i in train_idx[:8]]
    grade_targets = torch.stack([b[1] for b in batch])
    lesion_targets = torch.stack([b[2] for b in batch])
    loss_fn = RetinalLoss(pos_weight)
    total, lesion_part, grade_part = loss_fn(torch.randn(8, 7), torch.randn(8, 5), lesion_targets, grade_targets)
    print("\nloss with random outputs -> total:", round(total.item(), 4),
          "| lesion part:", round(lesion_part, 4), "| grade part:", round(grade_part, 4))

    val_lesions = torch.tensor([data[i].lesions for i in val_idx], dtype=torch.float32)
    val_grades = torch.tensor([data[i].grade for i in val_idx])

    print("\nRandom guesses (expect AUC about 0.5, kappa about 0):")
    print_metrics(compute_metrics(torch.rand(len(val_idx), 7), val_lesions,
                                  torch.randint(0, 5, (len(val_idx),)), val_grades))

    print("\nPerfect answers (expect AUC 1.0, accuracy 1.0, kappa 1.0):")
    print_metrics(compute_metrics(val_lesions, val_lesions, val_grades, val_grades))
