"""Exercise 7: train/validation split, the loss function, and the metrics."""

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
    # Shuffle the training image numbers and keep a slice aside for validation.
    # The test images (ts*) are never touched here; they are only for the final numbers.
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)  # same seed -> same split every run
    val_size = int(len(indices) * val_fraction)
    return indices[val_size:], indices[:val_size]  # train indices, validation indices


def compute_pos_weight(dataset, indices):
    # For each lesion: (number of images without it) / (number with it).
    # Rare lesions get a big weight so the model cannot just always answer "no".
    positives = [0] * len(LESION_NAMES)
    for i in indices:
        for j, value in enumerate(dataset[i].lesions):
            positives[j] += value
    total = len(indices)
    weights = [(total - p) / max(p, 1) for p in positives]  # max(p, 1) avoids dividing by 0
    return torch.tensor(weights, dtype=torch.float32), positives


class RetinalLoss(nn.Module):
    # One number that says how wrong the model is, made of two parts:
    #   lesions: 7 separate yes/no questions -> weighted binary cross-entropy
    #   grade:   pick one of 5 grades        -> cross-entropy
    def __init__(self, pos_weight, grade_loss_weight=1.0):
        super().__init__()
        self.lesion_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.grade_loss = nn.CrossEntropyLoss()
        self.grade_loss_weight = grade_loss_weight

    def forward(self, lesion_logits, grade_logits, lesion_targets, grade_targets):
        lesion_part = self.lesion_loss(lesion_logits, lesion_targets)
        grade_part = self.grade_loss(grade_logits, grade_targets)
        total = lesion_part + self.grade_loss_weight * grade_part
        return total, lesion_part.item(), grade_part.item()


def compute_metrics(lesion_scores, lesion_targets, grade_preds, grade_targets):
    # lesion_scores:  [N, 7] numbers between 0 and 1 (after sigmoid)
    # lesion_targets: [N, 7] 0 or 1
    # grade_preds, grade_targets: [N] whole numbers 0 to 4
    lesion_scores = torch.as_tensor(lesion_scores).tolist()
    lesion_targets = torch.as_tensor(lesion_targets).tolist()
    grade_preds = torch.as_tensor(grade_preds).tolist()
    grade_targets = torch.as_tensor(grade_targets).tolist()

    lesion_auc = {}
    for j, name in enumerate(LESION_NAMES):
        y_true = [row[j] for row in lesion_targets]
        y_score = [row[j] for row in lesion_scores]
        if len(set(y_true)) < 2:
            lesion_auc[name] = None  # AUC needs both "yes" and "no" examples
        else:
            lesion_auc[name] = round(roc_auc_score(y_true, y_score), 4)

    valid_aucs = [a for a in lesion_auc.values() if a is not None]
    mean_auc = round(sum(valid_aucs) / len(valid_aucs), 4) if valid_aucs else None

    return {
        "lesion_auc": lesion_auc,
        "mean_lesion_auc": mean_auc,
        "grade_accuracy": round(accuracy_score(grade_targets, grade_preds), 4),
        # Quadratic weighted kappa: the usual DR-grading score. Guessing grade 3 for a
        # grade 4 eye is punished less than guessing grade 0. 1 = perfect, 0 = chance.
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

    # 1. Split
    train_idx, val_idx = split_train_val(data)
    print("CFP train images:", len(data), "-> train:", len(train_idx), "| validation:", len(val_idx))
    print("overlap between train and validation:", len(set(train_idx) & set(val_idx)))

    # 2. Class weights (from the train part only)
    pos_weight, positives = compute_pos_weight(data, train_idx)
    print("\nlesion                    positives   pos_weight")
    for name, p, w in zip(LESION_NAMES, positives, pos_weight.tolist()):
        print(f"  {name:<25} {p:>6}   {w:>8.2f}")

    # 3. Loss on real labels from 8 images, with random model outputs (no model needed)
    batch = [data.get_item(i) for i in train_idx[:8]]
    grade_targets = torch.stack([b[1] for b in batch])
    lesion_targets = torch.stack([b[2] for b in batch])
    loss_fn = RetinalLoss(pos_weight)
    total, lesion_part, grade_part = loss_fn(torch.randn(8, 7), torch.randn(8, 5), lesion_targets, grade_targets)
    print("\nloss with random outputs -> total:", round(total.item(), 4),
          "| lesion part:", round(lesion_part, 4), "| grade part:", round(grade_part, 4))

    # 4. Sanity check of the metrics on the validation labels
    val_lesions = torch.tensor([data[i].lesions for i in val_idx], dtype=torch.float32)
    val_grades = torch.tensor([data[i].grade for i in val_idx])

    print("\nRandom guesses (expect AUC about 0.5, kappa about 0):")
    print_metrics(compute_metrics(torch.rand(len(val_idx), 7), val_lesions,
                                  torch.randint(0, 5, (len(val_idx),)), val_grades))

    print("\nPerfect answers (expect AUC 1.0, accuracy 1.0, kappa 1.0):")
    print_metrics(compute_metrics(val_lesions, val_lesions, val_grades, val_grades))
