"""Score a trained model on a TEST set (ts* images it never saw during training).

Run with, for example:
    /opt/anaconda3/bin/python3 evaluate.py --model ../checkpoints/cfp/best.pt --modality cfp
"""

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import dataset
from dataset import MMRDRDataset
from metrics import compute_metrics, print_metrics
from retfound_model import RETFoundNetwork
from trainer import TrainingImages, pick_device

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_on_test(model_path, modality, data_root=None, batch_size=16):
    if data_root:
        dataset.DATA_ROOT = Path(data_root)  # e.g. the shrunk 224 copy used for training
    device = pick_device()

    network = RETFoundNetwork()  # no RETFound file needed: best.pt already has every weight
    network.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    network.to(device).eval()

    data = MMRDRDataset(modality, "test")
    loader = DataLoader(TrainingImages(data, list(range(len(data))), augment=False), batch_size=batch_size)
    print(f"device: {device} | {modality} test images: {len(data)} | data: {dataset.DATA_ROOT}")

    all_scores, all_lesions, all_grade_preds, all_grades = [], [], [], []
    start = time.time()
    with torch.no_grad():
        for step, (images, grades, lesions) in enumerate(loader, start=1):
            lesion_logits, grade_logits = network(images.to(device))
            all_scores.append(torch.sigmoid(lesion_logits).cpu())
            all_grade_preds.append(grade_logits.argmax(dim=1).cpu())
            all_lesions.append(lesions)
            all_grades.append(grades)
            if step % 20 == 0 or step == len(loader):
                print(f"  batch {step}/{len(loader)}  ({time.time() - start:.0f}s)")

    grade_preds, grade_true = torch.cat(all_grade_preds), torch.cat(all_grades)
    metrics = compute_metrics(torch.cat(all_scores), torch.cat(all_lesions), grade_preds, grade_true)
    # Rows = true grade, columns = predicted grade. Numbers on the diagonal are correct answers.
    metrics["grade_confusion_matrix"] = confusion_matrix(grade_true, grade_preds, labels=[0, 1, 2, 3, 4]).tolist()
    metrics["num_images"] = len(data)
    return metrics


def print_confusion(matrix):
    print("  grade confusion matrix (rows = true grade, columns = predicted grade):")
    print("          " + "".join(f"pred {g:<4}" for g in range(5)))
    for g, row in enumerate(matrix):
        print(f"  true {g}  " + "".join(f"{n:<9}" for n in row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(REPO_ROOT / "checkpoints" / "cfp" / "best.pt"))
    parser.add_argument("--modality", default="cfp", choices=["cfp", "uwf"])
    parser.add_argument("--data_root", default=None, help="folder containing MMRDR-CFP / MMRDR-UWF (default: the original data)")
    parser.add_argument("--output", default=None, help="where to save the results JSON")
    args = parser.parse_args()

    results = evaluate_on_test(args.model, args.modality, args.data_root)
    print(f"\nTEST RESULTS ({args.modality}):")
    print_metrics(results)
    print_confusion(results["grade_confusion_matrix"])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", args.output)
