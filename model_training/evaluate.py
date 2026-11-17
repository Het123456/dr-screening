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


def load_network(model_path, device):
    network = RETFoundNetwork()
    network.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return network.to(device).eval()


def predict(network, data, indices, device, flip=False, batch_size=16):
    loader = DataLoader(TrainingImages(data, indices, augment=False), batch_size=batch_size)
    lesion_probs, grade_probs, lesion_true, grade_true = [], [], [], []
    start = time.time()
    with torch.no_grad():
        for step, (images, grades, lesions) in enumerate(loader, start=1):
            images = images.to(device)
            lesion_logits, grade_logits = network(images)
            lesion_p, grade_p = torch.sigmoid(lesion_logits), torch.softmax(grade_logits, dim=1)
            if flip:
                lesion_logits, grade_logits = network(torch.flip(images, dims=[3]))
                lesion_p = (lesion_p + torch.sigmoid(lesion_logits)) / 2
                grade_p = (grade_p + torch.softmax(grade_logits, dim=1)) / 2
            lesion_probs.append(lesion_p.cpu())
            grade_probs.append(grade_p.cpu())
            lesion_true.append(lesions)
            grade_true.append(grades)
            if step % 20 == 0 or step == len(loader):
                print(f"  batch {step}/{len(loader)}  ({time.time() - start:.0f}s)")
    return torch.cat(lesion_probs), torch.cat(grade_probs), torch.cat(lesion_true), torch.cat(grade_true)


def score(lesion_probs, grade_probs, lesion_true, grade_true):
    grade_preds = grade_probs.argmax(dim=1)
    metrics = compute_metrics(lesion_probs, lesion_true, grade_preds, grade_true)
    metrics["grade_confusion_matrix"] = confusion_matrix(grade_true, grade_preds, labels=[0, 1, 2, 3, 4]).tolist()
    metrics["num_images"] = len(grade_true)
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
    parser.add_argument("--flip", action="store_true", help="also average with the mirror image")
    parser.add_argument("--output", default=None, help="where to save the results JSON")
    args = parser.parse_args()

    if args.data_root:
        dataset.DATA_ROOT = Path(args.data_root)
    device = pick_device()
    network = load_network(args.model, device)
    data = MMRDRDataset(args.modality, "test")
    print(f"device: {device} | {args.modality} test images: {len(data)} | flip: {args.flip} | data: {dataset.DATA_ROOT}")

    results = score(*predict(network, data, list(range(len(data))), device, flip=args.flip))
    print(f"\nTEST RESULTS ({args.modality}):")
    print_metrics(results)
    print_confusion(results["grade_confusion_matrix"])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", args.output)
