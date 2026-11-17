import argparse
import json
from pathlib import Path

import torch

from metrics import LESION_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERABLE_GRADE = 2


def sensitivity_specificity(scores, present, cutoff):
    flagged = scores >= cutoff
    sensitivity = (flagged & present).sum().item() / max(1, present.sum().item())
    specificity = (~flagged & ~present).sum().item() / max(1, (~present).sum().item())
    return round(sensitivity, 4), round(specificity, 4)


def choose_cutoff(scores, present):
    best_cutoff = 0.5
    best_total = -1.0
    for step in range(5, 100):
        cutoff = step / 100
        sensitivity, specificity = sensitivity_specificity(scores, present, cutoff)
        if sensitivity + specificity > best_total:
            best_total = sensitivity + specificity
            best_cutoff = cutoff
    return best_cutoff


def false_alarm_rate(scores, present, cutoff):
    absent = ~present
    false_alarms = ((scores >= cutoff) & absent).sum().item()
    return int(absent.sum().item()), round(false_alarms / max(1, absent.sum().item()), 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="cfp", choices=["cfp", "uwf"])
    parser.add_argument("--tag", default="v2")
    args = parser.parse_args()

    name = args.modality + "_" + args.tag
    with open(REPO_ROOT / "results" / (name + "_referral.json")) as f:
        use_flip = json.load(f)["use_flip"]

    folder = REPO_ROOT / "checkpoints" / name
    val_scores, _, val_labels, _ = torch.load(folder / "predictions_val.pt")[use_flip]
    test_scores, _, test_labels, test_grades = torch.load(folder / "predictions_test.pt")[use_flip]
    sick = test_grades >= REFERABLE_GRADE

    lesions = []
    for index, lesion_name in enumerate(LESION_NAMES):
        cutoff = choose_cutoff(val_scores[:, index], val_labels[:, index] > 0.5)
        scores = test_scores[:, index]
        present = test_labels[:, index] > 0.5
        sensitivity, specificity = sensitivity_specificity(scores, present, cutoff)
        absent_count, rate = false_alarm_rate(scores[sick], present[sick], cutoff)
        _, rate_half = false_alarm_rate(scores[sick], present[sick], 0.5)
        lesions.append({
            "name": lesion_name,
            "cutoff": cutoff,
            "test_sensitivity": sensitivity,
            "test_specificity": specificity,
            "referable_eyes_without_lesion": absent_count,
            "false_alarm_rate_in_referable_eyes": rate,
            "false_alarm_rate_in_referable_eyes_at_0.5": rate_half,
        })
        print(f"{lesion_name:26} cutoff {cutoff:.2f} | test sens {sensitivity} spec {specificity} | "
              f"sick eyes without it: {absent_count}, false alarms {rate} (at 0.5: {rate_half})")

    output = REPO_ROOT / "results" / (name + "_lesion_cutoffs.json")
    with open(output, "w") as f:
        json.dump({"use_flip": use_flip, "chosen_on": "validation, best sensitivity + specificity",
                   "num_referable_test_eyes": int(sick.sum().item()), "lesions": lesions}, f, indent=2)
    print("saved", output.name)
