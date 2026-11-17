import argparse
import json
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from evaluate import score
from metrics import LESION_NAMES
from referral import REFERABLE_GRADE, choose_cutoff, referral_score, sensitivity_specificity

REPO_ROOT = Path(__file__).resolve().parent.parent
LIKELY = 0.7
POSSIBLE = 0.3


def to_logit(scores):
    scores = scores.double().clamp(1e-6, 1 - 1e-6)
    return torch.log(scores / (1 - scores))


def fit_line(scores, present):
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(to_logit(scores).reshape(-1, 1).numpy(), present.int().numpy())
    return round(float(model.coef_[0][0]), 4), round(float(model.intercept_[0]), 4)


def calibrate(scores, slope, intercept):
    return torch.sigmoid(slope * to_logit(scores) + intercept)


def share_present(present, chosen):
    count = int(chosen.sum().item())
    if count == 0:
        return {"eyes": 0, "share_with_lesion": None}
    return {"eyes": count, "share_with_lesion": round((present & chosen).sum().item() / count, 4)}


def choose_flip(val):
    with_flip, without_flip = score(*val[True]), score(*val[False])
    total_with = with_flip["mean_lesion_auc"] + with_flip["grade_kappa"]
    total_without = without_flip["mean_lesion_auc"] + without_flip["grade_kappa"]
    return total_with >= total_without


def save_json(path, values):
    with open(path, "w") as f:
        json.dump(values, f, indent=2)
    print("saved", path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder inside checkpoints/, e.g. cfp_retfound_512")
    parser.add_argument("--modality", required=True, choices=["cfp", "uwf"])
    parser.add_argument("--name", required=True, help="start of the result file names, e.g. cfp_512")
    parser.add_argument("--target_sensitivity", type=float, default=0.90)
    args = parser.parse_args()

    folder = REPO_ROOT / "checkpoints" / args.run
    val = torch.load(folder / "predictions_val.pt")
    test = torch.load(folder / "predictions_test.pt")
    results = REPO_ROOT / "results"

    flip = choose_flip(val)
    print("mirror averaging helps on validation:", flip)
    val_lesions, val_grade_probs, val_labels, val_grades = val[flip]
    test_lesions, test_grade_probs, test_labels, test_grades = test[flip]

    cutoff = choose_cutoff(referral_score(val_grade_probs), val_grades, args.target_sensitivity)
    test_ref = referral_score(test_grade_probs)
    old = sensitivity_specificity(test_grade_probs.argmax(dim=1) >= REFERABLE_GRADE, test_grades)
    new = sensitivity_specificity(test_ref >= cutoff, test_grades)
    auc = round(roc_auc_score((test_grades >= REFERABLE_GRADE).int(), test_ref), 4)
    print(f"referral cutoff {cutoff} | test sensitivity {new[0]} specificity {new[1]} | referral AUC {auc}")
    save_json(results / f"{args.name}_referral.json", {
        "referable_grade": REFERABLE_GRADE, "use_flip": flip, "cutoff": cutoff,
        "target_sensitivity_on_validation": args.target_sensitivity,
        "test_old_rule": {"sensitivity": old[0], "specificity": old[1]},
        "test_new_rule": {"sensitivity": new[0], "specificity": new[1]},
        "test_referral_auc": auc,
    })
    save_json(results / f"{args.name}_model_on_{args.modality}_test_flip.json", score(*test[True]))

    lesions = []
    for index, lesion_name in enumerate(LESION_NAMES):
        slope, intercept = fit_line(val_lesions[:, index], val_labels[:, index] > 0.5)
        calibrated = calibrate(test_lesions[:, index], slope, intercept)
        present = test_labels[:, index] > 0.5
        likely = share_present(present, calibrated >= LIKELY)
        possible = share_present(present, (calibrated >= POSSIBLE) & (calibrated < LIKELY))
        unlikely = share_present(present, calibrated < POSSIBLE)
        lesions.append({"name": lesion_name, "slope": slope, "intercept": intercept,
                        "test_likely": likely, "test_possible": possible, "test_unlikely": unlikely})
        print(f"{lesion_name:26} slope {slope} intercept {intercept} | likely {likely} | "
              f"possible {possible} | unlikely {unlikely}")
    save_json(results / f"{args.name}_lesion_calibration.json", {
        "use_flip": flip, "fitted_on": "validation", "likely_from": LIKELY, "possible_from": POSSIBLE,
        "lesions": lesions,
    })
