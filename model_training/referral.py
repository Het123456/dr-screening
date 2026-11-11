import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

import dataset
from dataset import MMRDRDataset
from evaluate import load_network, predict, score
from metrics import split_train_val
from trainer import pick_device

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERABLE_GRADE = 2


def referral_score(grade_probs):
    return grade_probs[:, REFERABLE_GRADE:].sum(dim=1)


def sensitivity_specificity(refer, grade_true):
    needs = grade_true >= REFERABLE_GRADE
    sensitivity = (refer & needs).sum().item() / needs.sum().item()
    specificity = (~refer & ~needs).sum().item() / (~needs).sum().item()
    return round(sensitivity, 4), round(specificity, 4)


def choose_cutoff(scores, grade_true, target_sensitivity):
    for cutoff in sorted(set(scores.tolist()), reverse=True):
        sens, _ = sensitivity_specificity(scores >= cutoff, grade_true)
        if sens >= target_sensitivity:
            return round(cutoff, 4)
    return 0.0


def get_predictions(network, data, indices, device, cache_file):
    if cache_file.exists():
        print("using saved predictions:", cache_file.name)
        return torch.load(cache_file)
    results = {}
    for flip in (False, True):
        print(f"predicting {cache_file.stem} (flip={flip})...")
        results[flip] = predict(network, data, indices, device, flip=flip)
    torch.save(results, cache_file)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(REPO_ROOT / "checkpoints" / "cfp" / "best.pt"))
    parser.add_argument("--modality", default="cfp", choices=["cfp", "uwf"])
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--target_sensitivity", type=float, default=0.90)
    parser.add_argument("--tag", default="", help="added to the result file names, e.g. v2 -> cfp_v2_referral.json")
    args = parser.parse_args()

    if args.data_root:
        dataset.DATA_ROOT = Path(args.data_root)
    device = pick_device()
    network = load_network(args.model, device)
    cache_dir = Path(args.model).parent

    train_data = MMRDRDataset(args.modality, "train")
    _, val_idx = split_train_val(train_data)
    test_data = MMRDRDataset(args.modality, "test")
    val = get_predictions(network, train_data, val_idx, device, cache_dir / "predictions_val.pt")
    test = get_predictions(network, test_data, list(range(len(test_data))), device, cache_dir / "predictions_test.pt")

    print("\nMirror-image averaging (test set):")
    test_scores = {}
    for flip in (False, True):
        test_scores[flip] = score(*test[flip])
        m = test_scores[flip]
        print(f"  flip={flip!s:<5}  mean lesion AUC {m['mean_lesion_auc']} | kappa {m['grade_kappa']} | accuracy {m['grade_accuracy']}")
    val_flip, val_plain = score(*val[True]), score(*val[False])
    flip = val_flip["mean_lesion_auc"] + val_flip["grade_kappa"] >= val_plain["mean_lesion_auc"] + val_plain["grade_kappa"]
    print(f"  on validation, flip {'helps' if flip else 'does not help'} -> using flip={flip}")

    _, val_grade_probs, _, val_grades = val[flip]
    _, test_grade_probs, _, test_grades = test[flip]
    val_ref, test_ref = referral_score(val_grade_probs), referral_score(test_grade_probs)

    print("\nReferral trade-off on VALIDATION (refer if referral score >= cutoff):")
    print("  cutoff   sensitivity (sick eyes caught)   specificity (healthy eyes cleared)")
    for cutoff in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05):
        sens, spec = sensitivity_specificity(val_ref >= cutoff, val_grades)
        print(f"  {cutoff:<8} {sens:<32} {spec}")

    cutoff = choose_cutoff(val_ref, val_grades, args.target_sensitivity)
    print(f"\nchosen cutoff (target sensitivity {args.target_sensitivity} on validation): {cutoff}")

    old = sensitivity_specificity(test_grade_probs.argmax(dim=1) >= REFERABLE_GRADE, test_grades)
    new = sensitivity_specificity(test_ref >= cutoff, test_grades)
    auc = round(roc_auc_score((test_grades >= REFERABLE_GRADE).int(), test_ref), 4)
    print("\nTEST SET referral results:")
    print(f"  old rule (most likely grade >= 2): sensitivity {old[0]} | specificity {old[1]}")
    print(f"  new rule (score >= {cutoff}):     sensitivity {new[0]} | specificity {new[1]}")
    print(f"  referral AUC: {auc}")

    out = REPO_ROOT / "results"
    name = args.modality + (f"_{args.tag}" if args.tag else "")
    with open(out / f"{name}_referral.json", "w") as f:
        json.dump({
            "referable_grade": REFERABLE_GRADE, "use_flip": flip, "cutoff": cutoff,
            "target_sensitivity_on_validation": args.target_sensitivity,
            "test_old_rule": {"sensitivity": old[0], "specificity": old[1]},
            "test_new_rule": {"sensitivity": new[0], "specificity": new[1]},
            "test_referral_auc": auc,
        }, f, indent=2)
    with open(out / f"{name}_model_on_{args.modality}_test_flip.json", "w") as f:
        json.dump(test_scores[True], f, indent=2)
    print(f"saved results/{name}_referral.json and results/{name}_model_on_{args.modality}_test_flip.json")
