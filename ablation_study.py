"""
Feature Ablation & Permutation Analysis Engine
SIH 26109: Bovine Mastitis Early Forecasting System (MastiFore)

Performs:
1. Full Feature Baseline Evaluation
2. Leave-One-Feature-Out (LOFO) Systematic Ablation
3. Low-Cost Hardware Feature Subset Permutations (Testing pairs, triplets, and 5-feature sets)
4. Measures Accuracy, Sensitivity/Recall, Precision, F1-Score, and ROC-AUC for every permutation.
"""

import os
import csv
import math
import json
import random
from itertools import combinations

DATASET_PATH = os.path.join(os.path.dirname(__file__), "tier2_mastitis_training_data.csv")
RESULTS_SAVE_PATH = os.path.join(os.path.dirname(__file__), "ablation_study_results.json")


def load_dataset(csv_path):
    features = []
    labels = []
    feature_names = ["Milk_Temperature", "Milk_pH", "Milk_Conductivity", "Somatic_Cell_Count", "Milk_Yield", "Clotting"]

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row or len(row) < 9:
                continue
            temp = float(row[2])
            ph = float(row[3])
            ec = float(row[4])
            scc = float(row[5])
            yield_l = float(row[6])
            clotting = float(row[7])
            label = int(row[8])

            features.append([temp, ph, ec, scc, yield_l, clotting])
            labels.append(label)

    return features, labels, feature_names


def compute_statistics(X):
    n_samples = len(X)
    n_features = len(X[0])
    means = [0.0] * n_features
    stds = [0.0] * n_features

    for j in range(n_features):
        col_vals = [X[i][j] for i in range(n_samples)]
        m = sum(col_vals) / n_samples
        variance = sum((v - m) ** 2 for v in col_vals) / max(1, n_samples - 1)
        means[j] = m
        stds[j] = max(1e-5, math.sqrt(variance))

    return means, stds


def standardize(X, means, stds):
    return [[(X[i][j] - means[j]) / stds[j] for j in range(len(stds))] for i in range(len(X))]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def train_logistic_regression(X_train, y_train, lr=0.08, epochs=800, l2_reg=0.005):
    n_samples = len(X_train)
    n_features = len(X_train[0])
    weights = [0.0] * n_features
    bias = 0.0

    for epoch in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0

        for i in range(n_samples):
            z = sum(weights[j] * X_train[i][j] for j in range(n_features)) + bias
            p = sigmoid(z)
            error = p - y_train[i]

            for j in range(n_features):
                grad_w[j] += error * X_train[i][j]
            grad_b += error

        for j in range(n_features):
            weights[j] -= lr * ((grad_w[j] / n_samples) + (l2_reg * weights[j]))
        bias -= lr * (grad_b / n_samples)

    return weights, bias


def evaluate_metrics(y_true, y_pred, y_probs):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-5, (precision + recall))

    pos_probs = [p for yt, p in zip(y_true, y_probs) if yt == 1]
    neg_probs = [p for yt, p in zip(y_true, y_probs) if yt == 0]
    pairs = 0
    concordant = 0
    for pp in pos_probs:
        for np in neg_probs:
            pairs += 1
            if pp > np:
                concordant += 1
            elif pp == np:
                concordant += 0.5
    auc = concordant / max(1, pairs)

    return {
        "accuracy": round(accuracy * 100, 2),
        "recall_sensitivity": round(recall * 100, 2),
        "precision": round(precision * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(auc * 100, 2)
    }


def train_and_eval_subset(X_raw, y, feature_indices, train_idx, test_idx):
    # Slice only the selected feature columns
    X_train_sub = [[X_raw[i][col] for col in feature_indices] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test_sub = [[X_raw[i][col] for col in feature_indices] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    means, stds = compute_statistics(X_train_sub)
    X_train_std = standardize(X_train_sub, means, stds)
    X_test_std = standardize(X_test_sub, means, stds)

    weights, bias = train_logistic_regression(X_train_std, y_train)

    test_probs = [sigmoid(sum(weights[j] * X_test_std[i][j] for j in range(len(weights))) + bias) for i in range(len(X_test_std))]
    test_preds = [1 if p >= 0.50 else 0 for p in test_probs]

    return evaluate_metrics(y_test, test_preds, test_probs)


def run_ablation_study():
    print("=" * 85)
    print(" SIH 26109: MASTIFORE FEATURE ABLATION & PERMUTATION STUDY")
    print("=" * 85)

    X_raw, y, all_feature_names = load_dataset(DATASET_PATH)

    # Fixed 80-20 Stratified Split
    random.seed(42)
    healthy_idx = [i for i, v in enumerate(y) if v == 0]
    mastitis_idx = [i for i, v in enumerate(y) if v == 1]
    random.shuffle(healthy_idx)
    random.shuffle(mastitis_idx)

    train_idx = healthy_idx[:int(0.8 * len(healthy_idx))] + mastitis_idx[:int(0.8 * len(mastitis_idx))]
    test_idx = healthy_idx[int(0.8 * len(healthy_idx)):] + mastitis_idx[int(0.8 * len(mastitis_idx)):]

    results = {}

    # 1. Full 6-Feature Baseline (Includes SCC)
    full_indices = list(range(len(all_feature_names)))
    base_metrics = train_and_eval_subset(X_raw, y, full_indices, train_idx, test_idx)
    results["Baseline (All 6 Features)"] = {
        "features": all_feature_names,
        "metrics": base_metrics
    }

    # 2. Leave-One-Feature-Out (LOFO) Ablation
    print("\n[PART 1: LEAVE-ONE-FEATURE-OUT (LOFO) ABLATION ANALYSIS]")
    print("-" * 85)
    print(f"{'Ablated Feature (Removed)':<32} | {'Remaining Features':<8} | {'Accuracy':<9} | {'Sensitivity':<11} | {'F1-Score':<9} | {'ROC-AUC':<9}")
    print("-" * 85)

    print(f"{'None (Full 6-Feature Baseline)':<32} | {'6 / 6':<8} | {base_metrics['accuracy']:>7.2f}% | {base_metrics['recall_sensitivity']:>9.2f}% | {base_metrics['f1_score']:>7.2f}% | {base_metrics['roc_auc']:>7.2f}%")

    lofo_results = {}
    for idx_to_remove, removed_name in enumerate(all_feature_names):
        sub_indices = [i for i in range(len(all_feature_names)) if i != idx_to_remove]
        sub_names = [all_feature_names[i] for i in sub_indices]
        m = train_and_eval_subset(X_raw, y, sub_indices, train_idx, test_idx)
        lofo_results[f"Without {removed_name}"] = {"removed": removed_name, "metrics": m}
        print(f"{'Without ' + removed_name:<32} | {'5 / 6':<8} | {m['accuracy']:>7.2f}% | {m['recall_sensitivity']:>9.2f}% | {m['f1_score']:>7.2f}% | {m['roc_auc']:>7.2f}%")

    results["lofo_ablation"] = lofo_results

    # 3. Low-Cost Hardware Feature Permutations (SCC Excluded)
    # Hardware Features: Milk_Temperature(0), Milk_pH(1), Milk_Conductivity(2), Milk_Yield(4), Clotting(5)
    hw_feature_map = {
        0: "Temp",
        1: "pH",
        2: "EC",
        4: "Yield",
        5: "Clot"
    }
    hw_indices = [0, 1, 2, 4, 5]

    print("\n[PART 2: HARDWARE FEATURE SUBSET PERMUTATIONS (EXCLUDING LAB SCC)]")
    print("-" * 85)
    print(f"{'Feature Permutation Combination':<35} | {'Size':<6} | {'Accuracy':<9} | {'Sensitivity':<11} | {'F1-Score':<9} | {'ROC-AUC':<9}")
    print("-" * 85)

    # 5-Feature Hardware Set (Our Production Model)
    hw_5_metrics = train_and_eval_subset(X_raw, y, hw_indices, train_idx, test_idx)
    print(f"{'Our 5-Sensor Model (Temp+pH+EC+Yld+Clot)':<35} | {'5 / 5':<6} | {hw_5_metrics['accuracy']:>7.2f}% | {hw_5_metrics['recall_sensitivity']:>9.2f}% | {hw_5_metrics['f1_score']:>7.2f}% | {hw_5_metrics['roc_auc']:>7.2f}%")

    # Single-sensor individual predictive power
    single_results = {}
    for idx in hw_indices:
        name = hw_feature_map[idx]
        m = train_and_eval_subset(X_raw, y, [idx], train_idx, test_idx)
        single_results[name] = m
        print(f"{'Single Sensor: [' + name + ']':<35} | {'1 / 5':<6} | {m['accuracy']:>7.2f}% | {m['recall_sensitivity']:>9.2f}% | {m['f1_score']:>7.2f}% | {m['roc_auc']:>7.2f}%")

    # 2-sensor pair combinations
    pair_results = {}
    for pair in combinations(hw_indices, 2):
        pair_name = "+".join(hw_feature_map[i] for i in pair)
        m = train_and_eval_subset(X_raw, y, list(pair), train_idx, test_idx)
        pair_results[pair_name] = m
        print(f"{'Pair: [' + pair_name + ']':<35} | {'2 / 5':<6} | {m['accuracy']:>7.2f}% | {m['recall_sensitivity']:>9.2f}% | {m['f1_score']:>7.2f}% | {m['roc_auc']:>7.2f}%")

    # 3-sensor triplet combinations
    triplet_results = {}
    for trip in combinations(hw_indices, 3):
        trip_name = "+".join(hw_feature_map[i] for i in trip)
        m = train_and_eval_subset(X_raw, y, list(trip), train_idx, test_idx)
        triplet_results[trip_name] = m
        print(f"{'Triplet: [' + trip_name + ']':<35} | {'3 / 5':<6} | {m['accuracy']:>7.2f}% | {m['recall_sensitivity']:>9.2f}% | {m['f1_score']:>7.2f}% | {m['roc_auc']:>7.2f}%")

    results["hardware_5_sensor"] = hw_5_metrics
    results["single_sensor_ablation"] = single_results
    results["pair_permutations"] = pair_results
    results["triplet_permutations"] = triplet_results

    with open(RESULTS_SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 85)
    print(f"\n[Artifact Saved]: Full Ablation & Permutation Results saved to {RESULTS_SAVE_PATH}")
    print("=" * 85)


if __name__ == "__main__":
    run_ablation_study()
