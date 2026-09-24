import os
import csv
import math
import json
import random
from itertools import combinations

DATASET_PATH = os.path.join(os.path.dirname(__file__), "tier2_mastitis_training_data.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "ablation_study_results.json")


def load_dataset(csv_path):
    features = []
    labels = []
    feature_names = ["Milk_Temperature", "Milk_pH", "Milk_Conductivity", "Somatic_Cell_Count", "Milk_Yield", "Clotting"]

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            if not row or len(row) < 9:
                continue
            features.append([
                float(row[2]),  # Temp
                float(row[3]),  # pH
                float(row[4]),  # EC
                float(row[5]),  # SCC
                float(row[6]),  # Yield
                float(row[7])   # Clotting
            ])
            labels.append(int(row[8]))

    return features, labels, feature_names


def calculate_mean_and_std(X):
    n_samples = len(X)
    n_features = len(X[0])
    means = [0.0] * n_features
    stds = [0.0] * n_features

    for j in range(n_features):
        vals = [X[i][j] for i in range(n_samples)]
        m = sum(vals) / n_samples
        variance = sum((v - m) ** 2 for v in vals) / max(1, n_samples - 1)
        means[j] = m
        stds[j] = max(1e-5, math.sqrt(variance))

    return means, stds


def scale(X, means, stds):
    return [[(X[i][j] - means[j]) / stds[j] for j in range(len(stds))] for i in range(len(X))]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def train_logistic(X_train, y_train, lr=0.08, epochs=800, l2_reg=0.005):
    n_samples = len(X_train)
    n_features = len(X_train[0])
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
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


def evaluate_subset(y_true, y_pred, y_probs):
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
        "sensitivity": round(recall * 100, 2),
        "precision": round(precision * 100, 2),
        "f1": round(f1 * 100, 2),
        "roc_auc": round(auc * 100, 2)
    }


def train_and_test(X_raw, y, feature_indices, train_idx, test_idx):
    X_train_sub = [[X_raw[i][col] for col in feature_indices] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test_sub = [[X_raw[i][col] for col in feature_indices] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    means, stds = calculate_mean_and_std(X_train_sub)
    X_train_scaled = scale(X_train_sub, means, stds)
    X_test_scaled = scale(X_test_sub, means, stds)

    weights, bias = train_logistic(X_train_scaled, y_train)

    probs = [sigmoid(sum(weights[j] * X_test_scaled[i][j] for j in range(len(weights))) + bias) for i in range(len(X_test_scaled))]
    preds = [1 if p >= 0.50 else 0 for p in probs]

    return evaluate_subset(y_test, preds, probs)


def run_ablation():
    print("Running feature ablation & permutation benchmark...")
    X_raw, y, feature_names = load_dataset(DATASET_PATH)

    random.seed(42)
    healthy = [i for i, v in enumerate(y) if v == 0]
    mastitis = [i for i, v in enumerate(y) if v == 1]
    random.shuffle(healthy)
    random.shuffle(mastitis)

    train_idx = healthy[:int(0.8 * len(healthy))] + mastitis[:int(0.8 * len(mastitis))]
    test_idx = healthy[int(0.8 * len(healthy)):] + mastitis[int(0.8 * len(mastitis)):]

    results = {}

    # 1. Full 6-feature baseline
    all_indices = list(range(len(feature_names)))
    base_metrics = train_and_test(X_raw, y, all_indices, train_idx, test_idx)
    results["baseline_all_features"] = base_metrics

    print(f"\n1. Baseline (All 6 Features): Accuracy={base_metrics['accuracy']}%, Sensitivity={base_metrics['sensitivity']}%")

    # 2. Leave-One-Feature-Out (LOFO)
    print("\n2. Leave-One-Feature-Out (LOFO) Test:")
    lofo = {}
    for i, name in enumerate(feature_names):
        sub_indices = [idx for idx in range(len(feature_names)) if idx != i]
        m = train_and_test(X_raw, y, sub_indices, train_idx, test_idx)
        lofo[f"without_{name}"] = m
        print(f"   Without {name:<22}: Acc={m['accuracy']:>5.2f}%, Sens={m['sensitivity']:>5.2f}%")
    results["lofo"] = lofo

    # 3. Hardware sensors only (excluding SCC)
    # [0: Temp, 1: pH, 2: EC, 4: Yield, 5: Clotting]
    hw_map = {0: "Temp", 1: "pH", 2: "EC", 4: "Yield", 5: "Clotting"}
    hw_indices = [0, 1, 2, 4, 5]

    hw_metrics = train_and_test(X_raw, y, hw_indices, train_idx, test_idx)
    results["hardware_5_sensors"] = hw_metrics
    print(f"\n3. Production 5 Hardware Sensors: Accuracy={hw_metrics['accuracy']}%, Sensitivity={hw_metrics['sensitivity']}%")

    # 4. Pair and triplet combinations
    pairs = {}
    for p in combinations(hw_indices, 2):
        name = "+".join(hw_map[i] for i in p)
        pairs[name] = train_and_test(X_raw, y, list(p), train_idx, test_idx)
    results["pairs"] = pairs

    triplets = {}
    for t in combinations(hw_indices, 3):
        name = "+".join(hw_map[i] for i in t)
        triplets[name] = train_and_test(X_raw, y, list(t), train_idx, test_idx)
    results["triplets"] = triplets

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation study results exported to {RESULTS_PATH}")


if __name__ == "__main__":
    run_ablation()
