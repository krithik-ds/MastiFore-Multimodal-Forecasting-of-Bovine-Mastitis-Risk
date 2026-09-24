import os
import csv
import math
import json
import random

DATASET_PATH = os.path.join(os.path.dirname(__file__), "tier2_mastitis_training_data.csv")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "tier2_model_metadata.json")


def load_dataset(csv_path):
    features = []
    labels = []
    feature_names = ["Milk_Temperature", "Milk_pH", "Milk_Conductivity", "Milk_Yield", "Clotting"]

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            if not row or len(row) < 9:
                continue
            temp = float(row[2])
            ph = float(row[3])
            ec = float(row[4])
            # row[5] is Somatic Cell Count - omitted because lab tests are expensive
            yield_l = float(row[6])
            clotting = float(row[7])
            label = int(row[8])

            features.append([temp, ph, ec, yield_l, clotting])
            labels.append(label)

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


def scale_features(X, means, stds):
    return [[(X[i][j] - means[j]) / stds[j] for j in range(len(stds))] for i in range(len(X))]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def train_model(X_train, y_train, lr=0.08, epochs=1000, l2_penalty=0.005):
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
            weights[j] -= lr * ((grad_w[j] / n_samples) + (l2_penalty * weights[j]))
        bias -= lr * (grad_b / n_samples)

    return weights, bias


def evaluate(y_true, y_pred, y_probs):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
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
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "specificity": round(specificity * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(auc * 100, 2),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }


def main():
    print("Loading mastitis training dataset...")
    X_raw, y, feature_names = load_dataset(DATASET_PATH)

    # 80/20 train-test split
    random.seed(42)
    healthy = [i for i, v in enumerate(y) if v == 0]
    mastitis = [i for i, v in enumerate(y) if v == 1]
    random.shuffle(healthy)
    random.shuffle(mastitis)

    train_idx = healthy[:int(0.8 * len(healthy))] + mastitis[:int(0.8 * len(mastitis))]
    test_idx = healthy[int(0.8 * len(healthy)):] + mastitis[int(0.8 * len(mastitis)):]

    X_train_raw = [X_raw[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test_raw = [X_raw[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    means, stds = calculate_mean_and_std(X_train_raw)
    X_train = scale_features(X_train_raw, means, stds)
    X_test = scale_features(X_test_raw, means, stds)

    print("Training model...")
    weights, bias = train_model(X_train, y_train, lr=0.08, epochs=1000, l2_penalty=0.005)

    test_probs = [sigmoid(sum(weights[j] * X_test[i][j] for j in range(len(weights))) + bias) for i in range(len(X_test))]
    test_preds = [1 if p >= 0.50 else 0 for p in test_probs]

    results = evaluate(y_test, test_preds, test_probs)
    print("\nModel Evaluation Results (20% Holdout):")
    print(f"  Accuracy:    {results['accuracy']}%")
    print(f"  Sensitivity: {results['recall']}%")
    print(f"  Specificity: {results['specificity']}%")
    print(f"  ROC-AUC:     {results['roc_auc']}%")

    abs_weights = [abs(w) for w in weights]
    total_w = sum(abs_weights)
    importances = {name: round((w / total_w) * 100, 2) for name, w in zip(feature_names, abs_weights)}

    metadata = {
        "model_type": "Logistic Regression Sensor Classifier",
        "features": feature_names,
        "means": [round(m, 4) for m in means],
        "stds": [round(s, 4) for s in stds],
        "standardized_weights": [round(w, 4) for w in weights],
        "bias": round(bias, 4),
        "feature_importances": importances,
        "metrics": results
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
