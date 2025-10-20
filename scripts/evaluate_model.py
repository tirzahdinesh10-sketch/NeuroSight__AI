#!/usr/bin/env python3
"""Evaluate the latest saved classifier on the Testing set.

Finds the most recent classifier_*.joblib and corresponding ipca_*.joblib under models/ (or use provided paths),
loads the test `.npy` files from `data_preprocessed/Testing`, predicts labels, prints and writes a metrics file.
"""
from pathlib import Path
import argparse
import logging
import sys
import time

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def list_latest(models_dir: Path, prefix: str):
    files = sorted(models_dir.glob(f"{prefix}_*.joblib"))
    return files[-1] if files else None


def list_npy_files(root: Path):
    return [p for p in root.rglob("*.npy") if p.is_file()]


def load_batch(file_paths):
    X = []
    y = []
    for p in file_paths:
        try:
            arr = np.load(p)
        except Exception:
            continue
        X.append(arr.ravel())
        y.append(p.parent.name)
    if not X:
        return np.zeros((0,)), []
    X = np.stack(X)
    return X, y


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate saved classifier on Testing set")
    p.add_argument("--data-dir", type=Path, default=Path("data_preprocessed"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--model", type=Path, default=None, help="Path to classifier joblib (optional)")
    p.add_argument("--pca", type=Path, default=None, help="Path to ipca joblib (optional)")
    args = p.parse_args(argv)

    test_dir = args.data_dir / "Testing"
    if not test_dir.exists():
        logging.error(f"Testing directory not found: {test_dir}")
        return 2

    model_path = args.model or list_latest(args.models_dir, 'classifier')
    pca_path = args.pca or list_latest(args.models_dir, 'ipca')

    if model_path is None or pca_path is None:
        logging.error(f"Could not find model/ipca files under {args.models_dir}. Provide --model and --pca.")
        return 3

    logging.info(f"Loading classifier from {model_path}")
    clf = joblib.load(model_path)
    logging.info(f"Loading PCA from {pca_path}")
    ipca = joblib.load(pca_path)

    test_files = list_npy_files(test_dir)
    logging.info(f"Found {len(test_files)} test files")

    X_test, y_test = load_batch(test_files)
    if X_test.size == 0:
        logging.error("No test data loaded")
        return 4

    X_test_reduced = ipca.transform(X_test)
    y_pred = clf.predict(X_test_reduced)

    acc = accuracy_score(y_test, y_pred)
    crep = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    timestamp = int(time.time())
    out_path = args.models_dir / f'eval_metrics_{timestamp}.txt'
    with open(out_path, 'w') as f:
        f.write(f"Test accuracy: {acc}\n\n")
        f.write("Classification report:\n")
        f.write(crep)
        f.write('\nConfusion matrix:\n')
        f.write(np.array2string(cm))

    logging.info(f"Test accuracy: {acc:.4f}")
    logging.info(f"Saved evaluation metrics to {out_path}")
    print(crep)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
