#!/usr/bin/env python3
"""Train a classifier on preprocessed .npy images.

Pipeline:
- Load .npy images from data_preprocessed/Training and data_preprocessed/Testing
- Labels are taken from the parent folder name (e.g., glioma, notumor)
- Use IncrementalPCA to reduce dimensionality
- Train a LogisticRegression classifier on reduced features
- Save classifier, PCA, and evaluation metrics to models/
"""
from pathlib import Path
import argparse
import logging
import shutil
import sys
import time

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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


def iter_files_in_batches(files, batch_size):
    for i in range(0, len(files), batch_size):
        yield files[i : i + batch_size]


def main(argv=None):
    p = argparse.ArgumentParser(description="Train classifier from preprocessed .npy images")
    p.add_argument("--data-dir", type=Path, default=Path("data_preprocessed"), help="Preprocessed data dir")
    p.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory to save models")
    p.add_argument("--pca-components", type=int, default=256, help="Number of PCA components")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for IncrementalPCA")
    p.add_argument("--clean", action="store_true", help="Clean models dir before saving")
    args = p.parse_args(argv)

    train_dir = args.data_dir / "Training"
    test_dir = args.data_dir / "Testing"

    if not train_dir.exists():
        logging.error(f"Training directory not found: {train_dir}")
        return 2

    args.models_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        logging.info(f"Cleaning models dir {args.models_dir}")
        shutil.rmtree(args.models_dir)
        args.models_dir.mkdir(parents=True, exist_ok=True)

    train_files = list_npy_files(train_dir)
    test_files = list_npy_files(test_dir) if test_dir.exists() else []

    logging.info(f"Found {len(train_files)} training files and {len(test_files)} testing files")

    # Fit IncrementalPCA on training data
    ipca = IncrementalPCA(n_components=args.pca_components)

    # First pass: partial fit
    for batch_files in iter_files_in_batches(train_files, args.batch_size):
        X_batch, _ = load_batch(batch_files)
        if X_batch.size == 0:
            continue
        ipca.partial_fit(X_batch)

    # Transform training data
    X_train = []
    y_train = []
    for batch_files in iter_files_in_batches(train_files, args.batch_size):
        Xb, yb = load_batch(batch_files)
        if Xb.size == 0:
            continue
        X_train.append(ipca.transform(Xb))
        y_train.extend(yb)
    if not X_train:
        logging.error("No training data found after loading")
        return 3
    X_train = np.vstack(X_train)

    # Transform testing data
    X_test = []
    y_test = []
    if test_files:
        for batch_files in iter_files_in_batches(test_files, args.batch_size):
            Xb, yb = load_batch(batch_files)
            if Xb.size == 0:
                continue
            X_test.append(ipca.transform(Xb))
            y_test.extend(yb)
        if X_test:
            X_test = np.vstack(X_test)

    logging.info(f"Training feature matrix: {X_train.shape}")

    # Train classifier
    clf = LogisticRegression(max_iter=2000, solver='saga', multi_class='multinomial', n_jobs=-1)
    logging.info("Training classifier...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    logging.info(f"Training completed in {t1-t0:.1f}s")

    # Evaluate
    report = {}
    y_pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    report['train_accuracy'] = acc_train
    report['train_classification_report'] = classification_report(y_train, y_pred_train)

    if test_files and X_test.size:
        y_pred_test = clf.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        report['test_accuracy'] = acc_test
        report['test_classification_report'] = classification_report(y_test, y_pred_test)
        report['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)

    # Save artifacts
    timestamp = int(time.time())
    model_path = args.models_dir / f'classifier_{timestamp}.joblib'
    pca_path = args.models_dir / f'ipca_{timestamp}.joblib'
    metrics_path = args.models_dir / f'metrics_{timestamp}.txt'

    joblib.dump(clf, model_path)
    joblib.dump(ipca, pca_path)

    with open(metrics_path, 'w') as f:
        f.write(f"Train accuracy: {report['train_accuracy']}\n\n")
        f.write("Train classification report:\n")
        f.write(report['train_classification_report'])
        f.write('\n')
        if 'test_accuracy' in report:
            f.write(f"Test accuracy: {report['test_accuracy']}\n\n")
            f.write("Test classification report:\n")
            f.write(report['test_classification_report'])
            f.write('\nConfusion matrix:\n')
            f.write(np.array2string(report['confusion_matrix']))

    logging.info(f"Saved classifier to {model_path}")
    logging.info(f"Saved PCA to {pca_path}")
    logging.info(f"Saved metrics to {metrics_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
