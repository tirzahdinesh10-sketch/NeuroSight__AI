#!/usr/bin/env python3
"""Compute confusion matrix using latest model and save to models/ as PNG and .npy
"""
from pathlib import Path
import argparse
import logging
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def latest(models_dir: Path, prefix: str):
    files = sorted(models_dir.glob(f"{prefix}_*.joblib"))
    return files[-1] if files else None


def list_npy(root: Path):
    return [p for p in root.rglob('*.npy') if p.is_file()]


def load_batch(files):
    X=[]; y=[]
    for p in files:
        try:
            arr = np.load(p)
        except Exception:
            continue
        X.append(arr.ravel()); y.append(p.parent.name)
    if not X:
        return np.zeros((0,)), []
    return np.stack(X), y


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('data_preprocessed'))
    p.add_argument('--models-dir', type=Path, default=Path('models'))
    p.add_argument('--out-name', type=str, default='confusion_matrix_v1')
    args = p.parse_args(argv)

    models_dir = args.models_dir
    clf_path = latest(models_dir, 'classifier')
    pca_path = latest(models_dir, 'ipca')
    if clf_path is None or pca_path is None:
        logging.error('Missing model or PCA in models/')
        return 2
    clf = joblib.load(clf_path)
    pca = joblib.load(pca_path)

    test_dir = args.data_dir / 'Testing'
    files = list_npy(test_dir)
    logging.info(f'Found {len(files)} test files')
    X_test, y_test = load_batch(files)
    if X_test.size == 0:
        logging.error('No test data')
        return 3

    Xred = pca.transform(X_test)
    ypred = clf.predict(Xred)
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, ypred, labels=labels)

    out_png = models_dir / f"{args.out_name}.png"
    out_npy = models_dir / f"{args.out_name}.npy"

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_png)
    np.save(out_npy, cm)

    logging.info(f'Saved confusion matrix PNG to {out_png} and numpy to {out_npy}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
