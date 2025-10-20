#!/usr/bin/env python3
"""Load models/confusion_matrix_v1.npy and create an annotated PNG with numeric values.
Saves to models/confusion_matrix_v1_annotated.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    models = Path('models')
    arr_path = models / 'confusion_matrix_v1.npy'
    if not arr_path.exists():
        print('confusion .npy not found:', arr_path)
        return 2
    cm = np.load(arr_path)
    labels = ['glioma','meningioma','notumor','pituitary']

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    # annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax)
    out = models / 'confusion_matrix_v1_annotated.png'
    plt.tight_layout()
    plt.savefig(out)
    print('Saved annotated confusion matrix to', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
