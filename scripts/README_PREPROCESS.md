Preprocessing script
====================

This folder contains a small script to preprocess JPG/JPEG images found under the repository `data/` folder and write the results to `data_preprocessed/`.

Install dependencies (inside the devcontainer):

```bash
python -m pip install --user -r /workspaces/NeuroSight__AI/scripts/requirements.txt
```

Quick test (process up to 10 files):

```bash
python /workspaces/NeuroSight__AI/scripts/preprocess_images.py --input-dir data/neurosight_data/Training --output-dir data_preprocessed --max-files 10 --img-size 256 --save-as npy
```

Process entire dataset:

```bash
python /workspaces/NeuroSight__AI/scripts/preprocess_images.py --input-dir data/neurosight_data --output-dir data_preprocessed --img-size 256 --save-as npy
```

By default the script will mirror the folder structure from `--input-dir` into `--output-dir` and save `.npy` files containing HWC float32 arrays normalized to [0,1].
