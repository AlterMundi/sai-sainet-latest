# SAINet - Latest smoke detection checkpoint for the SAI

This repository always publishes the **latest production checkpoint** of the computer vision model
used in the **[Sistema de Alerta de Incendios (SAI)](https://sainet.info)**.

The repo is intentionally minimal: it exposes a single checkpoint file (`model/model.pt`) plus a
compact configuration file so that others can **download, run and reproduce** the model under the
terms of the license.

> **Current checkpoint**
>
>
> - **Model name**: SAINet v10.0
> - **Architecture**: Ultralytics YOLOv12m (medium)
> - **Checkpoint path**: `model/SAINet_v10.0.pt`
> - **Training hyperparameters**: `model/train_hyperparams.yaml`
> - **Main training dataset**: [SAINetset v8.0](https://huggingface.co/datasets/SAINetset/SAINetset_v8.0) (~65K images, smoke & fire detection)
> - **Evaluation datasets**:
>   - Validation split of SAINetset 8.0 (generalization)
>   - In-situ SAI datasets (La Rancherita and La Serranita nodes, Cordoba, Argentina)

---

## Repository layout

The repository is kept deliberately small:

- `LICENSE`  
  The **GNU Affero General Public License v3.0 (AGPL-3.0)** that applies to this model.

- `README.md`  
  This file. Explains the purpose of the repo, datasets, benchmarks and how to use the current
  checkpoint.

- `model/train_hyperparams.yaml`  
  YOLO training hyperparameters for the current checkpoint (task, epochs, batch size, image size,
  optimizer, data augmentations, etc.). Use this file as the single source of truth for the
  training configuration.

- `model/model.pt`  
  The **only checkpoint** published here. It always corresponds to the **latest production model**
  used in the SAI system.

> Previous checkpoints are not stored as extra files in this repository.  

---

## Quick usage

The checkpoint is an **Ultralytics YOLOv12** model trained for **two detection classes**: `smoke` and `fire`.

### Python example

```python
from ultralytics import YOLO

# Load the latest SAINet checkpoint from this repo
model = YOLO("model/SAINet_v10.0.pt")

# Run inference on an image (both smoke and fire)
results = model("path/to/an/image.jpg", conf=0.25)
results.show()  # or results.save()

# Run inference for smoke only
results = model("path/to/an/image.jpg", conf=0.25, classes=[0])
```

### Inference notes

- The model is designed for **early wildfire smoke** in outdoor landscapes.
- The operational policy of the SAI is **recall-first** (missing a plume is worse than raising an
  extra alert), so we typically:
  - use **low confidence thresholds**,
  - calibrate **IoU / NMS settings**, and
  - rely on **temporal consensus** at system level to control false positives.

When you deploy this model in a different context, you should:

- Re-tune `conf`, `iou` and NMS options for your scenario.
- Always check performance on:
  - a **held-out test set**, and  
  - a **real-world evaluation set** that matches your deployment conditions.

---

## Training setup

Training is done with **Ultralytics YOLOv12** on top of **PyTorch**, starting from a COCO-pretrained
YOLOv12m (medium) backbone and neck.

All relevant training settings are stored in:

- `model/train_hyperparams.yaml`

This YAML file is meant to be the **canonical training config** for this checkpoint and includes
(at least):

- task and model type (e.g. `task: detect`, `model: yolo12m`),
- list of classes used for training (`smoke` and `fire`),
- number of epochs, batch size, image size,
- optimizer and learning rate schedule.

---

## Datasets

### Current SAINet training dataset (SAINetset 8.0)

> **Download the dataset**: [SAINetset v8.0 on Hugging Face](https://huggingface.co/datasets/SAINetset/SAINetset_v8.0)

The main training dataset, **SAINetset 8.0**, is a curated smoke & fire detection dataset designed for
outdoor, long-range wildfire detection. It combines multiple data sources:

1. **D-Fire** - a drone-captured fire & smoke dataset with realistic forest fire scenes.
2. **Pyronear (pyro-sdis)** - wildfire detection images from the Pyronear project.
3. **SAI field data** - real-world images from deployed SAI nodes in Córdoba, Argentina.

SAINetset 8.0 is defined at the level of **images**:

- **Positive image**: at least one smoke or fire bounding box.
- **Negative image**: no smoke/fire annotations (background only).

#### Composition by split (images)

| Split   | Total images | Positives | Negatives | Positive % | Negative % |
|---------|-------------:|----------:|----------:|-----------:|-----------:|
| Train   | 56,815       | 38,113    | 18,702    | 67.1 %     | 32.9 %     |
| Val     | 7,894        | 4,828     | 3,066     | 61.2 %     | 38.8 %     |
| **Overall** | **64,709** | **42,941** | **21,768** | **66.4 %** | **33.6 %** |

#### Class distribution (bounding boxes)

| Class | Train boxes | Val boxes | Total boxes |
|-------|------------:|----------:|------------:|
| smoke | 41,806      | 5,478     | 47,284      |
| fire  | 13,146      | 1,458     | 14,604      |

The dataset is intentionally rich in:

- diverse **positive smoke and fire patterns**, and
- **hard negatives** that are typical sources of false alarms in real SAI deployments (clouds, haze, reflections, steam, etc.).

#### Download SAINetset

The dataset is publicly available on Hugging Face. You can download it using any of these methods:

**Option 1: Git LFS (recommended)**
```bash
git lfs install
git clone https://huggingface.co/datasets/SAINetset/SAINetset_v8.0
```

**Option 2: Hugging Face Datasets library**
```python
from datasets import load_dataset

ds = load_dataset("SAINetset/SAINetset_v8.0")
sample = ds["train"][0]
```

**Option 3: Direct use with Ultralytics YOLO**
```python
from ultralytics import YOLO

model = YOLO("yolo12m.pt")
model.train(data="path/to/sainetset/data.yaml", epochs=100)
```

### In-situ evaluation datasets

To evaluate performance under real operating conditions, the model is also tested on separate
**in-situ datasets** from production SAI nodes:

- **La Rancherita** (Cordoba, Argentina) - first deployed node.
- **La Serranita** (Cordoba, Argentina) - second deployed node.

These datasets contain:
  - a small number of confirmed smoke events (positives), and
  - a large number of typical background frames (hills, vegetation, clouds, changing lighting).
- **No synthetic data** and no images from other regions.
- These datasets are used **only for evaluation**, not for training.

---

## Benchmarks (current checkpoint)

This section summarizes the performance of the **current checkpoint** on:

1. The **SAINetset 6.1 test split** (generalization on the main dataset).
2. The **in-situ evaluation dataset** (La Rancherita node).

All metrics are for **smoke detection only** (no fire class) using standard Ultralytics metrics:

- P = precision
- R = recall
- F1 = harmonic mean of P and R
- mAP@50 = mean Average Precision at IoU 0.5
- mAP@50-95 = mean AP averaged over IoU thresholds from 0.5 to 0.95

Two inference configurations are reported:

- **Recall-oriented** - tuned to minimize false negatives (FN ~ 0), accepting more false positives.
- **Precision-oriented** - tuned to reduce false positives while maintaining acceptable recall.


### 1. SAINetset 8.0 - test split

| Config                                   | P      | R      | F1     | mAP@50 | mAP@50-95 |
|------------------------------------------|--------|--------|--------|--------|-----------|
| Recall-oriented (conf=0.15, IoU=0.10)    | 0.833  | 0.797  | 0.815  | 0.842  | 0.494     |
| Precision-oriented (conf=0.15, IoU=0.40) | 0.854  | 0.774  | 0.812  | 0.844  | 0.491     |

### 2. In-situ dataset - La Rancherita

| Config                                   | P      | R      | F1     | mAP@50 | mAP@50-95 |
|------------------------------------------|--------|--------|--------|--------|-----------|
| Recall-oriented (conf=0.25, IoU=0.70)    | 0.859  | 0.861  | 0.860  | 0.906  | 0.697     |
| Precision-oriented (conf=0.25, IoU=0.10) | 0.883  | 0.842  | 0.862  | 0.900  | 0.697     |


---

## Related Links

- **SAI Project Website**: [sainet.info](https://sainet.info)
- **Training Dataset**: [SAINetset v8.0 on Hugging Face](https://huggingface.co/datasets/SAINetset/SAINetset_v8.0)
- **Model Repository**: [sai-sainet-latest on GitHub](https://github.com/SAINetset/sai-sainet-latest)

---

## License

This model and all files in this repository are released under the
**GNU Affero General Public License v3.0 (AGPL-3.0)**.

If you use this model to provide a service to third parties over a network, the AGPL-3.0 requires
you to **make the corresponding source code available**, including any local modifications, in
accordance with the license terms.
