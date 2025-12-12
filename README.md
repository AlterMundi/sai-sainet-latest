# SAINet – Latest smoke detection checkpoint for the SAI

This repository always publishes the **latest production checkpoint** of the computer vision model
used in the **Sistema de Alerta de Incendios (SAI)**.

The repo is intentionally minimal: it exposes a single checkpoint file (`model/model.pt`) plus a
compact configuration file so that others can **download, run and reproduce** the model under the
terms of the license.

> **Current checkpoint**
>
>
> - **Model name**: SAINet  
> - **Architecture**: Ultralytics YOLOv11m (medium)  
> - **Checkpoint path**: `model/model.pt`  
> - **Training hyperparameters**: `config/train_hyperparams.yaml`  
> - **Main training dataset**: SAINetset 6.1 (smoke / non-smoke)  
> - **Evaluation datasets**:
>   - Test split of SAINetset 6.1 (generalization)
>   - In-situ SAI dataset (La Rancherita node, Córdoba, Argentina)

---

## Repository layout

The repository is kept deliberately small:

- `LICENSE`  
  The **GNU Affero General Public License v3.0 (AGPL-3.0)** that applies to this model.

- `README.md`  
  This file. Explains the purpose of the repo, datasets, benchmarks and how to use the current
  checkpoint.

- `config/train_hyperparams.yaml`  
  YOLO training hyperparameters for the current checkpoint (task, epochs, batch size, image size,
  optimizer, data augmentations, etc.). Use this file as the single source of truth for the
  training configuration.

- `model/model.pt`  
  The **only checkpoint** published here. It always corresponds to the **latest production model**
  used in the SAI system.

> Previous checkpoints are not stored as extra files in this repository.  

---

## Quick usage

The checkpoint is a **Ultralytics YOLOv11** model trained for a **single detection class**: `smoke`.

### Python example

```python
from ultralytics import YOLO

# Load the latest SAINet checkpoint from this repo
model = YOLO("model/model.pt")

# Run inference on an image
results = model("path/to/an/image.jpg", conf=0.25)
results.show()  # or results.save()
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

Training is done with **Ultralytics YOLOv11** on top of **PyTorch**, starting from a COCO-pretrained
YOLOv11m (medium) backbone and neck.

All relevant training settings are stored in:

- `config/train_hyperparams.yaml`

This YAML file is meant to be the **canonical training config** for this checkpoint and includes
(at least):

- task and model type (e.g. `task: detect`, `model: yolo11m`),
- list of classes used for training (SAINet trains only on `smoke`),
- number of epochs, batch size, image size,
- early stopping / patience,
- optimizer and learning rate schedule,
- data augmentations.

---

## Datasets

### Current SAINet training dataset (SAINetset 6.1)

The main training dataset, **SAINetset 6.1**, is a curated smoke / non-smoke dataset designed for
outdoor, long-range wildfire smoke detection. It combines:

1. **D-Fire** – a drone-captured fire & smoke dataset with realistic forest fire scenes.
2. **SAI field images** – frames from SAI cameras, including:
   - confirmed smoke events (true positives),
   - historical false positives (clouds, haze, reflections, steam, etc.),
   - “clean” negative frames without smoke.
3. **Synthetic smoke data** – photo-realistic smoke composited over real SAI backgrounds using open
   community datasets (e.g. Pyro-SDIS) and modern image editing models.

SAINetset 6.1 is defined at the level of **images**:

- **Positive image**: at least one smoke bounding box.  
- **Negative image**: no smoke annotations.

#### Composition by split (images)

> These values refer to the current SAINetset 6.1 and are independent of the specific checkpoint.

| Split   | Total images | Positives | Negatives | Positive % | Negative % |
|---------|-------------:|----------:|----------:|-----------:|-----------:|
| Train   | 49,928       | 36,011    | 13,917    | 72.1 %     | 27.9 %     |
| Val     | 6,687        | 4,599     | 2,088     | 68.8 %     | 31.2 %     |
| Test    | 4,306        | 2,301     | 2,005     | 53.4 %     | 46.6 %     |
| Overall | 60,921       | 42,911    | 18,010    | 70.4 %     | 29.6 %     |

The dataset is intentionally rich in:

- diverse **positive smoke patterns**, and  
- **hard negatives** that are typical sources of false alarms in real SAI deployments.

### In-situ evaluation dataset (La Rancherita)

To evaluate performance under real operating conditions, the model is also tested on a separate
**in-situ dataset**:

- Images captured by a production SAI node in **La Rancherita (Córdoba, Argentina)**.
- Contains:
  - a small number of confirmed smoke events (positives), and
  - a large number of typical background frames (hills, vegetation, clouds, changing lighting).
- **No synthetic data** and no images from other regions.
- This dataset is used **only for evaluation**, not for training.

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
- mAP@50–95 = mean AP averaged over IoU thresholds from 0.5 to 0.95  

Two inference configurations are reported:

- **Recall-oriented** – tuned to minimize false negatives (FN ≈ 0), accepting more false positives.  
- **Precision-oriented** – tuned to reduce false positives while maintaining acceptable recall.


### 1. SAINetset 6.1 – test split

| Config                                   | P      | R      | F1     | mAP@50 | mAP@50–95 |
|------------------------------------------|--------|--------|--------|--------|-----------|
| Recall-oriented (conf=0.15, IoU=0.10)    | 0.833  | 0.797  | 0.815  | 0.842  | 0.494     |
| Precision-oriented (conf=0.15, IoU=0.40) | 0.854  | 0.774  | 0.812  | 0.844  | 0.491     |

### 2. In-situ dataset – La Rancherita

| Config                                   | P      | R      | F1     | mAP@50 | mAP@50–95 |
|------------------------------------------|--------|--------|--------|--------|-----------|
| Recall-oriented (conf=0.25, IoU=0.70)    | 0.859  | 0.861  | 0.860  | 0.906  | 0.697     |
| Precision-oriented (conf=0.25, IoU=0.10) | 0.883  | 0.842  | 0.862  | 0.900  | 0.697     |


---

## License

This model and all files in this repository are released under the
**GNU Affero General Public License v3.0 (AGPL-3.0)**.

If you use this model to provide a service to third parties over a network, the AGPL-3.0 requires
you to **make the corresponding source code available**, including any local modifications, in
accordance with the license terms.
