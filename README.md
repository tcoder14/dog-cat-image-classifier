# Dog vs Cat Image Classifier

Binary image classification (dog vs. cat) with **TensorFlow/Keras**, comparing three approaches on the same data and the same training setup:

| # | Approach | Backbone |
|---|----------|----------|
| 1 | Custom CNN trained from scratch | – |
| 2 | Feature extraction (linear probing, frozen backbone) | MobileNetV2 (ImageNet) |
| 3 | Full fine-tuning (all layers trainable) | EfficientNetB0 (ImageNet) |

The whole project lives in a single Google Colab notebook: [`Dog_Cat_Image_Classifier.ipynb`](Dog_Cat_Image_Classifier.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tcoder14/dog-cat-image-classifier/blob/main/Dog_Cat_Image_Classifier.ipynb)

---

## Results

All metrics are measured on the held-out test set (5,000 images, 2,500 per class) unless noted. Training loss/accuracy are re-evaluated on the clean, un-augmented training set. Precision, recall and F1 treat "dog" as the positive class.

| Metric | Custom CNN | Feature Extraction (MobileNetV2) | Fine-Tuned (EfficientNetB0) |
|---|---:|---:|---:|
| Total parameters | 1,208,545 | 2,422,081 | 4,050,852 |
| Training loss | 0.4627 | 0.0201 | 0.0063 |
| Training accuracy | 0.7801 | 0.9927 | 0.9986 |
| Validation loss | 0.4646 | 0.0362 | 0.0246 |
| Validation accuracy | 0.7791 | 0.9872 | 0.9892 |
| Test loss | 0.4616 | 0.0323 | **0.0292** |
| Test accuracy | 0.7834 | 0.9884 | **0.9906** |
| Test precision | 0.8084 | 0.9904 | 0.9881 |
| Test recall | 0.7428 | 0.9864 | 0.9932 |
| Test F1 score | 0.7742 | 0.9884 | **0.9906** |
| Misclassified test images | 1,083 | 58 | 47 |

### Key findings

- **Fine-tuned EfficientNetB0 is the most accurate** (99.06% test accuracy, 47 errors out of 5,000), but only slightly ahead of MobileNetV2 feature extraction (98.84%, 58 errors).
- **Transfer learning is the big win.** Both pretrained models beat the custom CNN by about 20 percentage points.
- **The custom CNN is under-trained, not overfit.** Its training accuracy (~78%) matches its validation accuracy, and early stopping (patience 3) ended training at epoch 12 while the training accuracy was still rising.
- **MobileNetV2 feature extraction is the most efficient.** It trains only 164K parameters and converged in 8 epochs.
- **Remaining errors are mostly hard cases:** animals behind cage bars or fences, close-ups of puppies or kittens, several animals in one image, and cluttered backgrounds. A few appear to be mislabeled images in the dataset.

### Deployment takeaway

| Scenario | Suggested model |
|---|---|
| Best accuracy, server/GPU available | Fine-tuned EfficientNetB0 |
| Mobile/edge, small and cheap to run | MobileNetV2 feature extraction (about 0.2 pp less accurate) |
| Either of the above | Not the custom CNN |

---

## Dataset

[laxmimerit/dog-cat-full-dataset](https://github.com/laxmimerit/dog-cat-full-dataset), cloned directly inside the notebook:

```
dog-cat-full-dataset/data/
├── train/   (cats/, dogs/)  → 19,989 images
└── test/    (cats/, dogs/)  →  5,000 images
```

| Split | Images |
|---|---:|
| Training (80% of `train/`) | 15,992 |
| Validation (20% of `train/`) | 3,997 |
| Test | 5,000 |

Class names: `['cats', 'dogs']` (label 0 = cat, label 1 = dog).

---

## Pipeline

**Data loading** (`tf.keras.utils.image_dataset_from_directory`)
- Images resized to 224 × 224, batch size 32, binary labels.
- 80/20 train/validation split with a fixed seed (42); the test set is not shuffled so predictions line up with labels.
- Pixel values normalized to [0, 1] and datasets optimized with `prefetch()`.

**Data augmentation** (training set only)
- `RandomFlip` (horizontal), `RandomRotation(0.1)`, `RandomZoom(0.15)`, `RandomTranslation(0.1, 0.1)`, `RandomContrast(0.2)`.
- Applied to raw 0–255 pixels before normalization. Validation and test data are never augmented.

**Models**

1. **Custom CNN:** 4 blocks (32 → 64 → 128 → 256 filters), each 2 × (Conv → BatchNorm → ReLU) → MaxPool → Dropout(0.25). Head: GlobalAveragePooling → Dense(128) → BatchNorm → Dropout(0.5) → Dense(1, sigmoid). Adam, lr = 1e-3.
2. **Feature extraction (MobileNetV2):** ImageNet weights, classifier removed, backbone frozen. New head: GlobalAveragePooling → Dense(128) → Dropout(0.3) → Dense(1, sigmoid). Adam, lr = 1e-3. A `Rescaling(2, -1)` layer maps [0, 1] inputs to the [-1, 1] range MobileNetV2 expects.
3. **Full fine-tuning (EfficientNetB0):** ImageNet weights, classifier replaced, **all layers unfrozen**. New head: GlobalAveragePooling → Dropout(0.3) → Dense(1, sigmoid). Adam, lr = 5e-5. A `Rescaling(255)` layer restores the [0, 255] range EfficientNetB0 expects.

**Training:** every model uses binary cross-entropy and the same callback:

```python
tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
```

with a maximum of 30 epochs. Early stopping ended training at epoch 12 (custom CNN), 8 (feature extraction) and 9 (fine-tuning).

**Error analysis:** a reusable `error_analysis(model, test_dataset, class_names)` function that predicts on the test set, prints a classification report, plots a confusion matrix, and displays the 9 most confidently misclassified images with actual label, predicted label and confidence.

---

## Notebook contents

1. Dataset preparation
2. Data augmentation
3. Custom CNN
4. Transfer learning – feature extraction (and why freezing the backbone helps)
5. Full fine-tuning (and feature extraction vs. fine-tuning)
6. Model comparison (table, chart and discussion)
7. Error analysis
8. Observations and conclusion

---

## How to run

### Google Colab 

1. Click the **Open in Colab** badge above (or upload the notebook to [Google Colab](https://colab.research.google.com/)).
2. Set the runtime to GPU: `Runtime → Change runtime type → GPU (T4)`.
3. Run all cells: `Runtime → Run all`.

The dataset is downloaded automatically by the first cells (`git clone`, about 540 MB).

> ⏱️ On a free Colab T4 each epoch takes roughly 3.5–5 minutes, so a full run takes about 2 hours in total. The input pipeline with augmentation is a likely bottleneck, so per-epoch time was similar for all three models.


## Requirements

- Python 3
- TensorFlow / Keras (the notebook was run with TensorFlow 2.21.0)
- scikit-learn
- pandas
- matplotlib
- NumPy

---

## Possible improvements

- Test-time augmentation and ensembling the two pretrained models
- Higher input resolution
- Cleaning mislabeled images in the dataset
- A stronger from-scratch baseline (e.g., a lower learning rate or a learning-rate schedule)
