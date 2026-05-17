# Training README for HOG + SVM and HOG + Random Forest

This document explains the training notebooks in `sign_classify_train/`:

- `sign_classify_train/HOG_SVM/train_hog_svm_6x3.ipynb`
- `sign_classify_train/HOG_SVM/train_hog_svm_8x2.ipynb`
- `sign_classify_train/HOG_RandomForest/train_hog_rf_6x3.ipynb`
- `sign_classify_train/HOG_RandomForest/train_hog_rf_8x2.ipynb`

All four notebooks follow the same high-level pipeline:

1. Read images from `data/train` and `data/test`.
2. Split the original training set into train and validation sets with an 80/20 ratio.
3. Preprocess each image: load, remove alpha if present, convert to grayscale, resize to `128x128`.
4. Extract HOG features from the resized grayscale image.
5. Extract HSV histogram features from the resized color image.
6. Concatenate HOG and HSV into one feature vector.
7. Use Optuna to search for good hyperparameters on the validation set.
8. Retrain the best model on train + validation.
9. Evaluate on the test set and save a `.joblib` payload.

## 1. Problem Definition

The task is traffic sign classification into five classes:

- `Cam`
- `Chidan`
- `Hieulenh`
- `Nguyhiem`
- `Phu`

Instead of feeding raw pixels directly to the classifier, each image is converted into a meaningful numeric feature vector:

- HOG describes shape, edges, contours, and gradient directions.
- HSV histogram describes color distribution, saturation, and brightness.

The final feature vector is then passed into either SVM or Random Forest.

## 2. Dataset and Results

Dataset split:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Cam | 490 | 123 | 68 | 681 |
| Chidan | 454 | 113 | 63 | 630 |
| Hieulenh | 470 | 117 | 65 | 652 |
| Nguyhiem | 473 | 118 | 66 | 657 |
| Phu | 297 | 75 | 41 | 413 |
| Total | 2184 | 546 | 303 | 3033 |

Test-set results:

| Model | Accuracy | Weighted F1 | Macro F1 | Test samples |
|---|---:|---:|---:|---:|
| HOG SVM 8x2 | 0.9142 | 0.9142 | 0.9080 | 303 |
| HOG RF 8x2 | 0.9109 | 0.9106 | 0.9059 | 303 |
| HOG SVM 6x3 | 0.9010 | 0.9009 | 0.8921 | 303 |
| HOG RF 6x3 | 0.8878 | 0.8875 | 0.8821 | 303 |

Main interpretation:

- `HOG SVM 8x2` is the best model in the current experiments.
- `8x2` has fewer dimensions than `6x3`, but generalizes better on the test set.
- `6x3` captures finer local details, but the much longer vector can also capture noise, background variation, crop errors, and small image artifacts.
- SVM works well with HOG because HOG already converts images into continuous, geometry-aware descriptors.

## 3. What Do 6x3 and 8x2 Mean?

In the code, `6x3` and `8x2` are shorthand names for two HOG configurations:

```python
# 6x3
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (6, 6),
    "cells_per_block": (3, 3),
    "block_norm": "L2-Hys",
}

# 8x2
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
```

`6x3` means:

- `6`: each HOG cell is `6x6` pixels.
- `3`: each HOG block contains `3x3` cells.

`8x2` means:

- `8`: each HOG cell is `8x8` pixels.
- `2`: each HOG block contains `2x2` cells.

With resized images of `128x128`, the HOG dimensionality is:

### 8x2

- Cells per axis: `floor(128 / 8) = 16`.
- Blocks per axis: `16 - 2 + 1 = 15`.
- Values per block: `2 x 2 x 9 = 36`.
- HOG descriptors: `15 x 15 x 36 = 8100`.
- HSV descriptors: `16 + 8 + 8 = 32`.
- Total features: `8100 + 32 = 8132`.

### 6x3

- Cells per axis: `floor(128 / 6) = 21`.
- Blocks per axis: `21 - 3 + 1 = 19`.
- Values per block: `3 x 3 x 9 = 81`.
- HOG descriptors: `19 x 19 x 81 = 29241`.
- HSV descriptors: `16 + 8 + 8 = 32`.
- Total features: `29241 + 32 = 29273`.

Meaning:

- `6x3` produces a much longer and more detailed descriptor.
- `8x2` is more compact, faster, and less likely to overfit.
- In the current results, `8x2` wins for both SVM and Random Forest, meaning that extra local detail from `6x3` does not improve test performance here.

## 4. How HOG Works

HOG stands for Histogram of Oriented Gradients. Its goal is to represent object shape by counting gradient directions.

Basic mathematical process:

1. Compute horizontal and vertical gradients:

```text
Gx(x, y) = I(x + 1, y) - I(x - 1, y)
Gy(x, y) = I(x, y + 1) - I(x, y - 1)
```

2. Compute gradient magnitude and angle:

```text
magnitude = sqrt(Gx^2 + Gy^2)
angle = atan2(Gy, Gx)
```

3. Divide the image into cells, for example `8x8` or `6x6` pixels.
4. For each cell, build a histogram of gradient directions. The code uses `orientations=9`, so there are 9 angle bins.
5. Group neighboring cells into blocks, for example `2x2` or `3x3`, and normalize the block vector using `L2-Hys`.
6. Concatenate all block descriptors into the final HOG vector.

Why HOG fits traffic signs:

- Traffic signs have strong shapes: circles, triangles, rectangles, arrows, borders.
- HOG is less dependent on raw color than pixels, because it focuses on edges and gradients.
- Under moderate illumination changes, HOG can preserve shape information better than raw pixels.

## 5. How HSV Histogram Works

HSV contains:

- `H` - Hue: color tone, represented by OpenCV in `[0, 180]`.
- `S` - Saturation: color intensity, in `[0, 256]`.
- `V` - Value: brightness, in `[0, 256]`.

The code uses:

```python
HSV_HIST_BINS = (16, 8, 8)
```

This means:

- Hue is divided into 16 bins.
- Saturation is divided into 8 bins.
- Value is divided into 8 bins.
- Total HSV dimensions = `16 + 8 + 8 = 32`.

The code does not compute a joint 3D histogram `H x S x V`. Instead, it computes one histogram per channel and concatenates them:

```python
hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).flatten()
hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).flatten()
hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).flatten()
color_feature = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
color_feature /= color_feature.sum() + 1e-8
```

Why HSV is useful:

- Traffic signs often have distinctive colors: red, blue, yellow, white, black.
- HSV separates hue from brightness better than RGB.
- HSV complements HOG: HOG captures shape, HSV captures color.

## 6. Line-by-Line Explanation of Shared Cells

This section applies to both SVM and Random Forest notebooks, because preprocessing, feature extraction, splitting, visualization, and evaluation are shared.

### Cell 1: Imports

```python
from __future__ import annotations
```

Enables modern type hints and postpones annotation evaluation.

```python
from pathlib import Path
```

Provides object-oriented path handling.

```python
import cv2
```

OpenCV is used for RGB resizing, RGB-to-HSV conversion, HSV histograms, color mapping, and dilation for HOG visualization.

```python
import joblib
```

Used to save trained models and metadata as `.joblib` files.

```python
import matplotlib.pyplot as plt
```

Used for plots: data distribution, image samples, HOG visualization, confusion matrix, and prediction examples.

```python
import numpy as np
```

Used for arrays, feature vectors, stacking, concatenation, and numeric operations.

```python
import optuna
```

Used for automatic hyperparameter optimization.

```python
import pandas as pd
```

Used for tables such as dataset counts, Optuna trials, feature dimensions, and misclassification summaries.

```python
from joblib import Parallel, delayed
```

Used to extract features from many images in parallel.

```python
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
```

`MedianPruner` stops weak trials early. `TPESampler` proposes new hyperparameters based on previous trial results.

```python
from skimage import exposure
```

Imported but not directly used in the current notebooks.

```python
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
```

These functions convert RGB to grayscale, compute HOG descriptors, read image files, and resize grayscale images.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
```

These are used for evaluation, train/validation splitting, and converting string labels into integer labels.

```python
from tqdm import tqdm
```

Displays a progress bar during feature extraction.

The only model-specific import differs:

- SVM notebooks import `from sklearn.svm import SVC`.
- Random Forest notebooks import `from sklearn.ensemble import RandomForestClassifier`.

### Cell 2: Data Directory Resolution and Path Collection

```python
def resolve_data_dir() -> Path:
```

Defines a function that returns the path to the `data` directory.

```python
here = Path.cwd().resolve()
```

Gets the current working directory as an absolute path.

```python
for parent in [here, *here.parents]:
```

Searches from the current directory upward through all parent directories. This makes the notebook robust to being run from the project root or from a subfolder.

```python
candidate = parent / "data"
```

Creates a possible `data` path.

```python
if (candidate / "train").is_dir() and (candidate / "test").is_dir():
    return candidate
```

Accepts the candidate only if both `data/train` and `data/test` exist.

```python
raise FileNotFoundError("Could not find data/train and data/test")
```

Raises a clear error if the dataset cannot be found.

```python
def collect_paths(split_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
```

Reads one split folder, such as `data/train`, and returns image paths, labels, and class names.

```python
labels = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
```

Each subfolder name is treated as a class label. Sorting keeps class order deterministic.

```python
if not labels:
    raise RuntimeError(...)
```

Stops if no class folders exist.

```python
paths: list[Path] = []
y: list[str] = []
```

Initializes image paths and labels.

```python
for label in labels:
    for path in sorted((split_dir / label).glob("*.*")):
        if path.is_file():
            paths.append(path)
            y.append(label)
```

For each class folder, the code collects all files and assigns the folder name as the label.

```python
if not paths:
    raise RuntimeError(...)
```

Stops if no image files are found.

```python
return paths, np.array(y), labels
```

Returns paths, labels as a NumPy array, and class names.

### Cell 3: Flexible Table Display

```python
def show_table(df: pd.DataFrame) -> None:
```

Defines a helper for displaying a DataFrame.

```python
try:
    display(df)
except NameError:
    print(df.to_string(index=False))
```

Uses Jupyter's rich `display` when available; otherwise prints a plain text table.

### Cell 4: Image Preprocessing and Per-Image Feature Extraction

#### `load_image_steps`

```python
def load_image_steps(path: Path, image_size: tuple[int, int] | None = None):
```

Loads one image and returns the original image, grayscale image, and resized grayscale image.

```python
if image_size is None:
    image_size = IMAGE_SIZE
```

Defaults to the global `IMAGE_SIZE = (128, 128)`.

```python
image = imread(path)
```

Reads the image file.

```python
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[..., :3]
```

Removes the alpha channel if the image is RGBA.

```python
if image.ndim == 3:
    gray = rgb2gray(image)
```

Converts color images to grayscale for HOG.

```python
else:
    gray = image.astype(np.float32)
    if gray.max() > 1:
        gray = gray / 255.0
```

If the image is already grayscale, converts it to float and normalizes 0-255 data to 0-1.

```python
resized = resize(gray, image_size, anti_aliasing=True)
```

Resizes the grayscale image to `128x128`. Anti-aliasing reduces resize artifacts.

```python
return image, gray, resized
```

Returns all three stages for both training and visualization.

#### `to_uint8_rgb`

This helper ensures that the image is RGB `uint8` in `[0, 255]` before OpenCV HSV conversion.

```python
if image.ndim == 2:
    image = np.stack([image, image, image], axis=-1)
```

Converts grayscale into three identical channels.

```python
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[..., :3]
```

Removes alpha if present.

```python
rgb = image.astype(np.float32)
if rgb.max() <= 1.0:
    rgb = rgb * 255.0
return np.clip(rgb, 0, 255).astype(np.uint8)
```

Converts to float, rescales 0-1 images to 0-255, clips values, and casts to `uint8`.

#### `resize_rgb_image`

```python
rgb = to_uint8_rgb(image)
height, width = image_size
return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
```

OpenCV expects size as `(width, height)`, while the project stores image size as `(height, width)`. `INTER_AREA` is appropriate for image resizing, especially downsampling.

#### `enhance_hog_image`

This function only improves HOG visualization. It does not change the actual HOG feature vector used for training.

```python
image = np.asarray(hog_image, dtype=np.float32)
image = np.nan_to_num(...)
image = np.maximum(image, 0.0)
```

Converts the visualization to float, replaces invalid values with zero, and removes negative values.

```python
if float(image.max()) <= float(image.min()):
    return np.full((*image.shape, 3), 246, dtype=np.uint8)
```

If the image has no contrast, returns a light gray canvas.

```python
active = image[image > 0]
```

Keeps pixels with positive HOG signal.

```python
low = np.percentile(active, 5.0)
high = np.percentile(active, 99.5)
```

Uses percentiles to reduce the influence of outliers.

```python
image = np.clip((image - low) / (high - low + 1e-8), 0.0, 1.0)
image = np.power(image, 0.32)
```

Normalizes to 0-1 and applies gamma correction so weak edges become more visible.

```python
image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
image_u8 = cv2.dilate(image_u8, np.ones((2, 2), dtype=np.uint8), iterations=1)
```

Converts to 8-bit and thickens the HOG strokes for visualization.

```python
color_bgr = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
```

Applies a color map and converts BGR to RGB for Matplotlib.

```python
canvas = np.full((*image_u8.shape, 3), 246, dtype=np.uint8)
mask = image_u8 > 10
alpha = (image_u8.astype(np.float32) / 255.0)[..., None]
blended = canvas * (1 - alpha) + color_rgb * alpha
canvas[mask] = blended[mask].astype(np.uint8)
return canvas
```

Blends the colored HOG response onto a light canvas and keeps only meaningful pixels.

#### `extract_hog_visual`

```python
descriptors, hog_image = hog(gray_resized, visualize=True, **hog_params)
```

Computes both the HOG descriptor and its visualization. `**hog_params` passes `orientations`, `pixels_per_cell`, `cells_per_block`, and `block_norm`.

```python
hog_image = enhance_hog_image(hog_image)
return descriptors, hog_image
```

Enhances the visualization and returns both outputs.

#### `extract_hsv_histogram`

```python
rgb_resized = resize_rgb_image(image, image_size)
hsv = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2HSV)
```

Resizes the color image and converts it to HSV.

```python
h_bins, s_bins, v_bins = hsv_bins
```

Unpacks the number of bins for each channel.

```python
hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).flatten()
hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).flatten()
hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).flatten()
```

Computes one histogram per channel and flattens each into a one-dimensional vector.

```python
color_feature = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
color_feature /= color_feature.sum() + 1e-8
return color_feature
```

Concatenates the three histograms into a 32-dimensional vector and normalizes the sum to 1.

#### `extract_combined_feature`

```python
hog_feature = hog(gray_resized, **hog_params).astype(np.float32)
color_feature = extract_hsv_histogram(image, image_size, hsv_bins)
return np.concatenate([hog_feature, color_feature])
```

Computes the training HOG vector, computes HSV color features, and concatenates them.

The final vector has the form:

```text
X_one = [HOG_0, HOG_1, ..., HOG_n, HSV_0, HSV_1, ..., HSV_31]
```

#### `split_combined_feature`

```python
return feature[:hog_dim], feature[hog_dim:hog_dim + color_dim]
```

Splits a combined vector back into HOG and HSV parts. It is a helper and is not central to the training flow in the current notebooks.

#### `choose_representative_sample`

This function selects a sample image for visualization. If `preferred_label` exists, it returns the first image from that class; otherwise it returns the first available image.

### Cell 5: Feature Extraction for a Full Split

```python
def compute_features(...):
```

Receives image paths and labels, then returns feature matrix `X` and label vector `y`.

```python
if cache_file is not None and cache_file.exists():
```

If a cache file exists, the function tries to reuse it.

```python
with np.load(cache_file) as data:
    X_cached = data["X"]
    y_cached = data["y"]
```

Loads cached features and labels.

```python
if len(y_cached) == len(labels):
    return X_cached, y_cached
```

Uses the cache only if the sample count matches.

```python
def extract(path: Path) -> np.ndarray | None:
```

Defines an inner function for extracting one image feature vector.

```python
try:
    image, _, gray_resized = load_image_steps(path, image_size)
    return extract_combined_feature(...)
except Exception:
    return None
```

If an image cannot be read or processed, the function returns `None` so the whole run does not crash.

```python
features = Parallel(n_jobs=n_jobs)(
    delayed(extract)(p) for p in tqdm(paths, desc=desc, total=len(paths))
)
```

Extracts features in parallel with a progress bar.

```python
kept_feats = []
kept_labels = []
skipped = 0
```

Initializes storage for valid samples and skipped count.

```python
for feat, label in zip(features, labels):
    if feat is None:
        skipped += 1
        continue
    kept_feats.append(feat)
    kept_labels.append(label)
```

Keeps valid features and skips failed images.

```python
if not kept_feats:
    raise RuntimeError(...)
```

Stops if no features were extracted.

```python
X = np.vstack(kept_feats)
y = np.array(kept_labels)
```

Stacks feature vectors into a 2D matrix:

```text
X.shape = (num_images, num_features)
```

```python
np.savez_compressed(cache_file, X=X, y=y)
```

Saves compressed features for reuse.

```python
return X, y
```

Returns the aligned features and labels.

### Cell 6: Dataset Distribution

`plot_dataset_distribution` counts images by class and split, displays a pivot table, and plots a grouped bar chart.

Why this matters:

- A highly imbalanced dataset can make accuracy misleading.
- This is why the report also includes macro F1 and weighted F1.

### Cell 7: Class Sample Visualization

`plot_class_samples` displays up to `max_per_class=4` sample images for each class. It helps check:

- Whether labels look correct.
- Whether there are corrupted, blurry, or badly cropped images.
- Which classes have high visual variation.

### Cell 8: HOG + HSV Pipeline Visualization

`plot_preprocessing_and_hog` shows five panels:

1. Original image.
2. Grayscale image.
3. Resized `128x128` image.
4. HOG visualization.
5. HSV histogram.

The key lines are:

```python
hog_descriptors, hog_image = extract_hog_visual(resized, HOG_PARAMS)
color_descriptors = extract_hsv_histogram(image, IMAGE_SIZE, HSV_HIST_BINS)
combined = np.concatenate([hog_descriptors.astype(np.float32), color_descriptors])
```

They show exactly how the final feature vector is built.

### Cell 9: Feature Overview

```python
color_dim = sum(HSV_HIST_BINS)
hog_dim = X_train.shape[1] - color_dim
```

HSV always has 32 dimensions. HOG dimensions equal total dimensions minus 32.

The table displays:

- Number of samples in train/validation/test.
- Number of HOG descriptors.
- Number of HSV descriptors.
- Total feature dimensions.

## 7. SVM-Specific Explanation

### SVM Cell 10: Optuna Objective

```python
def objective(trial: optuna.Trial) -> float:
```

Each trial is one hyperparameter configuration.

```python
kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
```

The search tries two kernels:

- `linear`: a linear separating hyperplane.
- `rbf`: a nonlinear kernel that can model curved decision boundaries.

```python
params = {
    "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
    "kernel": kernel,
    "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    "shrinking": trial.suggest_categorical("shrinking", [True, False]),
    "probability": False,
    "cache_size": SVM_CACHE_SIZE_MB,
    "random_state": SEED,
}
```

Parameter meanings:

- `C`: penalty strength. Larger `C` tries to fit training data more strictly; smaller `C` allows a wider margin and more regularization.
- Log-scale search from `0.01` to `100` is used because `C` acts by orders of magnitude.
- `class_weight=None`: all classes have equal weight.
- `class_weight="balanced"`: minority classes receive larger weights.
- `shrinking`: enables or disables a training-speed heuristic.
- `probability=False`: disabled during search for speed.
- `cache_size=1000`: gives SVM 1000 MB kernel cache.
- `random_state=42`: improves reproducibility.

```python
if kernel == "rbf":
    params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
```

`gamma` only matters for RBF. Larger `gamma` makes each sample influence a smaller region and can overfit; smaller `gamma` gives smoother boundaries.

```python
clf = SVC(**params)
clf.fit(X_train, y_train_enc)
```

Initializes and trains SVM on the training split.

```python
acc = accuracy_score(y_val_enc, clf.predict(X_val))
```

Predicts validation labels and computes validation accuracy.

```python
trial.report(acc, step=1)
if trial.should_prune():
    raise optuna.TrialPruned()
```

Reports the result to Optuna and allows early stopping of weak trials.

```python
trial.set_user_attr("model", clf)
return acc
```

Stores the trial model and returns the metric to maximize.

### SVM Cell 11: Optuna Progress

`plot_optuna_progress` builds a DataFrame of completed trials, computes `best_so_far = cummax(value)`, and plots per-trial validation accuracy against the best value so far.

### SVM Cell 12: Support Vector Summary

SVM decision boundaries are defined by support vectors, which are training points close to or violating the margin.

The function reads:

```python
model.n_support_
```

to count support vectors per class.

Actual saved-model support vectors:

| Model | Support vectors per class `[Cam, Chidan, Hieulenh, Nguyhiem, Phu]` | Total |
|---|---:|---:|
| SVM 6x3 | `[261, 401, 369, 169, 268]` | 1468 |
| SVM 8x2 | `[253, 387, 344, 145, 251]` | 1380 |

SVM 8x2 uses fewer support vectors and achieves higher accuracy, suggesting a more compact and better-generalizing representation.

### SVM Cell 13: Confusion Matrix

`plot_confusion_matrix_view` visualizes prediction mistakes:

- Rows are true labels.
- Columns are predicted labels.
- Diagonal cells are correct predictions.
- Off-diagonal cells are errors.

### SVM Cells 14-16: Single-Image Prediction and Visualization

`predict_one_image` repeats the exact training-time feature pipeline:

```python
image, gray, resized = load_image_steps(path)
feature = extract_combined_feature(image, resized, HOG_PARAMS, HSV_HIST_BINS, IMAGE_SIZE)
X_one = feature.reshape(1, -1)
pred_enc = model.predict(X_one)[0]
pred_label = label_encoder.inverse_transform([pred_enc])[0]
```

Important point: prediction must use the same resize, HOG parameters, HSV bins, and label encoder as training.

SVM probabilities are available because the final model uses:

```python
SVM_FINAL_PROBABILITY = True
```

### SVM Cell 17: Main Configuration

```python
SEED = 42
IMAGE_SIZE = (128, 128)
FEATURE_EXTRACTOR_NAME = "HOG_HSV"
HSV_HIST_BINS = (16, 8, 8)
VAL_SIZE = 0.2
N_JOBS = 8
OPTUNA_TRIALS = 30
OPTUNA_JOBS = 4
SVM_CACHE_SIZE_MB = 1000
SVM_FINAL_PROBABILITY = True
```

Meaning:

- `SEED=42`: reproducible splitting and Optuna sampling.
- `IMAGE_SIZE=(128,128)`: common input size for all images.
- `FEATURE_EXTRACTOR_NAME="HOG_HSV"`: metadata.
- `HSV_HIST_BINS=(16,8,8)`: 32 color features.
- `VAL_SIZE=0.2`: 20% of the original training set becomes validation.
- `N_JOBS=8`: parallel feature extraction.
- `OPTUNA_TRIALS=30`: 30 SVM hyperparameter trials.
- `OPTUNA_JOBS=4`: four trials run in parallel.
- `SVM_CACHE_SIZE_MB=1000`: SVM kernel cache size.
- `SVM_FINAL_PROBABILITY=True`: final SVM can output probabilities.

### SVM Cell 18: Load Data and Split

```python
all_train_paths, all_train_labels, train_classes = collect_paths(DATA_DIR / "train")
test_paths, test_labels, test_classes = collect_paths(DATA_DIR / "test")
```

Loads original training images and test images.

```python
all_classes = sorted(set(train_classes) | set(test_classes))
```

Uses the union of train/test class folders so the label encoder knows all classes.

```python
train_test_split(..., test_size=VAL_SIZE, random_state=SEED, stratify=all_train_labels)
```

Splits original training data into train and validation. `stratify` preserves class ratios.

### SVM Cells 19-21: EDA and Feature Visualization

Cell 19 plots dataset distribution. Cell 20 displays training samples by class. Cell 21 selects a `Nguyhiem` sample, plots the HOG+HSV pipeline, and prints feature counts.

### SVM Cell 22: Cache Tag, Feature Extraction, and Label Encoding

```python
cache_tag = f"hog_hsv_svm_{...}"
```

The cache name includes model type, HOG configuration, HSV bins, image size, validation ratio, and seed. This prevents `6x3` and `8x2` caches from being mixed.

```python
X_train, y_train = compute_features(...)
X_val, y_val = compute_features(...)
X_test, y_test = compute_features(...)
```

Builds feature matrices for all splits.

```python
le = LabelEncoder()
le.fit(all_classes)
y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)
```

Converts string labels to integer labels. With alphabetical class order, the mapping is:

```text
Cam -> 0
Chidan -> 1
Hieulenh -> 2
Nguyhiem -> 3
Phu -> 4
```

### SVM Cells 23-26: Create and Run Optuna Study

```python
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(...),
    sampler=TPESampler(seed=SEED),
)
```

The study maximizes validation accuracy. The pruner stops weak trials early. TPE proposes promising hyperparameters based on previous trials.

```python
study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=OPTUNA_JOBS)
```

Runs 30 trials with 4 parallel jobs.

### SVM Cell 27: Train Final Model

```python
X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train_enc, y_val_enc])
```

After hyperparameter selection, train and validation are combined to train the final model with more data.

```python
best_params = dict(study.best_params)
best_params.update({
    "probability": SVM_FINAL_PROBABILITY,
    "cache_size": SVM_CACHE_SIZE_MB,
    "random_state": SEED,
})
```

Uses Optuna's best parameters and enables probability output for the final saved model.

```python
best_clf = SVC(**best_params)
best_clf.fit(X_combined, y_combined)
```

Trains the final SVM.

Actual saved SVM parameters:

| Model | kernel | C | class_weight | shrinking | probability |
|---|---|---:|---|---|---|
| SVM 6x3 | linear | 7.356130968262376 | None | False | True |
| SVM 8x2 | linear | 0.03582517616218419 | balanced | False | True |

Interpretation:

- Optuna selected `linear` for both models, meaning HOG+HSV already creates a feature space where linear separation is strong enough.
- SVM 8x2 uses a very small `C` and `class_weight="balanced"`, favoring a wider margin and class balance.
- SVM 6x3 uses a larger `C`, likely because its feature space is larger and more detailed, but it generalizes worse on the test set.

### SVM Cells 28-30: Evaluation

```python
y_pred_test = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test_enc, y_pred_test)
test_cm = confusion_matrix(...)
```

Predicts the test set and computes accuracy and confusion matrix.

```python
classification_report(...)
```

Prints precision, recall, F1, and support.

Metric formulas:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

### SVM Cells 31-34: Error Analysis

`build_misclassification_df` creates a table of wrong predictions with:

- index,
- true label,
- predicted label,
- confidence,
- file name,
- path.

The code then groups errors by `true_label -> pred_label` to identify common confusion pairs.

`plot_all_misclassified_samples` displays wrong samples together with their HOG visualization. This helps explain whether mistakes come from similar shapes, blur, crop problems, or color ambiguity.

### SVM Cells 35-36: Prediction Examples

Cell 35 shows a grid of test predictions. Cell 36 shows a single-sample pipeline: original image -> grayscale -> resize -> HOG -> probability/prediction.

### SVM Cell 37: Save Model

```python
model_path = MODELS_DIR / "HOG_SVM_6x3.joblib"
```

or `HOG_SVM_8x2.joblib`, depending on the notebook.

```python
payload = {
    "model": best_clf,
    "algorithm": "SVC",
    "svm_params": best_params,
    "label_encoder": le,
    "feature_extractor": FEATURE_EXTRACTOR_NAME,
    "hog_params": HOG_PARAMS,
    "hsv_hist_bins": HSV_HIST_BINS,
    "image_size": IMAGE_SIZE,
    "classes": list(le.classes_),
}
```

The payload saves not only the classifier but also all metadata needed for correct inference:

- `label_encoder`: converts numeric predictions back to class names.
- `hog_params`: ensures inference uses the same HOG settings.
- `hsv_hist_bins`: ensures inference uses the same color histogram.
- `image_size`: ensures inference resizes images the same way.
- `classes`: stores class order.

```python
joblib.dump(payload, model_path)
```

Writes the payload to disk.

## 8. Random Forest-Specific Explanation

Random Forest uses the same preprocessing and feature extraction as SVM. The differences are the model import, objective function, feature importance plot, final model initialization, and saved file name.

### RF Cell 10: Optuna Objective

```python
params = {
    "n_estimators": 10,
    "max_depth": trial.suggest_int("max_depth", 5, 50),
    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    "random_state": SEED,
    "warm_start": True,
    "n_jobs": 1,
}
```

Parameter meanings:

- `n_estimators`: number of trees. The objective starts at 10 and increases through `ESTIMATOR_STEPS`.
- `max_depth`: maximum depth of each tree. Deeper trees fit more complex patterns but can overfit.
- `min_samples_split`: minimum samples required to split an internal node.
- `min_samples_leaf`: minimum samples required in a leaf node. Larger values smooth the trees.
- `max_features="sqrt"` or `"log2"`: each split considers only a subset of features, increasing tree diversity.
- `warm_start=True`: allows adding more trees without rebuilding the forest from scratch.
- `n_jobs=1`: each trial uses one job because Optuna already runs multiple trials in parallel.

```python
clf = RandomForestClassifier(**params)
acc = 0.0
for n_estimators in ESTIMATOR_STEPS:
    clf.set_params(n_estimators=n_estimators)
    clf.fit(X_train, y_train_enc)
    acc = accuracy_score(y_val_enc, clf.predict(X_val))
    trial.report(acc, n_estimators)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

Each trial evaluates the same tree hyperparameters while increasing the number of trees through `50, 100, 150, 200`. Weak trials can be pruned early.

```python
return acc
```

Returns the final validation accuracy, normally after 200 trees if not pruned.

### RF Cell 12: Feature Importance

Random Forest exposes:

```python
model.feature_importances_
```

This estimates which features reduce impurity most across the trees. `plot_feature_importance` takes the top 25 feature indices and plots them.

Note: the axis labels use `HOG[i]` for all indices. If an index falls into the final 32 HSV dimensions, the label would not be semantically exact, although HOG dominates the feature vector size.

### RF Cell 17: Main Configuration

```python
OPTUNA_TRIALS = 50
OPTUNA_JOBS = 4
RF_JOBS = 4
ESTIMATOR_STEPS = [50, 100, 150, 200]
```

RF uses 50 Optuna trials, more than SVM. The final model trains with 4 parallel jobs. `ESTIMATOR_STEPS` lets Optuna observe performance as the forest grows.

### RF Cell 24: Optuna Study

```python
sampler=TPESampler(multivariate=True, seed=SEED)
```

`multivariate=True` allows TPE to model interactions between hyperparameters such as `max_depth`, `min_samples_leaf`, and `min_samples_split`.

### RF Cell 27: Train Final Model

```python
best_params = dict(study.best_params)
best_params.update({
    "n_estimators": max(ESTIMATOR_STEPS),
    "random_state": SEED,
    "warm_start": False,
    "n_jobs": RF_JOBS,
})
```

The final RF uses 200 trees, disables `warm_start`, and enables 4 parallel jobs.

Actual saved RF parameters:

| Model | n_estimators | max_depth | min_samples_split | min_samples_leaf | max_features |
|---|---:|---:|---:|---:|---|
| RF 6x3 | 200 | 45 | 9 | 3 | sqrt |
| RF 8x2 | 200 | 21 | 4 | 2 | sqrt |

Interpretation:

- RF 6x3 uses deeper trees, likely because its feature space is larger and more complex.
- RF 8x2 uses shallower trees and still performs better, suggesting more stable features.
- Both use `max_features="sqrt"`, a common choice for classification because it increases diversity among trees.

### RF Cells 28-37

These cells mirror the SVM flow:

- predict on test,
- print classification report,
- plot feature importance,
- plot confusion matrix,
- build the misclassification table,
- visualize wrong predictions,
- show prediction examples,
- save `.joblib`.

The RF payload stores:

```python
payload = {
    "model": best_clf,
    "label_encoder": le,
    "feature_extractor": FEATURE_EXTRACTOR_NAME,
    "hog_params": HOG_PARAMS,
    "hsv_hist_bins": HSV_HIST_BINS,
    "image_size": IMAGE_SIZE,
    "classes": list(le.classes_),
}
```

## 9. Detailed Result Comparison

### SVM 6x3

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cam | 0.9559 | 0.9559 | 0.9559 | 68 |
| Chidan | 0.8333 | 0.8730 | 0.8527 | 63 |
| Hieulenh | 0.9000 | 0.8308 | 0.8640 | 65 |
| Nguyhiem | 0.9851 | 1.0000 | 0.9925 | 66 |
| Phu | 0.7857 | 0.8049 | 0.7952 | 41 |

Accuracy: `0.9010`.

### SVM 8x2

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cam | 0.9706 | 0.9706 | 0.9706 | 68 |
| Chidan | 0.8028 | 0.9048 | 0.8507 | 63 |
| Hieulenh | 0.9167 | 0.8462 | 0.8800 | 65 |
| Nguyhiem | 0.9851 | 1.0000 | 0.9925 | 66 |
| Phu | 0.8919 | 0.8049 | 0.8462 | 41 |

Accuracy: `0.9142`.

SVM 8x2 improves mostly on `Cam`, `Hieulenh`, and `Phu`, while `Nguyhiem` is already near perfect in both settings.

### RF 6x3

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cam | 0.9178 | 0.9853 | 0.9504 | 68 |
| Chidan | 0.7262 | 0.9683 | 0.8299 | 63 |
| Hieulenh | 0.9804 | 0.7692 | 0.8621 | 65 |
| Nguyhiem | 0.9394 | 0.9394 | 0.9394 | 66 |
| Phu | 1.0000 | 0.7073 | 0.8286 | 41 |

Accuracy: `0.8878`.

RF 6x3 has high recall for `Chidan` but low precision, meaning it predicts too many samples as `Chidan`.

### RF 8x2

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cam | 0.9315 | 1.0000 | 0.9645 | 68 |
| Chidan | 0.7500 | 1.0000 | 0.8571 | 63 |
| Hieulenh | 1.0000 | 0.7692 | 0.8696 | 65 |
| Nguyhiem | 0.9846 | 0.9697 | 0.9771 | 66 |
| Phu | 1.0000 | 0.7561 | 0.8611 | 41 |

Accuracy: `0.9109`.

RF 8x2 is strong on `Cam`, `Nguyhiem`, and `Phu` precision, but `Chidan` precision remains lower because some other classes are pulled into `Chidan`.

## 10. Why Does 8x2 Beat 6x3 Here?

More features do not always mean better performance.

| Configuration | HOG dim | HSV dim | Total dim | Best accuracy |
|---|---:|---:|---:|---:|
| 6x3 | 29241 | 32 | 29273 | 0.9010 with SVM |
| 8x2 | 8100 | 32 | 8132 | 0.9142 with SVM |

Possible reasons:

- `6x3` uses smaller cells and captures very local details, but those details may include noise, background, crop differences, and image artifacts.
- `6x3` has more than 3.5 times as many dimensions as `8x2`, which makes learning harder for a dataset of about 3,000 images.
- `8x2` gives a smoother descriptor that better captures the large-scale shape of traffic signs.
- Traditional models such as SVM and RF can be sensitive to very high-dimensional descriptors when the dataset is not large enough.

Suggested explanation for examiners:

> The 6x3 configuration has higher HOG resolution and can capture smaller edges. However, in this dataset, those small details may also include noise and crop variation. The 8x2 configuration is more compact and stable, so it generalizes better on the test set.

## 11. Why Combine HOG and HSV?

If we only use HOG:

- The model sees shape, borders, and arrows.
- It may confuse signs with similar shapes but different colors.

If we only use HSV:

- The model sees color distribution.
- It may confuse signs with similar colors but different shapes or symbols.

Combining them gives:

```text
feature = [shape_descriptor, color_descriptor]
```

so the classifier receives both shape and color information.

## 12. Why HOG on Grayscale but HSV on RGB?

HOG is based on intensity gradients, so grayscale is enough and reduces complexity. Its purpose is edge direction, not color.

HSV needs color information, so it starts from the RGB image and converts it to HSV.

## 13. Why Resize to 128x128?

Traditional ML models require fixed-length input vectors. If image sizes differ, HOG dimensions also differ. Resizing to `128x128` ensures every image produces the same number of features.

`128x128` is a practical compromise:

- Large enough to preserve traffic sign shape.
- Small enough for HOG, SVM, and RF to run efficiently.
- Compatible with both 8-pixel and 6-pixel HOG cell sizes.

## 14. Why Train/Validation/Test?

- Train: learn model parameters.
- Validation: choose hyperparameters with Optuna.
- Test: final evaluation on data not used for training or hyperparameter selection.

Using the test set for hyperparameter selection would make the final test result overly optimistic.

## 15. Why Use Optuna?

Hyperparameters strongly affect performance:

- SVM: `C`, `kernel`, `gamma`, `class_weight`.
- RF: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.

Manual search is slow. Optuna automates the search, learns from previous trials, and prunes weak trials early.

## 16. Possible Examiner Questions

**Question: Why not use CNN?**

Answer: CNNs can be stronger with larger datasets, but this project focuses on a classical Computer Vision pipeline: handcrafted HOG/HSV features followed by ML classifiers. This approach is explainable, lightweight, and does not require GPU training.

**Question: Does HOG capture color?**

No. HOG mainly captures gradient directions on grayscale images. That is why HSV histogram is added for color information.

**Question: Does HSV histogram lose spatial information?**

Yes. A global color histogram does not know where each color appears. However, because images are cropped around traffic signs, global color remains useful. HOG supplies local shape information.

**Question: Why does linear SVM work well?**

HOG+HSV transforms raw images into a structured feature space. In this space, classes can become close to linearly separable, and linear SVM is less prone to overfitting than RBF for high-dimensional features.

**Question: How does Random Forest probability work?**

Random Forest estimates class probability from tree votes. If 180 out of 200 trees vote for `Cam`, the probability is roughly 0.9.

**Question: Is SVM probability the same as RF probability?**

Not exactly. SVM naturally produces margins or decision scores. With `probability=True`, sklearn performs additional calibration to estimate probabilities.

**Question: Why did SVM 8x2 use `class_weight="balanced"`?**

Optuna selected it because it improved validation performance. The class `Phu` has fewer samples, so balanced weights can reduce bias toward larger classes.

**Question: What is the purpose of importing `exposure`?**

In the current notebooks it is imported but not used. It is likely leftover from an earlier experiment and does not affect training.

**Question: What is the cache for?**

HOG extraction for thousands of images can take time. The `.npz` cache stores `X` and `y`, so repeated runs can load features instead of recomputing them.

**Question: What happens if new images are added?**

If the sample count changes, the code detects that `len(y_cached) != len(labels)` and recomputes the cache.

**Question: What are the limitations of this pipeline?**

It depends on good cropping/localization. If the sign is small, blurry, heavily rotated, occluded, or in a complex scene, HOG+HSV may not be robust enough. A CNN or dedicated detector may perform better in difficult real-world settings.

## 17. Short Presentation Summary

The training pipeline converts each image into a HOG+HSV feature vector. HOG captures shape through gradient-orientation histograms on a `128x128` grayscale image. HSV histogram captures color using 32 bins. These features are concatenated and classified using SVM or Random Forest. Two HOG configurations are compared: `6x3` and `8x2`. `6x3` produces 29,273 total features and captures finer detail, but it is more sensitive to noise. `8x2` produces 8,132 total features, is more compact, and gives better test performance. The best current model is `HOG SVM 8x2`, with `0.9142` accuracy on 303 test images.
