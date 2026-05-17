# README huan luyen HOG + SVM va HOG + Random Forest

Tai lieu nay giai thich chi tiet cac notebook train trong thu muc `sign_classify_train/`:

- `sign_classify_train/HOG_SVM/train_hog_svm_6x3.ipynb`
- `sign_classify_train/HOG_SVM/train_hog_svm_8x2.ipynb`
- `sign_classify_train/HOG_RandomForest/train_hog_rf_6x3.ipynb`
- `sign_classify_train/HOG_RandomForest/train_hog_rf_8x2.ipynb`

Bon notebook co cung mot pipeline tong quat:

1. Doc anh tu `data/train` va `data/test`.
2. Chia `data/train` thanh train/validation theo ti le 80/20.
3. Tien xu ly anh: doc anh, bo kenh alpha neu co, tao anh grayscale, resize ve `128x128`.
4. Trich xuat dac trung HOG tren anh grayscale da resize.
5. Trich xuat histogram HSV tren anh mau da resize.
6. Noi dac trung HOG va HSV thanh mot vector dac trung duy nhat.
7. Dung Optuna tim hyperparameter tot tren validation set.
8. Train lai model tot nhat tren train + validation.
9. Danh gia tren test set va luu model `.joblib`.

## 1. Y tuong bai toan

Bai toan la phan loai bien bao giao thong vao 5 lop:

- `Cam`
- `Chidan`
- `Hieulenh`
- `Nguyhiem`
- `Phu`

Thay vi dua truc tiep pixel vao model, code bien moi anh thanh vector dac trung co y nghia:

- HOG mo ta hinh dang, canh, duong bien, huong gradient.
- HSV histogram mo ta phan bo mau sac, do bao hoa va do sang.

Sau do vector nay duoc dua vao SVM hoac Random Forest.

## 2. Du lieu va ket qua tong quat

Tap du lieu sau khi chia:

| Lop | Train | Validation | Test | Tong |
|---|---:|---:|---:|---:|
| Cam | 490 | 123 | 68 | 681 |
| Chidan | 454 | 113 | 63 | 630 |
| Hieulenh | 470 | 117 | 65 | 652 |
| Nguyhiem | 473 | 118 | 66 | 657 |
| Phu | 297 | 75 | 41 | 413 |
| Tong cong | 2184 | 546 | 303 | 3033 |

Ket qua tren test set:

| Model | Accuracy | Weighted F1 | Macro F1 | Test samples |
|---|---:|---:|---:|---:|
| HOG SVM 8x2 | 0.9142 | 0.9142 | 0.9080 | 303 |
| HOG RF 8x2 | 0.9109 | 0.9106 | 0.9059 | 303 |
| HOG SVM 6x3 | 0.9010 | 0.9009 | 0.8921 | 303 |
| HOG RF 6x3 | 0.8878 | 0.8875 | 0.8821 | 303 |

Nhan xet nhanh:

- `HOG SVM 8x2` la model tot nhat trong cac thi nghiem hien tai.
- `8x2` co so chieu thap hon `6x3` nhung tong quat hoa tot hon tren test set.
- `6x3` bat chi tiet min hon, nhung vector qua dai co the lam model nhay cam voi noise, anh lech, nen, chat luong anh va bien doi nho.
- SVM nhin chung phu hop voi HOG vi HOG da bien anh thanh vector so lien tuc kha giau thong tin ve bien dang.

## 3. 6x3 va 8x2 la gi?

Trong code, `6x3` va `8x2` la cach goi ngan gon cho hai cau hinh HOG:

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

Ten `6x3` co nghia:

- `6`: moi cell HOG co kich thuoc `6x6` pixel.
- `3`: moi block gom `3x3` cell.

Ten `8x2` co nghia:

- `8`: moi cell HOG co kich thuoc `8x8` pixel.
- `2`: moi block gom `2x2` cell.

Voi anh resize `128x128`, so chieu HOG duoc tinh nhu sau.

### Cau hinh 8x2

- So cell theo moi chieu: `floor(128 / 8) = 16`.
- So block theo moi chieu: `16 - 2 + 1 = 15`.
- Moi block co `2 x 2 x 9 = 36` gia tri.
- Tong HOG descriptor: `15 x 15 x 36 = 8100`.
- HSV descriptor: `16 + 8 + 8 = 32`.
- Tong feature: `8100 + 32 = 8132`.

### Cau hinh 6x3

- So cell theo moi chieu: `floor(128 / 6) = 21`.
- So block theo moi chieu: `21 - 3 + 1 = 19`.
- Moi block co `3 x 3 x 9 = 81` gia tri.
- Tong HOG descriptor: `19 x 19 x 81 = 29241`.
- HSV descriptor: `16 + 8 + 8 = 32`.
- Tong feature: `29241 + 32 = 29273`.

Y nghia:

- `6x3` tao vector dai hon nhieu, co do phan giai khong gian min hon.
- `8x2` gon hon, it chieu hon, tinh toan nhanh hon va co the giam overfitting.
- Trong ket qua hien tai, `8x2` tot hon o ca SVM va Random Forest, chung to dac trung qua chi tiet cua `6x3` khong giup them cho test set nay.

## 4. HOG hoat dong nhu the nao?

HOG la viet tat cua Histogram of Oriented Gradients. Muc tieu la mo ta hinh dang cua vat the bang phan bo huong gradient.

Quy trinh toan hoc co ban:

1. Tinh gradient theo truc x va y:

```text
Gx(x, y) = I(x + 1, y) - I(x - 1, y)
Gy(x, y) = I(x, y + 1) - I(x, y - 1)
```

2. Tinh do lon va huong gradient:

```text
magnitude = sqrt(Gx^2 + Gy^2)
angle = atan2(Gy, Gx)
```

3. Chia anh thanh cac cell nho, vi du `8x8` hoac `6x6` pixel.
4. Trong moi cell, gom cac pixel vao histogram huong. Code dung `orientations=9`, tuc la 9 bin huong.
5. Gom nhieu cell thanh block, vi du `2x2` hoac `3x3`, roi chuan hoa vector block bang `L2-Hys`.
6. Noi tat ca block descriptor thanh vector HOG cuoi cung.

Vi sao HOG phu hop voi bien bao:

- Bien bao co hinh dang ro: tron, tam giac, chu nhat, mui ten, vien do.
- HOG it phu thuoc vao mau hon pixel goc, vi no chu trong bien va gradient.
- Khi anh co thay doi do sang, HOG van giu duoc thong tin hinh dang tuong doi on dinh.

## 5. HSV histogram hoat dong nhu the nao?

HSV gom:

- `H` - Hue: mau sac, OpenCV bieu dien trong khoang `[0, 180]`.
- `S` - Saturation: do bao hoa, khoang `[0, 256]`.
- `V` - Value: do sang, khoang `[0, 256]`.

Code dung:

```python
HSV_HIST_BINS = (16, 8, 8)
```

Nghia la:

- Kenh H chia thanh 16 bin.
- Kenh S chia thanh 8 bin.
- Kenh V chia thanh 8 bin.
- Tong HSV feature = `16 + 8 + 8 = 32`.

Code khong tinh histogram 3D lien hop `H x S x V`, ma tinh rieng tung kenh roi noi lai:

```python
hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).flatten()
hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).flatten()
hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).flatten()
color_feature = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
color_feature /= color_feature.sum() + 1e-8
```

Vi sao dung HSV:

- Bien bao giao thong co mau dac trung: do, xanh, vang, trang, den.
- HSV tach mau sac khoi do sang nen de on dinh hon RGB khi dieu kien chieu sang thay doi.
- Histogram mau bo sung cho HOG: HOG nam hinh dang, HSV nam mau.

## 6. Giai thich line by line cac cell chung

Phan nay ap dung cho ca SVM va Random Forest, vi hai loai notebook dung chung tien xu ly, trich dac trung, chia du lieu, truc quan hoa va danh gia.

### Cell 1: import thu vien

```python
from __future__ import annotations
```

Cho phep dung type hint kieu moi va tri hoan viec danh gia annotation. Dong nay giup code tuong thich hon khi dung `Path`, `tuple[list[Path], ...]`.

```python
from pathlib import Path
```

Dung `Path` de thao tac duong dan theo kieu huong doi tuong, de doc hon string path.

```python
import cv2
```

OpenCV, duoc dung de resize anh RGB, chuyen RGB sang HSV, tinh histogram HSV, apply colormap va dilate anh HOG visualization.

```python
import joblib
```

Dung de luu model va metadata vao file `.joblib`.

```python
import matplotlib.pyplot as plt
```

Dung ve bieu do: phan bo du lieu, mau anh, HOG visualization, confusion matrix, prediction examples.

```python
import numpy as np
```

Thu vien tinh toan mang, dung cho vector feature, label encoded, concatenate, stack, arg sort, histogram array.

```python
import optuna
```

Thu vien toi uu hyperparameter tu dong.

```python
import pandas as pd
```

Dung tao bang thong ke: class distribution, feature overview, optuna trials, misclassification table.

```python
from joblib import Parallel, delayed
```

Dung chay trich xuat dac trung song song tren nhieu anh.

```python
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
```

`MedianPruner` cat som cac trial kem. `TPESampler` chon bo tham so tiep theo dua tren cac trial truoc do.

```python
from skimage import exposure
```

Thu vien xu ly anh. Trong notebook hien tai import nay khong duoc su dung truc tiep.

```python
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
```

`rgb2gray` chuyen anh RGB sang grayscale. `hog` tinh descriptor HOG. `imread` doc file anh. `resize` resize anh grayscale ve `128x128`.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
```

Metric danh gia, ham chia train/validation, va encoder chuyen label chu thanh so.

```python
from tqdm import tqdm
```

Hien thanh tien do khi trich feature.

Khac nhau giua hai loai notebook:

- Notebook SVM import `from sklearn.svm import SVC`.
- Notebook Random Forest import `from sklearn.ensemble import RandomForestClassifier`.

### Cell 2: tim data dir va thu thap path anh

```python
def resolve_data_dir() -> Path:
```

Khai bao ham tra ve duong dan `Path` toi thu muc `data`.

```python
here = Path.cwd().resolve()
```

Lay thu muc hien tai dang chay notebook, bien thanh duong dan tuyet doi.

```python
for parent in [here, *here.parents]:
```

Duyet tu thu muc hien tai len tat ca thu muc cha. Cach nay giup notebook van tim duoc `data` du chay tu root hay tu thu muc con.

```python
candidate = parent / "data"
```

Tao ung vien duong dan `data`.

```python
if (candidate / "train").is_dir() and (candidate / "test").is_dir():
    return candidate
```

Chi chap nhan neu co du ca `data/train` va `data/test`.

```python
raise FileNotFoundError("Could not find data/train and data/test")
```

Bao loi ro rang neu khong tim thay dataset.

```python
def collect_paths(split_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
```

Ham nhan vao thu muc split, vi du `data/train`, tra ve:

- danh sach path anh,
- mang label tuong ung,
- danh sach ten lop.

```python
labels = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
```

Lay ten cac folder con lam label, sap xep de thu tu lop on dinh.

```python
if not labels:
    raise RuntimeError(...)
```

Neu khong co folder lop thi dung som.

```python
paths: list[Path] = []
y: list[str] = []
```

Khoi tao danh sach anh va label.

```python
for label in labels:
    for path in sorted((split_dir / label).glob("*.*")):
        if path.is_file():
            paths.append(path)
            y.append(label)
```

Duyet tung lop, lay tat ca file co dau cham trong ten, sap xep de thu tu on dinh. Moi file anh duoc them vao `paths`, label folder duoc them vao `y`.

```python
if not paths:
    raise RuntimeError(...)
```

Neu co folder lop nhung khong co anh, bao loi.

```python
return paths, np.array(y), labels
```

Tra ve path list, label dang `numpy.ndarray`, va danh sach class.

### Cell 3: hien bang linh hoat

```python
def show_table(df: pd.DataFrame) -> None:
```

Ham hien thi DataFrame.

```python
try:
    display(df)
except NameError:
    print(df.to_string(index=False))
```

Neu chay trong Jupyter thi `display(df)` dep hon. Neu chay nhu script Python khong co `display`, in bang dang text.

### Cell 4: tien xu ly anh va trich dac trung tung anh

#### `load_image_steps`

```python
def load_image_steps(path: Path, image_size: tuple[int, int] | None = None):
```

Ham doc anh va tra ve 3 phien ban: anh goc, grayscale, grayscale da resize.

```python
if image_size is None:
    image_size = IMAGE_SIZE
```

Neu khong truyen size thi dung bien global `IMAGE_SIZE = (128, 128)`.

```python
image = imread(path)
```

Doc anh tu file. Ket qua co the la anh RGB, RGBA hoac grayscale.

```python
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[..., :3]
```

Neu anh co 4 kenh RGBA thi bo kenh alpha, giu RGB. HOG/HSV khong can alpha.

```python
if image.ndim == 3:
    gray = rgb2gray(image)
```

Anh mau duoc chuyen sang grayscale de tinh HOG.

```python
else:
    gray = image.astype(np.float32)
    if gray.max() > 1:
        gray = gray / 255.0
```

Neu anh da la grayscale, chuyen sang float. Neu pixel dang 0-255 thi normalize ve 0-1.

```python
resized = resize(gray, image_size, anti_aliasing=True)
```

Resize ve `128x128`. `anti_aliasing=True` giam rang cua khi thu nho anh.

```python
return image, gray, resized
```

Tra ve ca 3 buoc de vua train vua truc quan hoa duoc pipeline.

#### `to_uint8_rgb`

Ham nay dam bao anh co dang RGB `uint8` trong khoang 0-255 de OpenCV xu ly HSV.

```python
if image.ndim == 2:
    image = np.stack([image, image, image], axis=-1)
```

Neu anh grayscale thi nhan thanh 3 kenh giong nhau.

```python
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[..., :3]
```

Bo alpha neu co.

```python
rgb = image.astype(np.float32)
if rgb.max() <= 1.0:
    rgb = rgb * 255.0
return np.clip(rgb, 0, 255).astype(np.uint8)
```

Chuyen ve float, neu anh dang 0-1 thi nhan 255, cat ve [0, 255], roi ep kieu `uint8`.

#### `resize_rgb_image`

```python
rgb = to_uint8_rgb(image)
height, width = image_size
return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
```

OpenCV nhan size theo thu tu `(width, height)`, trong khi `IMAGE_SIZE` luu `(height, width)`. `INTER_AREA` phu hop khi resize anh, dac biet khi thu nho.

#### `enhance_hog_image`

Ham nay chi de lam anh HOG visualization dep va de nhin hon, khong anh huong vector HOG dung de train.

```python
image = np.asarray(hog_image, dtype=np.float32)
image = np.nan_to_num(...)
image = np.maximum(image, 0.0)
```

Chuyen anh HOG sang float, thay NaN/inf bang 0, bo gia tri am.

```python
if float(image.max()) <= float(image.min()):
    return np.full((*image.shape, 3), 246, dtype=np.uint8)
```

Neu anh gan nhu phang, tra ve nen xam nhat.

```python
active = image[image > 0]
```

Lay cac pixel HOG co tin hieu.

```python
low = np.percentile(active, 5.0)
high = np.percentile(active, 99.5)
```

Dung percentile de giam anh huong cua outlier.

```python
image = np.clip((image - low) / (high - low + 1e-8), 0.0, 1.0)
image = np.power(image, 0.32)
```

Normalize ve 0-1 va gamma correction. Mu `0.32` lam cac net yeu sang hon de de quan sat.

```python
image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
image_u8 = cv2.dilate(image_u8, np.ones((2, 2), dtype=np.uint8), iterations=1)
```

Doi ve anh 8-bit va lam day net bang dilation.

```python
color_bgr = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
```

To mau HOG visualization. OpenCV tra BGR nen can doi sang RGB de matplotlib hien dung.

```python
canvas = np.full((*image_u8.shape, 3), 246, dtype=np.uint8)
mask = image_u8 > 10
alpha = (image_u8.astype(np.float32) / 255.0)[..., None]
blended = canvas * (1 - alpha) + color_rgb * alpha
canvas[mask] = blended[mask].astype(np.uint8)
return canvas
```

Tron mau HOG len nen xam nhat, chi giu vung co tin hieu lon hon nguong 10.

#### `extract_hog_visual`

```python
descriptors, hog_image = hog(gray_resized, visualize=True, **hog_params)
```

Tinh ca vector HOG va anh truc quan hoa HOG. `**hog_params` truyen cac tham so nhu `orientations`, `pixels_per_cell`, `cells_per_block`.

```python
hog_image = enhance_hog_image(hog_image)
return descriptors, hog_image
```

Lam anh HOG de nhin hon, tra ve descriptor va visualization.

#### `extract_hsv_histogram`

```python
rgb_resized = resize_rgb_image(image, image_size)
hsv = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2HSV)
```

Resize anh mau va chuyen sang HSV.

```python
h_bins, s_bins, v_bins = hsv_bins
```

Tach so bin tung kenh.

```python
hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).flatten()
hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).flatten()
hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).flatten()
```

Tinh histogram rieng cho H, S, V. `flatten()` bien cot vector thanh vector 1 chieu.

```python
color_feature = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
color_feature /= color_feature.sum() + 1e-8
return color_feature
```

Noi 3 histogram thanh vector 32 chieu va normalize tong ve 1. `1e-8` tranh chia cho 0.

#### `extract_combined_feature`

```python
hog_feature = hog(gray_resized, **hog_params).astype(np.float32)
color_feature = extract_hsv_histogram(image, image_size, hsv_bins)
return np.concatenate([hog_feature, color_feature])
```

Tinh HOG that su dung cho train, tinh HSV, roi noi lai. Day la feature dau vao cua model.

Vector co dang:

```text
X_one = [HOG_0, HOG_1, ..., HOG_n, HSV_0, HSV_1, ..., HSV_31]
```

#### `split_combined_feature`

```python
return feature[:hog_dim], feature[hog_dim:hog_dim + color_dim]
```

Ham tach vector tong thanh phan HOG va HSV. Trong notebook hien tai ham nay la helper, khong phai buoc train chinh.

#### `choose_representative_sample`

Ham chon mot anh mau de visualize pipeline. Neu co `preferred_label`, vi du `Nguyhiem`, thi lay anh dau tien thuoc lop do; neu khong, lay anh dau tien trong danh sach.

### Cell 5: tinh feature cho ca tap anh

```python
def compute_features(...):
```

Ham nhan danh sach path va label, tra ve ma tran feature `X` va vector label `y`.

```python
if cache_file is not None and cache_file.exists():
```

Neu da co cache `.npz`, thu doc lai de tiet kiem thoi gian.

```python
with np.load(cache_file) as data:
    X_cached = data["X"]
    y_cached = data["y"]
```

Cache luu hai mang: feature va label.

```python
if len(y_cached) == len(labels):
    return X_cached, y_cached
```

Neu so sample khop thi dung cache.

```python
print(...)
```

Neu cache cu khong khop so anh hien tai, bao recompute.

```python
def extract(path: Path) -> np.ndarray | None:
```

Ham con trich feature cho mot anh.

```python
try:
    image, _, gray_resized = load_image_steps(path, image_size)
    return extract_combined_feature(...)
except Exception:
    return None
```

Neu doc/trich feature loi, tra `None` de bo qua file loi thay vi lam dung toan bo qua trinh.

```python
features = Parallel(n_jobs=n_jobs)(
    delayed(extract)(p) for p in tqdm(paths, desc=desc, total=len(paths))
)
```

Chay trich feature song song voi `n_jobs=8`, co thanh tien do.

```python
kept_feats = []
kept_labels = []
skipped = 0
```

Khoi tao danh sach feature hop le, label hop le, va dem file bi skip.

```python
for feat, label in zip(features, labels):
    if feat is None:
        skipped += 1
        continue
    kept_feats.append(feat)
    kept_labels.append(label)
```

Loai file loi, giu file thanh cong.

```python
if not kept_feats:
    raise RuntimeError(...)
```

Neu khong trich duoc feature nao thi dung.

```python
X = np.vstack(kept_feats)
y = np.array(kept_labels)
```

Ghep cac vector feature thanh ma tran 2D:

```text
X.shape = (so_anh, so_chieu_feature)
```

```python
np.savez_compressed(cache_file, X=X, y=y)
```

Luu cache nen vao `.npz`.

```python
return X, y
```

Tra ve feature va label da can chinh.

### Cell 6: thong ke phan bo dataset

Ham `plot_dataset_distribution` tao bang dem so anh theo lop va split. No dung `value_counts()`, `pivot_table()`, sau do ve bar chart. Muc dich la kiem tra dataset co bi lech lop qua nang hay khong.

Diem quan trong:

- Neu class imbalance lon, accuracy co the gay hieu nham.
- Vi vay report dung them macro F1 va weighted F1.

### Cell 7: hien anh mau theo lop

Ham `plot_class_samples` ve toi da `max_per_class=4` anh cho moi lop. No giup kiem tra truc quan:

- Anh co dung label khong.
- Co anh hu, mo, crop sai khong.
- Lop nao co do bien thien lon.

### Cell 8: visualize pipeline HOG + HSV cho mot anh

Ham `plot_preprocessing_and_hog` hien 5 cot:

1. Anh goc.
2. Anh grayscale.
3. Anh resize `128x128`.
4. HOG visualization.
5. HSV histogram.

Dong quan trong:

```python
hog_descriptors, hog_image = extract_hog_visual(resized, HOG_PARAMS)
color_descriptors = extract_hsv_histogram(image, IMAGE_SIZE, HSV_HIST_BINS)
combined = np.concatenate([hog_descriptors.astype(np.float32), color_descriptors])
```

Day la minh hoa dung cach feature duoc tao: HOG + HSV.

### Cell 9: in tong quan so chieu feature

```python
color_dim = sum(HSV_HIST_BINS)
hog_dim = X_train.shape[1] - color_dim
```

HSV luon co 32 chieu. HOG dim bang tong feature tru 32.

Bang hien:

- So sample train/validation/test.
- So descriptor HOG.
- So descriptor HSV.
- Tong feature.

## 7. Giai thich rieng notebook SVM

### Cell 10 SVM: ham objective cho Optuna

```python
def objective(trial: optuna.Trial) -> float:
```

Moi trial la mot lan thu bo hyperparameter.

```python
kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
```

Thu hai kernel:

- `linear`: bien phan tach la sieu phang trong khong gian feature.
- `rbf`: kernel phi tuyen, co the tao bien phan tach cong hon.

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

Y nghia:

- `C`: do manh cua phat sai. `C` lon co gang fit train chinh xac hon nhung de overfit; `C` nho chap nhan sai de margin rong hon.
- Tim theo log scale tu `0.01` den `100` vi `C` anh huong theo bac do lon.
- `class_weight=None`: moi class co trong so nhu nhau.
- `class_weight="balanced"`: class it mau duoc tang trong so theo tan suat.
- `shrinking`: bat/tat heuristic tang toc giai bai toan SVM.
- `probability=False`: trong luc search tat xac suat de train nhanh hon.
- `cache_size=1000`: cap 1000MB cache kernel.
- `random_state=42`: co dinh tinh tai lap.

```python
if kernel == "rbf":
    params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
```

`gamma` chi dung cho RBF. `gamma` lon lam vung anh huong cua moi sample hep, de overfit; `gamma` nho lam bien quyet dinh muot hon.

```python
clf = SVC(**params)
clf.fit(X_train, y_train_enc)
```

Khoi tao va train SVM tren train split.

```python
acc = accuracy_score(y_val_enc, clf.predict(X_val))
```

Du doan validation va tinh accuracy.

```python
trial.report(acc, step=1)
if trial.should_prune():
    raise optuna.TrialPruned()
```

Bao ket qua cho Optuna. Neu trial kem so voi median cac trial truoc, co the cat som.

```python
trial.set_user_attr("model", clf)
return acc
```

Luu model vao trial metadata va tra ve validation accuracy de Optuna maximize.

### Cell 11 SVM: plot tien trinh Optuna

Ham `plot_optuna_progress` lay cac trial co `trial.value`, tao DataFrame, tinh `best_so_far = cummax(value)`, ve duong accuracy tung trial va best-so-far. No giup giai thich qua trinh search co hoi tu hay khong.

### Cell 12 SVM: support vector summary

SVM quyet dinh bien phan tach bang cac support vector, tuc cac diem gan margin hoac vi pham margin.

Ham nay doc:

```python
model.n_support_
```

de biet moi lop co bao nhieu support vector.

Neu ty le support vector cao, model can nhieu mau train de quyet dinh bien phan tach. Dieu nay co the noi len bai toan kho, du lieu chong lap, hoac feature chua tach lop that ro.

Support vector thuc te:

| Model | Support vectors tung lop `[Cam, Chidan, Hieulenh, Nguyhiem, Phu]` | Tong |
|---|---:|---:|
| SVM 6x3 | `[261, 401, 369, 169, 268]` | 1468 |
| SVM 8x2 | `[253, 387, 344, 145, 251]` | 1380 |

SVM 8x2 can it support vector hon va accuracy cao hon, cho thay representation gon hon nhung tach lop tot hon.

### Cell 13 SVM: confusion matrix

`plot_confusion_matrix_view` ve ma tran nham lan:

- Hang la label that.
- Cot la label du doan.
- Duong cheo chinh la du doan dung.
- O ngoai duong cheo la cac nham lan.

### Cell 14-16 SVM: du doan mot anh va visualize prediction

`predict_one_image` lap lai dung pipeline train:

```python
image, gray, resized = load_image_steps(path)
feature = extract_combined_feature(image, resized, HOG_PARAMS, HSV_HIST_BINS, IMAGE_SIZE)
X_one = feature.reshape(1, -1)
pred_enc = model.predict(X_one)[0]
pred_label = label_encoder.inverse_transform([pred_enc])[0]
```

Luu y quan trong: khi predict, anh cung phai resize, HOG, HSV giong train. Neu pipeline predict khac pipeline train, ket qua se sai.

`predict_proba` chi co vi final model dat:

```python
SVM_FINAL_PROBABILITY = True
```

Trong SVM, probability duoc hieu chuan them, thuong ton thoi gian hon `predict` thong thuong.

### Cell 17 SVM: cau hinh chinh

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

Y nghia:

- `SEED=42`: giu chia du lieu va Optuna co tinh tai lap.
- `IMAGE_SIZE=(128,128)`: kich thuoc chung cho moi anh, can bang giua thong tin va toc do.
- `FEATURE_EXTRACTOR_NAME="HOG_HSV"`: metadata de biet model dung feature nao.
- `HSV_HIST_BINS=(16,8,8)`: 32 chieu mau.
- `VAL_SIZE=0.2`: 20% train goc lam validation.
- `N_JOBS=8`: song song hoa feature extraction.
- `OPTUNA_TRIALS=30`: thu 30 bo tham so SVM.
- `OPTUNA_JOBS=4`: chay 4 trial song song.
- `SVM_CACHE_SIZE_MB=1000`: cache kernel SVM.
- `SVM_FINAL_PROBABILITY=True`: model cuoi co the tra xac suat.

### Cell 18 SVM: load du lieu va chia train/validation

```python
all_train_paths, all_train_labels, train_classes = collect_paths(DATA_DIR / "train")
test_paths, test_labels, test_classes = collect_paths(DATA_DIR / "test")
```

Doc danh sach anh train goc va test.

```python
all_classes = sorted(set(train_classes) | set(test_classes))
```

Dung hop class train va test de label encoder biet du cac lop.

```python
train_test_split(..., test_size=VAL_SIZE, random_state=SEED, stratify=all_train_labels)
```

Chia train/validation. `stratify` giu ty le lop gan giong ban dau, rat quan trong khi dataset khong hoan toan can bang.

### Cell 19-21 SVM: EDA va visualize feature

Cell 19 ve phan bo du lieu. Cell 20 hien anh mau. Cell 21 chon mot anh `Nguyhiem`, ve pipeline HOG + HSV va in so feature.

### Cell 22 SVM: cache tag, feature extraction va label encoding

```python
cache_tag = f"hog_hsv_svm_{...}"
```

Ten cache chua:

- model type `svm`,
- cau hinh HOG,
- bins HSV,
- image size,
- validation ratio,
- seed.

Nho vay cache cua `6x3` va `8x2` khong bi dung chung sai.

```python
X_train, y_train = compute_features(...)
X_val, y_val = compute_features(...)
X_test, y_test = compute_features(...)
```

Tao ma tran feature cho 3 split.

```python
le = LabelEncoder()
le.fit(all_classes)
y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)
```

SVM can label so. `LabelEncoder` anh xa label chu sang so, vi du theo thu tu alphabet:

```text
Cam -> 0
Chidan -> 1
Hieulenh -> 2
Nguyhiem -> 3
Phu -> 4
```

### Cell 23-26 SVM: tao study va chay Optuna

```python
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(...),
    sampler=TPESampler(seed=SEED),
)
```

`direction="maximize"` vi can toi da hoa validation accuracy. `MedianPruner` cat trial kem. `TPESampler` la Bayesian optimization sampler dua tren Tree-structured Parzen Estimator.

```python
study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=OPTUNA_JOBS)
```

Chay 30 trial, 4 trial song song.

### Cell 27 SVM: train final model

```python
X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train_enc, y_val_enc])
```

Sau khi da chon tham so bang validation, gop train + validation de train model cuoi cung. Cach nay tan dung nhieu du lieu hon.

```python
best_params = dict(study.best_params)
best_params.update({
    "probability": SVM_FINAL_PROBABILITY,
    "cache_size": SVM_CACHE_SIZE_MB,
    "random_state": SEED,
})
```

Lay tham so tot nhat, bat probability cho final model.

```python
best_clf = SVC(**best_params)
best_clf.fit(X_combined, y_combined)
```

Train final SVM.

Tham so thuc te trong model da luu:

| Model | kernel | C | class_weight | shrinking | probability |
|---|---|---:|---|---|---|
| SVM 6x3 | linear | 7.356130968262376 | None | False | True |
| SVM 8x2 | linear | 0.03582517616218419 | balanced | False | True |

Nhan xet:

- Optuna chon `linear` cho ca hai, nghia la HOG+HSV da tao khong gian feature ma bien phan tach tuyen tinh du tot.
- `SVM 8x2` chon `C` rat nho va `class_weight="balanced"`, uu tien margin rong va can bang lop.
- `SVM 6x3` chon `C` lon hon, co the vi feature nhieu chieu hon can fit manh hon, nhung ket qua test lai kem hon.

### Cell 28-30 SVM: danh gia

```python
y_pred_test = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test_enc, y_pred_test)
test_cm = confusion_matrix(...)
```

Du doan test, tinh accuracy va confusion matrix.

```python
classification_report(...)
```

In precision, recall, F1, support.

Cong thuc:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

### Cell 31-34 SVM: phan tich loi

`build_misclassification_df` tao bang cac anh du doan sai voi:

- index,
- true_label,
- pred_label,
- confidence,
- file_name,
- path.

Sau do group theo cap `true_label -> pred_label` de xem model hay nham lop nao voi lop nao.

`plot_all_misclassified_samples` hien anh sai kem HOG visualization de phan tich vi sao sai: hinh dang giong nhau, anh mo, crop khong tot, mau sac gay nham lan.

### Cell 35-36 SVM: minh hoa prediction

Cell 35 hien grid du doan tren test set. Cell 36 hien pipeline cho mot anh: original -> grayscale -> resize -> HOG -> probability/prediction.

### Cell 37 SVM: luu model

```python
model_path = MODELS_DIR / "HOG_SVM_6x3.joblib"
```

Hoac `HOG_SVM_8x2.joblib` tuy notebook.

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

Khong chi luu model, ma luu ca metadata de predict dung:

- `label_encoder`: de doi so ve ten lop.
- `hog_params`: de trich HOG y het luc train.
- `hsv_hist_bins`: de trich HSV y het luc train.
- `image_size`: de resize dung.
- `classes`: de biet thu tu lop.

```python
joblib.dump(payload, model_path)
```

Ghi payload ra file.

## 8. Giai thich rieng notebook Random Forest

Random Forest dung cung tien xu ly va feature extraction voi SVM. Phan khac nam o import, objective, feature importance, final model va ten file luu.

### Cell 10 RF: ham objective cho Optuna

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

Y nghia tham so:

- `n_estimators`: so cay trong rung. Trong objective khoi dau 10 roi tang dan theo `ESTIMATOR_STEPS`.
- `max_depth`: do sau toi da cua moi cay. Sau lon fit phuc tap hon nhung de overfit.
- `min_samples_split`: so mau toi thieu de mot node duoc tach tiep.
- `min_samples_leaf`: so mau toi thieu o mot la. Gia tri lon lam cay muot hon, giam overfit.
- `max_features="sqrt"` hoac `"log2"`: moi lan split chi xem mot tap con feature, giup cac cay khac nhau va giam tuong quan.
- `warm_start=True`: cho phep tang so cay ma khong train lai tu dau.
- `n_jobs=1`: moi trial chi dung 1 job, vi Optuna da chay nhieu trial song song bang `OPTUNA_JOBS=4`.

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

Moi trial thu cung mot bo tham so cay, nhung tang so cay qua `50, 100, 150, 200`. Neu validation accuracy kem, Optuna co the prune som.

```python
return acc
```

Tra ve accuracy o buoc cuoi, tuc khi so cay da dat 200 neu khong bi prune.

### Cell 12 RF: feature importance

Random Forest co thuoc tinh:

```python
model.feature_importances_
```

No uoc luong feature nao giup giam impurity nhieu trong cac cay. Ham `plot_feature_importance` lay top 25 feature co importance cao nhat va ve bar chart.

Luu y: code dat label truc y la `HOG[i]` cho moi index. Neu index nam trong 32 chieu HSV cuoi thi label nay chua that chinh xac ve ten, nhung vi HOG chiem gan nhu toan bo feature nen top feature thuong la HOG.

### Cell 17 RF: cau hinh chinh

```python
OPTUNA_TRIALS = 50
OPTUNA_JOBS = 4
RF_JOBS = 4
ESTIMATOR_STEPS = [50, 100, 150, 200]
```

RF dung 50 trial, nhieu hon SVM. Final model dung `RF_JOBS=4` de train song song cac cay. `ESTIMATOR_STEPS` cho phep Optuna danh gia rung khi so cay tang dan.

### Cell 24 RF: Optuna study

```python
sampler=TPESampler(multivariate=True, seed=SEED)
```

`multivariate=True` cho phep TPE xet quan he giua cac hyperparameter, vi trong RF cac tham so nhu `max_depth`, `min_samples_leaf`, `min_samples_split` co lien he voi nhau.

### Cell 27 RF: train final model

```python
best_params = dict(study.best_params)
best_params.update({
    "n_estimators": max(ESTIMATOR_STEPS),
    "random_state": SEED,
    "warm_start": False,
    "n_jobs": RF_JOBS,
})
```

Final RF dung 200 cay, tat `warm_start` vi khong can tang cay nua, bat `n_jobs=4` de train nhanh.

Tham so thuc te trong model da luu:

| Model | n_estimators | max_depth | min_samples_split | min_samples_leaf | max_features |
|---|---:|---:|---:|---:|---|
| RF 6x3 | 200 | 45 | 9 | 3 | sqrt |
| RF 8x2 | 200 | 21 | 4 | 2 | sqrt |

Nhan xet:

- RF 6x3 can `max_depth=45`, sau hon RF 8x2, co the vi feature 6x3 nhieu chieu va phuc tap hon.
- RF 8x2 dung cay nong hon (`max_depth=21`) nhung accuracy cao hon, cho thay feature gon hon de hoc on dinh hon.
- Ca hai dung `max_features="sqrt"`, day la default pho bien cho classification vi moi split chi xet can bac hai so feature, lam cac cay da dang hon.

### Cell 28-37 RF

Tu Cell 28 tro di tuong tu SVM:

- predict test,
- in report,
- ve feature importance,
- ve confusion matrix,
- lap bang misclassification,
- hien anh sai,
- hien prediction examples,
- luu `.joblib`.

Payload RF luu:

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

## 9. So sanh ket qua chi tiet

### SVM 6x3

| Lop | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cam | 0.9559 | 0.9559 | 0.9559 | 68 |
| Chidan | 0.8333 | 0.8730 | 0.8527 | 63 |
| Hieulenh | 0.9000 | 0.8308 | 0.8640 | 65 |
| Nguyhiem | 0.9851 | 1.0000 | 0.9925 | 66 |
| Phu | 0.7857 | 0.8049 | 0.7952 | 41 |

Accuracy: `0.9010`.

### SVM 8x2

| Lop | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cam | 0.9706 | 0.9706 | 0.9706 | 68 |
| Chidan | 0.8028 | 0.9048 | 0.8507 | 63 |
| Hieulenh | 0.9167 | 0.8462 | 0.8800 | 65 |
| Nguyhiem | 0.9851 | 1.0000 | 0.9925 | 66 |
| Phu | 0.8919 | 0.8049 | 0.8462 | 41 |

Accuracy: `0.9142`.

SVM 8x2 tot hon SVM 6x3 chu yeu o `Cam`, `Hieulenh`, `Phu`, trong khi `Nguyhiem` gan nhu dat tran.

### RF 6x3

| Lop | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cam | 0.9178 | 0.9853 | 0.9504 | 68 |
| Chidan | 0.7262 | 0.9683 | 0.8299 | 63 |
| Hieulenh | 0.9804 | 0.7692 | 0.8621 | 65 |
| Nguyhiem | 0.9394 | 0.9394 | 0.9394 | 66 |
| Phu | 1.0000 | 0.7073 | 0.8286 | 41 |

Accuracy: `0.8878`.

RF 6x3 co recall cao cho `Chidan` nhung precision thap, nghia la model du doan nhieu mau thanh `Chidan`, dan den false positive.

### RF 8x2

| Lop | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Cam | 0.9315 | 1.0000 | 0.9645 | 68 |
| Chidan | 0.7500 | 1.0000 | 0.8571 | 63 |
| Hieulenh | 1.0000 | 0.7692 | 0.8696 | 65 |
| Nguyhiem | 0.9846 | 0.9697 | 0.9771 | 66 |
| Phu | 1.0000 | 0.7561 | 0.8611 | 41 |

Accuracy: `0.9109`.

RF 8x2 rat manh o `Cam`, `Nguyhiem`, `Phu` precision cao, nhung `Chidan` precision con thap do co mau lop khac bi keo ve `Chidan`.

## 10. Vi sao 8x2 tot hon 6x3 trong ket qua nay?

Mac du `6x3` co nhieu feature hon, nhieu hon khong dong nghia tot hon.

So sanh:

| Cau hinh | HOG dim | HSV dim | Total dim | Accuracy tot nhat |
|---|---:|---:|---:|---:|
| 6x3 | 29241 | 32 | 29273 | 0.9010 voi SVM |
| 8x2 | 8100 | 32 | 8132 | 0.9142 voi SVM |

Ly do co the:

- `6x3` chia cell nho hon nen bat chi tiet cuc bo, nhung anh dataset co the co noise, crop lech, nen khac nhau.
- `6x3` co so chieu cao gap hon 3.5 lan `8x2`, lam bai toan hoc kho hon voi dataset 3033 anh.
- `8x2` lam dac trung muot hon, giu cau truc bien bao cap vung, phu hop voi hinh dang lon cua bien bao.
- Model truyen thong nhu SVM/RF thuong nhay voi high-dimensional sparse feature neu du lieu khong du lon.

Cach noi voi giam khao:

> Cau hinh 6x3 co do phan giai HOG cao hon, nen ve ly thuyet bat duoc canh nho hon. Tuy nhien trong dataset nay, chi tiet nho co the bao gom noise, nen, va sai khac do crop. Cau hinh 8x2 tao descriptor gon hon, on dinh hon va tong quat hoa tot hon, nen accuracy test cao hon.

## 11. Vi sao ket hop HOG va HSV?

Neu chi dung HOG:

- Model thay ro hinh dang, vien, mui ten.
- Nhung co the nham cac bien co hinh dang gan nhau nhung mau khac.

Neu chi dung HSV:

- Model thay ro mau chu dao.
- Nhung co the nham cac bien cung mau nhung hinh dang/noi dung khac.

Ket hop:

```text
feature = [shape_descriptor, color_descriptor]
```

giup model co ca thong tin hinh dang va mau sac.

## 12. Vi sao HOG dung grayscale con HSV dung RGB?

HOG dua tren gradient cuong do, nen grayscale la du va giam nhieu. Muc tieu cua HOG la canh va huong, khong phai mau.

HSV can mau, nen phai dung anh RGB. Sau do chuyen RGB sang HSV de bieu dien mau on dinh hon.

## 13. Vi sao resize ve 128x128?

Model ML truyen thong can vector dau vao co cung kich thuoc. Neu anh co kich thuoc khac nhau, HOG dim se khac nhau. Resize ve `128x128` dam bao moi anh tao ra vector cung so chieu.

`128x128` la thoa hiep:

- Du lon de giu hinh dang bien bao.
- Khong qua lon de HOG/RF/SVM chay cham.
- Chia hop ly voi cell size 8 va 6.

## 14. Vi sao dung train/validation/test?

- Train: hoc tham so model.
- Validation: chon hyperparameter bang Optuna.
- Test: danh gia cuoi cung tren du lieu chua dung trong train va chon tham so.

Neu dung test de chon hyperparameter, ket qua test se bi lac quan gia.

## 15. Vi sao dung Optuna?

Hyperparameter anh huong lon:

- SVM: `C`, `kernel`, `gamma`, `class_weight`.
- RF: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.

Thu cong tung bo tham so rat ton thoi gian. Optuna tu dong thu, hoc tu ket qua truoc, va cat cac trial kem bang pruner.

## 16. Cau hoi giam khao co the hoi

**Hoi: Tai sao khong dung CNN?**

Tra loi: CNN co the manh hon khi du lieu lon, nhung do an nay tap trung vao pipeline Computer Vision truyen thong: trich dac trung thu cong HOG/HSV roi dung ML classifier. Cach nay de giai thich, it can GPU, va phu hop khi can minh bach feature.

**Hoi: HOG co bat duoc mau khong?**

Khong. HOG chu yeu bat huong gradient tren grayscale. Vi vay code bo sung HSV histogram de nam thong tin mau.

**Hoi: HSV histogram co mat thong tin vi tri khong?**

Co. Histogram mau toan anh khong biet mau nam o dau. Nhung trong bai toan nay anh da crop quanh bien bao, nen mau tong the van co ich. HOG giu them thong tin hinh dang/vi tri cuc bo.

**Hoi: Vi sao SVM linear tot?**

HOG+HSV da chuyen anh sang khong gian feature co cau truc tot. Trong khong gian nay, cac lop co the gan tuyen tinh hon so voi pixel goc. Linear SVM cung it overfit hon RBF khi feature dim cao.

**Hoi: Vi sao RF co predict_proba?**

Random Forest tinh xac suat bang ti le cay bo phieu cho moi lop. Neu 180/200 cay chon `Cam`, xac suat lop `Cam` xap xi 0.9.

**Hoi: Probability cua SVM co giong RF khong?**

Khong hoan toan. SVM goc cho decision score/margin. Khi `probability=True`, sklearn hieu chuan them de ra xac suat, ton thoi gian hon va xac suat chi mang tinh uoc luong.

**Hoi: Tai sao dung `class_weight="balanced"` o SVM 8x2?**

Optuna chon vi validation accuracy tot hon. Dataset co lop `Phu` it mau hon, nen balanced giup model khong thien qua cac lop nhieu mau.

**Hoi: Import `exposure` de lam gi?**

Trong notebook hien tai `exposure` duoc import nhung khong dung. Co the la phan con lai tu thu nghiem truoc. Viec nay khong anh huong train.

**Hoi: Cache `.npz` co tac dung gi?**

Trich HOG cho hang nghin anh ton thoi gian. Cache luu `X` va `y`, lan sau neu so sample khop thi load lai, khong can tinh lai feature.

**Hoi: Neu them anh moi thi sao?**

Neu so sample thay doi, code phat hien `len(y_cached) != len(labels)` va recompute cache.

**Hoi: Diem yeu cua pipeline nay?**

Can crop/localization tot. Neu anh co nen phuc tap, bien bao qua nho, mo, nghieng manh, che khuat, HOG/HSV co the khong du manh. CNN hoac detector chuyen dung co the tot hon trong moi truong thuc te phuc tap.

## 17. Tom tat de thuyet trinh ngan

Pipeline train bien moi anh thanh vector HOG+HSV. HOG lay thong tin hinh dang bang histogram huong gradient tren anh grayscale `128x128`; HSV histogram lay thong tin mau voi 32 bin. Hai feature duoc noi lai va dua vao SVM hoac Random Forest. Hai cau hinh HOG duoc so sanh la `6x3` va `8x2`. `6x3` tao 29273 feature, chi tiet hon nhung de nhieu noise hon; `8x2` tao 8132 feature, gon hon va trong ket qua thuc nghiem cho accuracy cao hon. Model tot nhat la `HOG SVM 8x2` voi accuracy `0.9142` tren 303 anh test.
