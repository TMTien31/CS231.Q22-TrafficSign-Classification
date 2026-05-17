# De cuong hoi dap phan training HOG + SVM va HOG + Random Forest

File nay dung de on van dap voi giam khao. Moi muc gom:

- Cau hoi co the bi hoi.
- Cau tra loi ngan gon nhung du y.
- Can cu truc tiep tu source code train trong `sign_classify_train/`.

Source chinh:

- `sign_classify_train/HOG_SVM/train_hog_svm_6x3.ipynb`
- `sign_classify_train/HOG_SVM/train_hog_svm_8x2.ipynb`
- `sign_classify_train/HOG_RandomForest/train_hog_rf_6x3.ipynb`
- `sign_classify_train/HOG_RandomForest/train_hog_rf_8x2.ipynb`

Ket qua va tham so duoc doi chieu tu:

- `models/HOG_SVM_6x3.joblib`
- `models/HOG_SVM_8x2.joblib`
- `models/HOG_RandomForest_6x3.joblib`
- `models/HOG_RandomForest_8x2.joblib`
- `report_assets/classification_report_*.txt`

## 1. Tong quan pipeline

### 1. Pipeline training cua em gom nhung buoc nao?

Pipeline gom 9 buoc:

1. Tim thu muc `data` bang `resolve_data_dir()`.
2. Doc path anh va label tu `data/train` va `data/test` bang `collect_paths()`.
3. Chia `data/train` thanh train va validation bang `train_test_split(..., test_size=0.2, stratify=all_train_labels)`.
4. Doc anh, bo alpha neu co, chuyen grayscale va resize ve `128x128`.
5. Trich HOG tren anh grayscale da resize.
6. Trich HSV histogram tren anh mau da resize.
7. Noi HOG + HSV thanh mot vector feature.
8. Dung Optuna de tim hyperparameter tot tren validation.
9. Train lai model tot nhat tren train + validation, danh gia test va luu `.joblib`.

### 2. Vi sao khong train truc tiep tren pixel anh?

Vi pixel raw rat nhieu chieu, nhay voi anh sang, kich thuoc, nen va noise. Code chon cach trich dac trung co y nghia truoc:

- HOG nam hinh dang va canh.
- HSV histogram nam mau sac.

Sau khi bien anh thanh vector HOG+HSV, SVM va Random Forest hoc de hon.

### 3. Model dang giai bai toan gi?

Day la bai toan multi-class classification voi 5 lop:

- `Cam`
- `Chidan`
- `Hieulenh`
- `Nguyhiem`
- `Phu`

Trong code, label ban dau la ten folder. Sau do `LabelEncoder` chuyen label chu sang so de model hoc.

### 4. Vi sao trong code co 4 notebook train?

Vi code so sanh 2 thuat toan va 2 cau hinh HOG:

- HOG + SVM voi `6x3`.
- HOG + SVM voi `8x2`.
- HOG + Random Forest voi `6x3`.
- HOG + Random Forest voi `8x2`.

Tong cong la `2 thuat toan x 2 cau hinh = 4 notebook`.

### 5. Diem khac nhau lon nhat giua notebook SVM va Random Forest la gi?

Phan preprocessing va feature extraction giong nhau. Khac nhau o:

- SVM import `SVC`, RF import `RandomForestClassifier`.
- Ham `objective()` cua SVM search `C`, `kernel`, `gamma`, `class_weight`, `shrinking`.
- Ham `objective()` cua RF search `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- SVM co support vector summary.
- RF co feature importance.
- File model luu ra khac ten.

## 2. Cau hoi ve dataset va split

### 6. Dataset duoc doc nhu the nao?

Code dung `collect_paths(split_dir)`. Moi folder con trong `data/train` hoac `data/test` duoc xem la mot class. Moi file trong folder do duoc them vao `paths`, va ten folder duoc them vao label `y`.

### 7. Vi sao label duoc lay tu ten folder?

Vi dataset duoc to chuc theo cau truc:

```text
data/
  train/
    Cam/
    Chidan/
    Hieulenh/
    Nguyhiem/
    Phu/
  test/
    Cam/
    Chidan/
    Hieulenh/
    Nguyhiem/
    Phu/
```

Trong cach to chuc nay, folder class chinh la ground truth label.

### 8. Tai sao code sort class va sort path?

`labels = sorted(...)` va `sorted(...glob("*.*"))` giup thu tu class va thu tu anh on dinh giua cac lan chay. Neu khong sort, thu tu doc file co the khac nhau tuy he dieu hanh, lam kho tai lap ket qua.

### 9. Train, validation, test khac nhau nhu the nao?

- Train: dung de fit model.
- Validation: dung de Optuna chon hyperparameter.
- Test: dung de danh gia cuoi cung.

Trong code, `data/test` khong tham gia training hay Optuna.

### 10. Vi sao chia validation tu `data/train`, khong dung `data/test` de tune tham so?

Neu dung test de tune tham so, test khong con la du lieu doc lap nua. Ket qua test se bi optimistic bias. Code dung validation de chon tham so, sau do moi dung test de bao cao cuoi.

### 11. `VAL_SIZE = 0.2` co nghia gi?

20% cua `data/train` duoc tach ra lam validation, 80% con lai dung train. Theo report:

- Train: 2184 anh.
- Validation: 546 anh.
- Test: 303 anh.

### 12. `stratify=all_train_labels` co tac dung gi?

No giu ti le cac lop trong train va validation gan voi tap train ban dau. Vi dataset co lop `Phu` it mau hon cac lop khac, stratify giup validation khong bi lech class qua nang.

### 13. Dataset co can bang khong?

Khong hoan toan. So mau:

- `Cam`: 681 tong.
- `Chidan`: 630 tong.
- `Hieulenh`: 652 tong.
- `Nguyhiem`: 657 tong.
- `Phu`: 413 tong.

`Phu` it mau hon. Do do code co thu `class_weight="balanced"` trong SVM.

### 14. Neu train/test co class folder khac nhau thi code xu ly sao?

Code kiem tra:

```python
if set(train_classes) != set(test_classes):
    print("Warning: train/test class folders differ; using the union of labels.")
all_classes = sorted(set(train_classes) | set(test_classes))
```

Tuc la no canh bao va dung hop cac class de fit `LabelEncoder`.

### 15. Vi sao dung `LabelEncoder`?

SVM va Random Forest trong sklearn can label dang so. `LabelEncoder` doi ten lop thanh so, va sau predict co the doi nguoc so ve ten lop bang `inverse_transform()`.

### 16. Thu tu label encoded la gi?

Voi class sorted theo alphabet:

```text
Cam -> 0
Chidan -> 1
Hieulenh -> 2
Nguyhiem -> 3
Phu -> 4
```

## 3. Cau hoi ve tien xu ly anh

### 17. Anh duoc resize ve kich thuoc nao?

Tat ca anh duoc resize ve `IMAGE_SIZE = (128, 128)`.

### 18. Vi sao phai resize tat ca anh ve cung kich thuoc?

HOG tao vector co so chieu phu thuoc vao kich thuoc anh. SVM/RF can input co cung so chieu. Resize ve `128x128` dam bao moi anh co cung do dai feature.

### 19. Vi sao chon `128x128`?

Day la muc can bang:

- Du lon de giu hinh dang bien bao.
- Du nho de train SVM/RF nhanh.
- Phu hop voi cell size 8 va 6 trong HOG.

### 20. Anh RGBA duoc xu ly the nao?

Code co:

```python
if image.ndim == 3 and image.shape[-1] == 4:
    image = image[..., :3]
```

Neu anh co 4 kenh RGBA, code bo alpha va giu 3 kenh RGB. Alpha khong can cho HOG/HSV.

### 21. Tai sao HOG dung grayscale?

HOG dua tren gradient cuong do, chu yeu nam canh va hinh dang. Grayscale da du de tinh gradient. Dung grayscale giam do phuc tap va tranh model phu thuoc qua nhieu vao mau.

### 22. Tai sao van can anh mau cho HSV?

HSV histogram can thong tin mau. Code tinh HOG tren grayscale nhung tinh HSV tren anh RGB da resize. Hai loai feature bo sung cho nhau.

### 23. Neu anh da la grayscale thi code xu ly ra sao?

Trong `load_image_steps()`, neu `image.ndim != 3`, code chuyen anh sang float. Neu gia tri lon hon 1, chia cho 255 de dua ve [0, 1].

### 24. `anti_aliasing=True` trong resize co y nghia gi?

No giam artefact khi resize, dac biet khi thu nho anh. Neu khong anti-aliasing, anh co the bi rang cua, lam gradient HOG bi nhieu.

### 25. OpenCV resize nhan size theo thu tu nao?

OpenCV dung `(width, height)`, trong khi `IMAGE_SIZE` la `(height, width)`. Code xu ly:

```python
height, width = image_size
cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
```

### 26. `INTER_AREA` dung de lam gi?

`cv2.INTER_AREA` la interpolation phu hop khi resize anh, dac biet khi downsample. No thuong cho ket qua muot hon so voi nearest neighbor.

## 4. Cau hoi ve HOG

### 27. HOG la gi?

HOG la Histogram of Oriented Gradients. No mo ta anh bang histogram huong gradient cuc bo. Noi don gian, HOG tra loi cau hoi: trong tung vung nho cua anh, cac canh dang nghieng theo huong nao va manh den dau.

### 28. HOG tinh gradient nhu the nao?

Ve ly thuyet:

```text
Gx = I(x + 1, y) - I(x - 1, y)
Gy = I(x, y + 1) - I(x, y - 1)
magnitude = sqrt(Gx^2 + Gy^2)
angle = atan2(Gy, Gx)
```

Sau do moi pixel dong gop vao bin huong tuong ung trong cell.

### 29. `orientations = 9` nghia la gi?

Moi cell co histogram gradient gom 9 bin huong. Neu dung unsigned gradient 0-180 do, moi bin dai khoang 20 do. 9 bin la cau hinh pho bien, du chi tiet de nam huong canh nhung khong qua nhieu chieu.

### 30. `pixels_per_cell` la gi?

La kich thuoc moi cell HOG theo pixel. Code co 2 bien the:

- `(6, 6)` trong cau hinh 6x3.
- `(8, 8)` trong cau hinh 8x2.

Cell nho hon bat chi tiet min hon nhung de nhay voi noise hon.

### 31. `cells_per_block` la gi?

La so cell trong moi block dung de normalize. Code co:

- `(3, 3)` trong cau hinh 6x3.
- `(2, 2)` trong cau hinh 8x2.

Block gom nhieu cell de descriptor on dinh hon voi thay doi anh sang cuc bo.

### 32. `block_norm = "L2-Hys"` la gi?

`L2-Hys` la chuan hoa vector block bang L2, clip cac gia tri lon, roi normalize lai. No giup HOG it nhay voi thay doi do sang va contrast.

### 33. Tai sao HOG can block normalization?

Neu anh sang thay doi, magnitude gradient co the tang/giam. Normalization giup feature phu thuoc nhieu hon vao ti le/huong gradient thay vi do sang tuyet doi.

### 34. 6x3 nghia la gi?

Trong code:

```python
"pixels_per_cell": (6, 6)
"cells_per_block": (3, 3)
```

Nen 6x3 nghia la cell 6x6 pixel, block 3x3 cell.

### 35. 8x2 nghia la gi?

Trong code:

```python
"pixels_per_cell": (8, 8)
"cells_per_block": (2, 2)
```

Nen 8x2 nghia la cell 8x8 pixel, block 2x2 cell.

### 36. So chieu HOG 8x2 tinh nhu the nao?

Anh `128x128`, cell `8x8`:

```text
cells_per_axis = floor(128 / 8) = 16
blocks_per_axis = 16 - 2 + 1 = 15
values_per_block = 2 * 2 * 9 = 36
HOG_dim = 15 * 15 * 36 = 8100
```

Them HSV 32 chieu, tong feature la `8132`.

### 37. So chieu HOG 6x3 tinh nhu the nao?

Anh `128x128`, cell `6x6`:

```text
cells_per_axis = floor(128 / 6) = 21
blocks_per_axis = 21 - 3 + 1 = 19
values_per_block = 3 * 3 * 9 = 81
HOG_dim = 19 * 19 * 81 = 29241
```

Them HSV 32 chieu, tong feature la `29273`.

### 38. Tai sao 6x3 co nhieu feature hon 8x2?

Vi cell 6x6 nho hon nen tren anh co nhieu cell hon. Block 3x3 cung co nhieu cell trong moi block hon. Ket qua la descriptor dai hon nhieu.

### 39. Nhieu feature hon co luon tot hon khong?

Khong. Ket qua code cho thay 8x2 tot hon 6x3. 6x3 nhieu feature hon nhung co the hoc ca noise, chi tiet nen, sai khac crop, lam tong quat hoa kem hon.

### 40. Vi sao HOG phu hop voi bien bao giao thong?

Bien bao co hinh dang va vien rat ro: tron, tam giac, mui ten, khung, duong bien. HOG nam cac pattern canh nay rat tot.

### 41. HOG co bat duoc mau khong?

Khong. HOG trong code tinh tren grayscale nen chu yeu bat hinh dang. Vi vay code bo sung HSV histogram de bat mau.

### 42. Anh HOG visualization co dung de train khong?

Khong. Anh HOG visualization chi dung de hien thi. Feature train la vector descriptor tra ve tu `hog(gray_resized, **hog_params)`.

### 43. `enhance_hog_image()` co lam thay doi ket qua train khong?

Khong. Ham nay chi lam dep anh HOG visualization bang normalize, gamma, dilation va colormap. Vector HOG train khong di qua ham nay.

## 5. Cau hoi ve HSV histogram

### 44. HSV la gi?

HSV gom:

- H: Hue, mau sac.
- S: Saturation, do bao hoa.
- V: Value, do sang.

Trong OpenCV, H nam trong `[0, 180]`, S va V nam trong `[0, 256]`.

### 45. `HSV_HIST_BINS = (16, 8, 8)` nghia la gi?

Code chia:

- H thanh 16 bin.
- S thanh 8 bin.
- V thanh 8 bin.

Tong HSV feature la `16 + 8 + 8 = 32`.

### 46. Code co tinh histogram HSV 3D khong?

Khong. Code tinh histogram rieng tung kenh:

```python
hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
```

Sau do concatenate 3 histogram lai.

### 47. Tai sao khong dung HSV histogram 3D?

Histogram 3D voi `16x8x8` se co 1024 chieu. Code chon histogram rieng tung kenh de vector gon hon, giam nguy co overfit, va du de bo sung thong tin mau tong quat.

### 48. Tai sao HSV feature duoc normalize?

Code co:

```python
color_feature /= color_feature.sum() + 1e-8
```

Normalize giup histogram the hien ti le mau thay vi phu thuoc so pixel. `1e-8` tranh chia cho 0.

### 49. HSV co mat thong tin vi tri khong?

Co. Histogram mau toan anh khong biet mau nam o dau. Nhung vi dataset la anh bien bao da crop tuong doi sat, mau tong the van co ich. Thong tin vi tri/hinh dang duoc HOG bo sung.

### 50. Tai sao dung HSV thay vi RGB histogram?

HSV tach mau sac khoi do sang tot hon RGB. Khi anh sang thay doi, hue co the on dinh hon gia tri RGB raw. Bien bao giao thong co mau dac trung nen HSV phu hop.

### 51. HSV giup phan biet lop nao?

HSV co the giup voi cac lop co mau dac trung. Vi du bien cam/nguy hiem thuong co mau do, bien chi dan co the co mau xanh, bien phu co mau/phoi mau khac. Tuy nhien HSV khong du mot minh vi nhieu lop co mau giong nhau.

## 6. Cau hoi ve feature vector va cache

### 52. Feature cuoi cung cua mot anh duoc tao nhu the nao?

Trong `extract_combined_feature()`:

```python
hog_feature = hog(gray_resized, **hog_params).astype(np.float32)
color_feature = extract_hsv_histogram(image, image_size, hsv_bins)
return np.concatenate([hog_feature, color_feature])
```

Tuc la:

```text
feature = [HOG descriptors, HSV histogram]
```

### 53. So chieu feature cuoi cung cua 8x2 la bao nhieu?

`8132`, gom:

- HOG: 8100.
- HSV: 32.

### 54. So chieu feature cuoi cung cua 6x3 la bao nhieu?

`29273`, gom:

- HOG: 29241.
- HSV: 32.

### 55. Vi sao dung `np.float32` cho feature?

`float32` giam bo nho so voi `float64`, du chinh xac cho ML truyen thong, va giup cache/model gon hon.

### 56. `compute_features()` lam gi?

Ham nay trich feature cho ca split train/validation/test. No:

1. Thu load cache neu co.
2. Neu cache khong hop le, trich feature tung anh.
3. Chay song song bang `joblib.Parallel`.
4. Bo qua anh loi.
5. Stack feature thanh ma tran `X`.
6. Luu cache `.npz`.

### 57. Tai sao can cache feature?

Trich HOG cho hang nghin anh ton thoi gian. Cache `.npz` luu `X` va `y`, lan sau chay lai co the load truc tiep neu so sample khop.

### 58. Cache co the bi sai khi doi tham so HOG khong?

Code giam rui ro bang `cache_tag` gom model type, HOG config, HSV bins, image size, validation ratio va seed. Vi vay cache 6x3 va 8x2 co ten khac nhau.

### 59. Neu them/xoa anh thi cache xu ly sao?

Code kiem tra `len(y_cached) == len(labels)`. Neu so sample khong khop, code in canh bao va recompute cache.

### 60. Diem yeu cua cach check cache nay la gi?

No chi check so luong sample, khong check danh sach file/path. Neu thay anh nhung tong so anh giu nguyen, cache co the van duoc dung. Neu can chat che hon, nen luu kem path list/hash anh trong cache.

### 61. Tai sao dung `Parallel(n_jobs=N_JOBS)`?

Feature extraction cho tung anh doc lap nhau, nen co the song song hoa. Code dat `N_JOBS = 8` de tang toc.

### 62. Neu mot anh bi loi doc thi sao?

Trong `extract(path)`, code `try/except` va tra `None` neu loi. Sau do `compute_features()` bo qua feature `None` va dem `skipped`.

## 7. Cau hoi ve SVM

### 63. SVM la gi?

SVM la Support Vector Machine. No tim hyperplane phan tach cac lop sao cho margin lon nhat. Cac diem quan trong nam gan bien phan tach duoc goi la support vectors.

### 64. Margin trong SVM la gi?

Margin la khoang cach tu hyperplane toi cac diem gan nhat cua moi lop. SVM co gang toi da hoa margin de model tong quat hoa tot hon.

### 65. Support vector la gi?

Support vector la cac mau train nam gan bien quyet dinh hoac vi pham margin. Chung quyet dinh vi tri hyperplane. Neu bo cac diem khong phai support vector, bien quyet dinh co the khong doi nhieu.

### 66. SVM trong code dung class nao?

Notebook SVM dung:

```python
from sklearn.svm import SVC
```

Final model la `SVC(**best_params)`.

### 67. SVM trong code co search nhung tham so nao?

Trong `objective()`:

- `kernel`: `linear` hoac `rbf`.
- `C`: tu `1e-2` den `1e2`, log scale.
- `class_weight`: `None` hoac `"balanced"`.
- `shrinking`: `True` hoac `False`.
- `gamma`: tu `1e-4` den `1e0` neu kernel la `rbf`.

### 68. `C` trong SVM co y nghia gi?

`C` dieu khien muc phat khi phan loai sai:

- `C` lon: phat sai nang, fit train chat hon, de overfit hon.
- `C` nho: chap nhan sai de margin rong hon, regularization manh hon.

### 69. Tai sao `C` search theo log scale?

Vi anh huong cua `C` nam theo bac do lon. Khac biet giua 0.01 va 0.1 tuong tu khac biet giua 10 va 100 ve ty le. Log scale giup search deu tren cac bac do lon.

### 70. `kernel="linear"` la gi?

Linear kernel tim bien phan tach tuyen tinh trong khong gian feature HOG+HSV. Neu feature da tot, linear SVM co the du manh va it overfit hon RBF.

### 71. `kernel="rbf"` la gi?

RBF la kernel phi tuyen, cho phep bien quyet dinh cong/phuc tap hon. No co them tham so `gamma`.

### 72. `gamma` trong RBF co y nghia gi?

`gamma` dieu khien pham vi anh huong cua moi sample:

- `gamma` lon: anh huong hep, bien quyet dinh phuc tap, de overfit.
- `gamma` nho: anh huong rong, bien quyet dinh muot hon.

### 73. Vi sao final SVM lai dung linear kernel?

Tham so thuc te trong model da luu:

- SVM 6x3: `kernel="linear"`.
- SVM 8x2: `kernel="linear"`.

Optuna chon linear vi validation accuracy tot nhat. Dieu nay cho thay HOG+HSV da tao feature space ma cac lop co the tach tuyen tinh kha tot.

### 74. `class_weight="balanced"` co tac dung gi?

No tang trong so cho class it mau hon theo tan suat class. Trong dataset, `Phu` it mau hon, nen balanced co the giup model khong bo qua class nho.

### 75. Final SVM 8x2 co tham so gi?

Theo model da luu:

```text
kernel = linear
C = 0.03582517616218419
class_weight = balanced
shrinking = False
probability = True
cache_size = 1000
random_state = 42
```

### 76. Final SVM 6x3 co tham so gi?

Theo model da luu:

```text
kernel = linear
C = 7.356130968262376
class_weight = None
shrinking = False
probability = True
cache_size = 1000
random_state = 42
```

### 77. Tai sao SVM 8x2 co `C` nho hon SVM 6x3 rat nhieu?

Day la ket qua Optuna. Co the giai thich:

- 8x2 feature gon hon, model chi can margin rong va regularization manh.
- 6x3 feature nhieu chieu hon, Optuna chon `C` lon hon de fit validation tot hon.
- Tuy nhien test cho thay 8x2 tong quat hoa tot hon.

### 78. `shrinking=False` co nghia gi?

Shrinking la heuristic trong solver SVM de bo tam thoi cac bien khong co kha nang la support vector, nham tang toc. Optuna chon `False` cho ca hai final SVM, nghia la trong validation setting nay tat shrinking cho ket qua tot hon hoac on dinh hon.

### 79. `probability=True` trong final SVM de lam gi?

No cho phep goi `predict_proba()` de lay xac suat uoc luong. Trong code search Optuna, `probability=False` de train nhanh hon. Khi train final, code bat `SVM_FINAL_PROBABILITY=True` de phuc vu hien confidence trong demo/visualization.

### 80. Xac suat cua SVM co phai xac suat truc tiep khong?

Khong hoan toan. SVM goc cho decision function/margin. Sklearn dung them calibration de uoc luong probability khi `probability=True`.

### 81. Support vector cua final SVM la bao nhieu?

Theo model da luu:

- SVM 6x3: `[261, 401, 369, 169, 268]`, tong `1468`.
- SVM 8x2: `[253, 387, 344, 145, 251]`, tong `1380`.

Thu tu lop la `[Cam, Chidan, Hieulenh, Nguyhiem, Phu]`.

### 82. SVM 8x2 co it support vector hon noi len dieu gi?

No can it diem train hon de xac dinh bien phan tach, dong thoi accuracy cao hon. Co the noi 8x2 tao feature space gon va tach lop tot hon.

## 8. Cau hoi ve Random Forest

### 83. Random Forest la gi?

Random Forest la tap hop nhieu decision tree. Moi cay duoc train tren mau bootstrap va tai moi split chi xem mot tap con feature. Ket qua cuoi la bo phieu da so cua cac cay.

### 84. Vi sao Random Forest giam overfit so voi mot decision tree?

Mot cay don le de overfit. Random Forest ket hop nhieu cay khac nhau, lam giam variance. Neu cac cay khong hoan toan giong nhau, bo phieu trung binh se on dinh hon.

### 85. RF trong code dung class nao?

Notebook RF dung:

```python
from sklearn.ensemble import RandomForestClassifier
```

Final model la `RandomForestClassifier(**best_params)`.

### 86. RF search nhung tham so nao?

Trong `objective()`:

- `max_depth`: 5 den 50.
- `min_samples_split`: 2 den 20.
- `min_samples_leaf`: 1 den 10.
- `max_features`: `"sqrt"` hoac `"log2"`.

So cay duoc tang theo `ESTIMATOR_STEPS = [50, 100, 150, 200]`.

### 87. `n_estimators` la gi?

La so cay trong rung. Final RF dung `n_estimators = 200`.

### 88. Vi sao dung 200 cay?

Trong code, `ESTIMATOR_STEPS = [50, 100, 150, 200]`. Final model update:

```python
"n_estimators": max(ESTIMATOR_STEPS)
```

nen dung 200 cay. Nhieu cay hon thuong on dinh hon, nhung ton thoi gian hon.

### 89. `max_depth` la gi?

La do sau toi da cua moi tree:

- Sau lon: hoc quan he phuc tap, de overfit.
- Sau nho: model don gian hon, co the underfit.

Final:

- RF 6x3: `max_depth=45`.
- RF 8x2: `max_depth=21`.

### 90. `min_samples_split` la gi?

So mau toi thieu de mot node duoc phep tach. Gia tri lon lam cay it tach hon, giam overfit.

Final:

- RF 6x3: `min_samples_split=9`.
- RF 8x2: `min_samples_split=4`.

### 91. `min_samples_leaf` la gi?

So mau toi thieu o moi leaf. Gia tri lon hon lam leaf khong qua nho, giam overfit va lam du doan muot hon.

Final:

- RF 6x3: `min_samples_leaf=3`.
- RF 8x2: `min_samples_leaf=2`.

### 92. `max_features="sqrt"` la gi?

Tai moi split, moi cay chi xem can bac hai cua tong so feature. Cach nay lam cac cay da dang hon va la cau hinh pho bien cho Random Forest classification.

### 93. Vi sao RF ca 6x3 va 8x2 deu chon `sqrt`?

Optuna chon `sqrt` vi validation accuracy tot hon `log2` trong cac trial. Voi feature HOG nhieu chieu, `sqrt` cho moi split xem nhieu feature hon `log2`, co the giup cay co thong tin hon.

### 94. `warm_start=True` trong objective RF de lam gi?

No cho phep tang `n_estimators` tu 50 len 100, 150, 200 ma co the tai su dung cac cay da train. Muc tieu la danh gia cung mot bo tham so khi so cay tang dan.

### 95. Vi sao final RF dat `warm_start=False`?

Final model khong can tang dan so cay nua. Code train truc tiep 200 cay tren train+validation, nen dat `warm_start=False`.

### 96. Vi sao trong objective RF `n_jobs=1`, nhung final RF `n_jobs=4`?

Objective dang chay Optuna song song voi `OPTUNA_JOBS=4`. Neu moi trial lai dung nhieu job, CPU co the bi qua tai. Final chi train mot model, nen dung `RF_JOBS=4` de nhanh hon.

### 97. Final RF 8x2 co tham so gi?

```text
n_estimators = 200
max_depth = 21
min_samples_split = 4
min_samples_leaf = 2
max_features = sqrt
random_state = 42
n_jobs = 4
```

### 98. Final RF 6x3 co tham so gi?

```text
n_estimators = 200
max_depth = 45
min_samples_split = 9
min_samples_leaf = 3
max_features = sqrt
random_state = 42
n_jobs = 4
```

### 99. Vi sao RF 6x3 can cay sau hon RF 8x2?

6x3 co 29273 feature, chi tiet va phuc tap hon. Optuna chon `max_depth=45` de cay du suc fit feature space nay. Nhung test accuracy van kem hon, cho thay fit phuc tap hon khong dong nghia tong quat hoa tot hon.

### 100. `feature_importances_` cua RF la gi?

La do quan trong cua feature dua tren muc giam impurity trong cac tree. Code ve top 25 feature co importance cao nhat bang `plot_feature_importance()`.

### 101. Diem can luu y khi giai thich feature importance trong code?

Code dat label truc y la `HOG[i]` cho tat ca index. Neu index roi vao 32 chieu HSV cuoi thi label nay khong hoan toan chinh xac. Tuy nhien HOG chiem phan lon feature nen top feature thuong la HOG.

### 102. Random Forest tinh `predict_proba()` nhu the nao?

Xac suat la ti le cay bo phieu cho moi class. Vi du 180/200 cay chon `Cam` thi xac suat `Cam` gan 0.9.

## 9. Cau hoi ve Optuna va hyperparameter tuning

### 103. Optuna duoc dung de lam gi?

Optuna tu dong tim hyperparameter tot nhat tren validation set. No giup tranh viec thu tay qua nhieu cau hinh.

### 104. `objective(trial)` la gi?

Do la ham ma Optuna goi cho moi trial. Ham nay:

1. De xuat bo tham so.
2. Train model tren train split.
3. Danh gia tren validation split.
4. Tra ve validation accuracy.

### 105. `direction="maximize"` co nghia gi?

Optuna se toi da hoa gia tri objective. Trong code, objective tra ve validation accuracy, nen can maximize.

### 106. `TPESampler` la gi?

TPE la Tree-structured Parzen Estimator, mot phuong phap Bayesian optimization. No dung ket qua cac trial truoc de de xuat bo tham so co kha nang tot hon.

### 107. `MedianPruner` la gi?

No cat som trial co ket qua kem hon median cac trial truoc. RF report accuracy sau moi muc so cay; SVM report mot lan sau khi train. Pruner giup tiet kiem thoi gian.

### 108. SVM chay bao nhieu Optuna trial?

`OPTUNA_TRIALS = 30` va `OPTUNA_JOBS = 4`.

### 109. RF chay bao nhieu Optuna trial?

`OPTUNA_TRIALS = 50` va `OPTUNA_JOBS = 4`.

### 110. Vi sao RF dung 50 trial con SVM dung 30 trial?

RF co nhieu tuong tac giua cac tham so cay va qua trinh tang so cay, nen code cho RF search nhieu trial hon. SVM search it hon de tiet kiem thoi gian vi SVC voi feature nhieu chieu co the train cham.

### 111. Sau khi Optuna chon tham so, vi sao train lai tren train + validation?

Validation da hoan thanh vai tro chon tham so. De model cuoi cung co nhieu du lieu hoc hon, code gop:

```python
X_combined = np.vstack([X_train, X_val])
y_combined = np.concatenate([y_train_enc, y_val_enc])
```

rồi train final model tren tap gop nay.

### 112. Co bi data leakage khong khi train lai tren train + validation?

Khong, vi test set van hoan toan doc lap. Validation duoc dung de chon tham so, sau do duoc gop vao train final la cach lam pho bien. Dieu quan trong la khong dung test de chon tham so.

## 10. Cau hoi ve ket qua va metric

### 113. Model nao tot nhat?

Tot nhat theo accuracy va weighted F1 la `HOG SVM 8x2`:

```text
Accuracy = 0.9142
Weighted F1 = 0.9142
Macro F1 = 0.9080
```

### 114. Thu tu ket qua cac model la gi?

Theo report:

1. HOG SVM 8x2: accuracy `0.9142`.
2. HOG RF 8x2: accuracy `0.9109`.
3. HOG SVM 6x3: accuracy `0.9010`.
4. HOG RF 6x3: accuracy `0.8878`.

### 115. Accuracy la gi?

Accuracy la ti le mau test du doan dung:

```text
accuracy = so_du_doan_dung / tong_so_mau
```

### 116. Precision la gi?

Precision cua mot lop la trong so mau model du doan la lop do, co bao nhieu mau dung:

```text
precision = TP / (TP + FP)
```

### 117. Recall la gi?

Recall cua mot lop la trong so mau that su thuoc lop do, model tim dung bao nhieu:

```text
recall = TP / (TP + FN)
```

### 118. F1-score la gi?

F1 la trung binh dieu hoa cua precision va recall:

```text
F1 = 2 * precision * recall / (precision + recall)
```

### 119. Macro F1 khac weighted F1 nhu the nao?

- Macro F1: trung binh F1 cac lop, moi lop co trong so bang nhau.
- Weighted F1: trung binh F1 co trong so theo support tung lop.

Macro F1 quan trong khi dataset lech lop.

### 120. Tai sao khong chi bao cao accuracy?

Vi dataset khong hoan toan can bang. Accuracy co the cao neu model lam tot tren lop nhieu mau nhung kem tren lop it mau. Precision/recall/F1 cho cai nhin chi tiet hon.

### 121. Lop nao SVM 8x2 lam tot nhat?

`Nguyhiem` co recall `1.0000` va F1 `0.9925`, rat cao. `Cam` cung cao voi F1 `0.9706`.

### 122. Lop nao SVM 8x2 con kho?

`Chidan` co precision `0.8028`, cho thay co mot so mau lop khac bi du doan nham thanh `Chidan`. `Phu` co recall `0.8049`, cho thay mot so mau `Phu` bi du doan sang lop khac.

### 123. RF 8x2 co diem manh gi?

RF 8x2 co recall `1.0000` cho `Cam` va `Chidan`, precision `1.0000` cho `Hieulenh` va `Phu`. Dieu nay cho thay RF rat tu tin voi mot so lop, nhung co xu huong keo nhieu mau ve `Chidan` lam precision `Chidan` thap hon.

### 124. Confusion matrix dung de xem gi?

No cho biet model nham lop nao voi lop nao. Hang la label that, cot la label du doan. O ngoai duong cheo chinh la loi.

### 125. Bang misclassification trong code dung de lam gi?

`build_misclassification_df()` liet ke tat ca mau sai, gom true label, predicted label, confidence, file name va path. Sau do code group theo cap nham lan de xem loi pho bien.

## 11. Cau hoi ve so sanh 6x3 va 8x2

### 126. 8x2 va 6x3 khac nhau co ban o dau?

Khac o cau hinh HOG:

- 8x2: cell `8x8`, block `2x2`, HOG dim `8100`.
- 6x3: cell `6x6`, block `3x3`, HOG dim `29241`.

### 127. Cau hinh nao cho ket qua tot hon?

8x2 tot hon trong ca SVM va RF:

- SVM 8x2: `0.9142` > SVM 6x3: `0.9010`.
- RF 8x2: `0.9109` > RF 6x3: `0.8878`.

### 128. Ket qua nay noi len dieu gi?

No cho thay trong dataset nay, descriptor gon va on dinh cua 8x2 tong quat hoa tot hon descriptor qua chi tiet cua 6x3.

### 129. Tai sao 6x3 bat chi tiet hon nhung lai kem hon?

Vi chi tiet nho khong chi gom thong tin bien bao, ma con gom noise, nen, do mo, sai khac crop. Khi so chieu tang tu `8132` len `29273`, model can nhieu du lieu hon de hoc on dinh.

### 130. Neu giam khao hoi "6x3 co nhieu feature hon, sao khong tot hon?", tra loi the nao?

Tra loi:

> Nhieu feature hon khong dam bao tot hon. 6x3 co do phan giai HOG cao hon nhung cung nhay voi noise hon. Voi dataset hien tai, 8x2 giu du hinh dang lon cua bien bao, it chieu hon, giam overfitting, nen test accuracy cao hon.

### 131. 6x3 co uu diem gi?

6x3 co the bat duoc canh nho, chi tiet cuc bo, ky hieu nho tren bien bao. Neu dataset lon hon, crop tot hon, va anh chat luong cao hon, 6x3 co the co loi the.

### 132. 8x2 co uu diem gi?

8x2 vector ngan hon, train nhanh hon, it bo nho hon, va ket qua trong project tot hon. No bat hinh dang cap vung lon, phu hop voi bien bao giao thong.

## 12. Cau hoi ve luu model va inference

### 133. Model duoc luu o dau?

Trong thu muc `models/`:

- `HOG_SVM_6x3.joblib`
- `HOG_SVM_8x2.joblib`
- `HOG_RandomForest_6x3.joblib`
- `HOG_RandomForest_8x2.joblib`

### 134. Trong file `.joblib` luu nhung gi?

Payload luu:

- `model`: classifier da train.
- `label_encoder`: de doi label so ve ten lop.
- `feature_extractor`: `"HOG_HSV"`.
- `hog_params`: tham so HOG.
- `hsv_hist_bins`: bins HSV.
- `image_size`: `(128, 128)`.
- `classes`: danh sach lop.
- Rieng SVM co them `algorithm` va `svm_params`.

### 135. Tai sao phai luu ca `hog_params` va `hsv_hist_bins`?

Inference bat buoc trich feature dung y het luc train. Neu dung sai HOG config hoac HSV bins, so chieu feature khac hoac y nghia feature khac, model se predict sai.

### 136. `predict_one_image()` lam gi?

No lap lai pipeline:

1. Load anh.
2. Resize/grayscale.
3. Tinh HOG visualization.
4. Tinh HOG+HSV feature.
5. Reshape thanh `(1, n_features)`.
6. `model.predict()`.
7. `label_encoder.inverse_transform()`.
8. Lay `predict_proba()` neu model co.

### 137. Vi sao phai `feature.reshape(1, -1)`?

Sklearn model can input 2D dang `(n_samples, n_features)`. Mot anh co vector 1D, nen reshape thanh 1 sample.

### 138. Neu inference dung model 8x2 nhung trich feature 6x3 thi sao?

Se sai. Model 8x2 can `8132` feature, 6x3 tao `29273` feature. Sklearn se bao loi dimension mismatch.

### 139. Neu label encoder khac luc train thi sao?

Du doan so co the bi doi sai ten lop. Vi vay payload luu chinh `label_encoder` da fit luc train.

## 13. Cau hoi ly thuyet mo rong

### 140. SVM va RF khac nhau ve ban chat nhu the nao?

SVM tim bien phan tach trong khong gian feature, dua nhieu vao support vectors va margin. RF la ensemble cua decision trees, dua tren nhieu quy tac split va bo phieu.

### 141. Khi nao SVM co loi the hon RF?

SVM thuong tot voi feature vector lien tuc, high-dimensional, nhu HOG. Linear SVM co the tong quat hoa tot khi feature da tach lop kha ro.

### 142. Khi nao RF co loi the hon SVM?

RF tot voi quan he phi tuyen, feature hon hop, it can scale feature, co feature importance de giai thich. Tuy nhien voi feature rat nhieu chieu nhu HOG, RF co the can nhieu cay va de bi anh huong boi feature noise.

### 143. Co can standardize feature truoc SVM khong?

Trong code khong dung `StandardScaler`. HOG da duoc normalize theo block, HSV histogram duoc normalize tong ve 1. Tuy nhien neu cai tien, co the thu `StandardScaler` hoac `MinMaxScaler` de xem SVM co tot hon khong.

### 144. Vi sao RF khong can scaling nhieu?

Decision tree split theo nguong tren tung feature rieng le, nen khong nhay voi scale feature nhu cac model dua tren khoang cach/margin. Vi vay RF thuong khong can standardization.

### 145. HOG co invariant voi rotation khong?

Khong hoan toan. HOG nam huong gradient, nen anh xoay manh co the lam histogram huong doi. Neu bien bao trong test bi xoay nhieu, model co the kem hon.

### 146. HOG co invariant voi scale khong?

Khong hoan toan. Code resize anh ve `128x128`, nen neu bien bao da crop sat thi scale duoc chuan hoa. Nhung neu anh full scene va bien bao chi chiem vung nho, HOG se bi anh huong.

### 147. HSV co invariant voi anh sang khong?

HSV tot hon RGB trong viec tach hue va brightness, nhung khong hoan toan invariant. Anh qua toi, qua sang, bi bong, hay camera sai mau van co the lam HSV histogram thay doi.

### 148. Vi sao khong dung LBP?

LBP phu hop texture, con bien bao giao thong manh ve hinh dang, duong bien va mau. HOG+HSV phu hop hon voi muc tieu nam shape + color.

### 149. Vi sao khong dung CNN?

CNN co the manh hon trong bai toan anh phuc tap, nhung can nhieu du lieu/tinh toan hon va kho giai thich hon. Do an nay dung classical CV de pipeline minh bach: HOG/HSV co the giai thich bang hinh dang va mau.

### 150. Han che lon nhat cua pipeline nay la gi?

Can anh crop/localization tot. Neu anh co nen phuc tap, bien bao qua nho, mo, che khuat, xoay manh, HOG+HSV co the khong du robust.

## 14. Cau hoi thuc nghiem va bao ve ket qua

### 151. Ket qua co dang tin khong?

Ket qua dang tin trong pham vi dataset hien tai vi test set rieng co 303 anh va khong dung de tune. Tuy nhien muon khang dinh manh hon nen dung cross-validation hoac test tren du lieu thuc te ben ngoai.

### 152. Tai sao test chi co 303 anh co du khong?

303 anh cho danh gia ban dau la du de so sanh cac cau hinh trong do an. Nhung de danh gia san pham thuc te, nen co test set lon va da dang hon.

### 153. Model co overfit khong?

Khong the khang dinh chi tu test report, nhung dau hieu la 6x3 nhieu feature hon lai test kem hon 8x2, co the do 6x3 overfit/noise hon. De kiem tra ro hon can so sanh train accuracy va validation/test accuracy.

### 154. Tai sao SVM 8x2 tot hon RF 8x2 chi mot chut?

SVM 8x2 accuracy `0.9142`, RF 8x2 `0.9109`. Chenh lech nho, cho thay ca hai deu khai thac HOG+HSV tot. SVM co loi the nhe voi feature HOG high-dimensional.

### 155. Nen chon model nao de demo?

Nen chon `HOG_SVM_8x2` vi accuracy va weighted F1 cao nhat, feature gon hon 6x3, support vector it hon SVM 6x3, va ket qua tong quat tot.

### 156. Neu uu tien toc do inference thi chon model nao?

Can benchmark thuc te, nhung `8x2` nen nhanh hon `6x3` vi feature it hon. Giua SVM va RF, toc do tuy implementation; RF 200 cay co the ton thoi gian bo phieu, SVM linear voi 1380 support vectors cung co chi phi. Trong project, `HOG_SVM_8x2` la lua chon can bang.

### 157. Neu giam khao hoi "ket qua 91.42% co cao khong?", tra loi sao?

Tra loi:

> Voi pipeline classical CV khong dung deep learning, accuracy 91.42% tren 5 lop la ket qua kha tot. Tuy nhien no van phu thuoc dataset crop va dieu kien anh. Em khong khang dinh model da san sang cho moi tinh huong thuc te, ma xem day la baseline HOG+HSV+SVM tot nhat trong cac thi nghiem.

### 158. Vi sao lop `Nguyhiem` co ket qua cao?

Co the vi bien nguy hiem co hinh dang/mau sac ro, mau do/vien/cau truc dac trung, nen HOG va HSV de phan biet. Trong SVM 8x2, recall `Nguyhiem = 1.0000`, F1 `0.9925`.

### 159. Vi sao `Chidan` precision thap hon?

Trong SVM 8x2, precision `Chidan = 0.8028`; RF 8x2 precision `Chidan = 0.7500`. Nghia la co nhieu anh lop khac bi du doan thanh `Chidan`. Nguyen nhan co the do hinh dang/mau/crop cua mot so anh gan voi `Chidan`, hoac feature chua du phan biet ky hieu noi dung.

### 160. Vi sao `Phu` recall thap hon mot so lop?

`Phu` co it mau hon, support test 41 anh. It du lieu hon lam model kho hoc du bien thien cua lop. Trong SVM 8x2, recall `Phu = 0.8049`; RF 8x2 recall `Phu = 0.7561`.

### 161. Neu muon cai thien ket qua, em se lam gi?

Co the:

1. Tang du lieu, dac biet lop `Phu`.
2. Data augmentation: rotate nhe, brightness, blur, crop jitter.
3. Thu scaler cho SVM.
4. Thu HOG config khac: `pixels_per_cell=(8,8)` voi block khac, orientations 12.
5. Thu color histogram theo vung thay vi global.
6. Thu CNN hoac transfer learning neu du lieu/tai nguyen cho phep.

### 162. Vi sao code khong augmentation?

Notebook hien tai tap trung vao training baseline HOG+HSV va so sanh SVM/RF. Augmentation la huong cai tien hop ly, nhung chua nam trong pipeline hien tai.

### 163. Neu anh test khong crop sat bien bao thi sao?

HOG+HSV se bi anh huong boi nen. HOG nam canh cua nen, HSV nam mau nen, nen predict co the sai. Demo co them localization/crop, nhung training notebook gia dinh anh da nam trong dataset crop theo class.

### 164. Tai sao trong README cu co noi demo khong train lai?

Vi demo load model `.joblib` da train san trong `models/`. Training chi nam trong notebook. Khi demo predict, no dung payload da luu de trich feature dung config va predict.

## 15. Cau hoi ve code robustness

### 165. `resolve_data_dir()` co loi ich gi?

No tim `data/train` va `data/test` tu current working directory di len cac parent. Nho do notebook co the chay tu root hoac tu subfolder ma van tim duoc dataset.

### 166. `show_table()` vi sao can try/except?

Trong Jupyter co `display(df)`, nhung khi chay nhu script co the khong co `display`. Try/except giup code linh hoat trong ca hai moi truong.

### 167. Vi sao `zero_division=0` trong `classification_report()`?

Neu mot lop nao do khong duoc predict, precision co the bi chia cho 0. `zero_division=0` tranh warning/loi va gan metric do bang 0.

### 168. Vi sao confusion matrix dung `labels=np.arange(len(le.classes_))`?

De dam bao matrix co day du cac class theo dung thu tu label encoder, ke ca neu mot class nao do khong xuat hien trong prediction.

### 169. Vi sao `cache_size=1000` trong SVM?

SVM kernel computation co the dung cache. `1000` MB giup giam tinh lai kernel va co the tang toc train, dac biet khi thu RBF. Final linear van luu tham so nay trong model.

### 170. Import `exposure` co dung khong?

Trong notebook hien tai, `from skimage import exposure` duoc import nhung khong dung. Co the la phan con lai tu thu nghiem truoc. No khong anh huong ket qua train.

## 16. Cau tra loi mau khi bi hoi tong hop

### 171. Hay tom tat toan bo phan training trong 1 phut.

Em train 4 model gom SVM va Random Forest voi hai cau hinh HOG 6x3 va 8x2. Moi anh duoc resize ve 128x128, trich HOG tren grayscale de lay hinh dang, trich HSV histogram tren anh mau de lay mau sac, sau do concatenate thanh vector feature. Em chia train thanh train/validation theo ti le 80/20 co stratify, dung Optuna tune hyperparameter tren validation, roi train final model tren train+validation va danh gia tren test. Ket qua tot nhat la HOG SVM 8x2 voi accuracy 0.9142.

### 172. Neu hoi "dong gop chinh cua feature extraction la gi?", tra loi sao?

HOG dong gop thong tin hinh dang/canh, HSV dong gop thong tin mau. Bien bao giao thong phu thuoc ca hinh dang va mau, nen ket hop hai loai feature giup model phan loai tot hon so voi chi dung mot loai.

### 173. Neu hoi "vi sao model tot nhat la SVM 8x2?", tra loi sao?

Viec SVM 8x2 tot nhat la ket qua thuc nghiem tren test set. Ve ly thuyet, HOG 8x2 tao feature gon, on dinh, it noise hon 6x3. SVM linear phu hop voi feature HOG high-dimensional. Ket hop lai, SVM 8x2 dat accuracy va weighted F1 cao nhat.

### 174. Neu hoi "co the tin vao Optuna khong?", tra loi sao?

Optuna khong dam bao tim global optimum, nhung no search co he thong hon thu tay va dung validation set de danh gia. Em cung khong dung test trong qua trinh search, nen ket qua test van doc lap.

### 175. Neu hoi "neu du lieu thay doi thi model co con dung khong?", tra loi sao?

Neu phan phoi du lieu thay doi nhieu, vi du anh thuc te co nen phuc tap, goc chup khac, bien bao nho/mo, model co the giam do chinh xac. Khi do can retrain voi du lieu moi hoac bo sung augmentation/detector/CNN.

### 176. Neu hoi "tai sao khong dung mot model duy nhat tu dau?", tra loi sao?

Em so sanh nhieu cau hinh de co bang chung thuc nghiem. Ket qua cho thay 8x2 tot hon 6x3 va SVM 8x2 tot nhat, nen viec chon model cuoi co co so thay vi cam tinh.

### 177. Neu hoi "phan training co diem nao can cai tien nhat?", tra loi sao?

Diem can cai tien nhat la them validation/cross-validation chat che hon, them augmentation, luu cache kem hash/path de tranh cache cu, va thu scaler hoac CNN baseline de so sanh.

### 178. Cau tra loi ngan cho "6x3 la gi, 8x2 la gi?"

6x3 la HOG cell 6x6 pixel va block 3x3 cell. 8x2 la HOG cell 8x8 pixel va block 2x2 cell. 6x3 tao 29273 total features, 8x2 tao 8132 total features.

### 179. Cau tra loi ngan cho "vi sao them HSV?"

Vi HOG khong nam mau. Bien bao giao thong co mau dac trung, nen HSV histogram bo sung thong tin mau cho vector HOG.

### 180. Cau tra loi ngan cho "vi sao ket qua cua RF kem SVM mot chut?"

HOG tao feature lien tuc, nhieu chieu, co cau truc phu hop voi margin-based classifier nhu SVM. RF van tot, nhung voi feature HOG high-dimensional, SVM 8x2 tong quat hoa nhe hon trong test set nay.

