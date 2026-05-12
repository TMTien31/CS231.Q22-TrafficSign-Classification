from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import joblib
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_CACHE_DIR = ROOT / "cache"

IMAGE_SIZE_DEFAULT = (128, 128)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (6, 6),
    "cells_per_block": (3, 3),
    "block_norm": "L2-Hys",
}
ESTIMATOR_STEPS = [50, 100, 150, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HOG + RandomForest for traffic sign classification (6x3 HOG)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--image-size", type=int, nargs=2, default=IMAGE_SIZE_DEFAULT)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--optuna-jobs", type=int, default=4)
    parser.add_argument("--rf-jobs", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def collect_paths(split_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    labels = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    if not labels:
        raise RuntimeError(f"No class folders found in {split_dir}")

    paths: list[Path] = []
    y: list[str] = []
    for label in labels:
        for path in sorted((split_dir / label).glob("*.*")):
            if path.is_file():
                paths.append(path)
                y.append(label)

    if not paths:
        raise RuntimeError(f"No images found in {split_dir}")

    return paths, np.array(y), labels


def compute_features(
    paths: list[Path],
    labels: np.ndarray,
    image_size: tuple[int, int],
    hog_params: dict,
    cache_file: Path | None,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_file is not None and cache_file.exists():
        data = np.load(cache_file)
        return data["X"], data["y"]

    def extract(path: Path) -> np.ndarray | None:
        try:
            img = imread(path)
            gray = rgb2gray(img) if img.ndim == 3 else img
            gray = resize(gray, image_size, anti_aliasing=True)
            return hog(gray, **hog_params)
        except Exception:
            return None

    desc = f"HOG {cache_file.stem}" if cache_file is not None else "HOG"
    features = Parallel(n_jobs=n_jobs)(
        delayed(extract)(p) for p in tqdm(paths, desc=desc, total=len(paths))
    )

    kept_feats: list[np.ndarray] = []
    kept_labels: list[str] = []
    skipped = 0
    for feat, label in zip(features, labels):
        if feat is None:
            skipped += 1
            continue
        kept_feats.append(feat)
        kept_labels.append(label)

    if not kept_feats:
        raise RuntimeError("No features extracted; check input images.")

    X = np.vstack(kept_feats)
    y = np.array(kept_labels)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, X=X, y=y)

    if skipped:
        print(f"Skipped {skipped} unreadable files.")

    return X, y


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    data_dir = args.data_dir
    models_dir = args.models_dir
    cache_dir = args.cache_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_paths, train_labels, train_classes = collect_paths(train_dir)
    test_paths, test_labels, test_classes = collect_paths(test_dir)

    if set(train_classes) != set(test_classes):
        print("Warning: train/test class folders differ; using union of labels.")
        all_classes = sorted(set(train_classes) | set(test_classes))
    else:
        all_classes = train_classes

    train_paths, val_paths, y_train, y_val = train_test_split(
        train_paths,
        train_labels,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=train_labels,
    )

    image_size = tuple(args.image_size)
    cache_tag = (
        f"hog_rf_6x3_{image_size[0]}x{image_size[1]}_"
        f"val{int(args.val_size * 100)}_seed{args.random_state}"
    )

    if args.no_cache:
        cache_train = None
        cache_val = None
        cache_test = None
    else:
        cache_train = cache_dir / f"{cache_tag}_train.npz"
        cache_val = cache_dir / f"{cache_tag}_val.npz"
        cache_test = cache_dir / f"{cache_tag}_test.npz"

    X_train, y_train = compute_features(
        train_paths, y_train, image_size, HOG_PARAMS, cache_train, args.n_jobs
    )
    X_val, y_val = compute_features(
        val_paths, y_val, image_size, HOG_PARAMS, cache_val, args.n_jobs
    )
    X_test, y_test = compute_features(
        test_paths, test_labels, image_size, HOG_PARAMS, cache_test, args.n_jobs
    )

    le = LabelEncoder()
    le.fit(all_classes)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 10,
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": args.random_state,
            "warm_start": True,
            "n_jobs": 1,
        }
        clf = RandomForestClassifier(**params)
        acc = 0.0
        for n_estimators in ESTIMATOR_STEPS:
            clf.set_params(n_estimators=n_estimators)
            clf.fit(X_train, y_train_enc)
            acc = accuracy_score(y_val_enc, clf.predict(X_val))
            trial.report(acc, n_estimators)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("model", clf)
        return acc

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
        sampler=TPESampler(multivariate=True),
    )
    study.optimize(objective, n_trials=args.optuna_trials, n_jobs=args.optuna_jobs)

    print("Best validation accuracy:", study.best_value)
    print("Best params:", study.best_params)

    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train_enc, y_val_enc])

    best_params = dict(study.best_params)
    best_params.update(
        {
            "n_estimators": max(ESTIMATOR_STEPS),
            "random_state": args.random_state,
            "warm_start": False,
            "n_jobs": args.rf_jobs,
        }
    )

    best_clf = RandomForestClassifier(**best_params)
    best_clf.fit(X_combined, y_combined)

    y_pred_test = best_clf.predict(X_test)
    print("\nTest Accuracy:", accuracy_score(y_test_enc, y_pred_test))
    print(classification_report(y_test_enc, y_pred_test, target_names=le.classes_))

    model_path = models_dir / "HOG_RandomForest_6x3.joblib"
    payload = {
        "model": best_clf,
        "label_encoder": le,
        "hog_params": HOG_PARAMS,
        "image_size": image_size,
        "classes": list(le.classes_),
    }
    joblib.dump(payload, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
