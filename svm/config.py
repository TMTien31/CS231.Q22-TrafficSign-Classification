from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SVM_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR_CANDIDATES = [
    SVM_DIR / "models",
    SVM_DIR / "svm_models",
]
MODEL_FILENAMES = [
    "svm_hog_tuned_model.joblib",
    "svm_hog_model.joblib",
]
MODEL_CANDIDATES = [
    model_dir / model_name
    for model_dir in MODEL_DIR_CANDIDATES
    for model_name in MODEL_FILENAMES
]

# Backward-compatible alias for any legacy imports.
MODELS_DIR = MODEL_DIR_CANDIDATES[0]

IMG_SIZE = (64, 64)
CLASSES = ["Cam", "Chidan", "Hieulenh", "Nguyhiem", "Phu"]

REPORT_PATH = SVM_DIR / "svm_classification_report.txt"
