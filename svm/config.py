from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SVM_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = SVM_DIR / "models"
MODEL_CANDIDATES = [
    MODELS_DIR / "svm_hog_tuned_model.joblib",
    MODELS_DIR / "svm_hog_model.joblib",
]

IMG_SIZE = (64, 64)
CLASSES = ["Cam", "Chidan", "Hieulenh", "Nguyhiem", "Phu"]

REPORT_PATH = SVM_DIR / "svm_classification_report.txt"
