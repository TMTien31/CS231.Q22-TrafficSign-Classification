import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import joblib
from flask import Flask, redirect, render_template, request
from werkzeug.utils import secure_filename

from demo.pipeline import run_inference
from svm.config import MODEL_CANDIDATES


def ensure_numpy_compat():
    import numpy as np

    if hasattr(np, "_core"):
        return

    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.numeric"] = numpy.core.numeric


def load_model():
    ensure_numpy_compat()
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path), path
    return None, None


def create_app():
    root_dir = Path(__file__).resolve().parents[1]
    template_dir = Path(__file__).resolve().parent
    static_dir = root_dir / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )

    upload_dir = static_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    app.config["UPLOAD_FOLDER"] = str(upload_dir)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    model, model_path = load_model()
    if model_path:
        app.logger.info("Loaded model from %s", model_path)
    else:
        app.logger.warning("Model file not found. Place a model in svm/models/.")

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            if "file" not in request.files:
                return redirect(request.url)

            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)

            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = upload_dir / unique_filename
            file.save(str(file_path))

            if model is None:
                return render_template(
                    "index.html",
                    error="Model chua duoc load. Vui long kiem tra duong dan model.",
                )

            img = cv2.imread(str(file_path))
            if img is None:
                return render_template(
                    "index.html",
                    error="Khong doc duoc file anh. Vui long upload file anh hop le.",
                )

            context = run_inference(img, model)
            return render_template("index.html", **context)

        return render_template("index.html")

    return app


app = create_app()
