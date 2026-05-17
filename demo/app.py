"""HOG model demo backend."""

from __future__ import annotations

import base64
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from skimage.feature import hog
from skimage.transform import resize

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
TEMPLATE_DIR = Path(__file__).resolve().parent

MODEL_REGISTRY = {
    "hog_svm_6x3": {
        "name": "HOG SVM 6x3",
        "file": "HOG_SVM_6x3.joblib",
        "type": "SVM",
        "tag": "6x3",
    },
    "hog_svm_8x2": {
        "name": "HOG SVM 8x2",
        "file": "HOG_SVM_8x2.joblib",
        "type": "SVM",
        "tag": "8x2",
    },
    "hog_rf_6x3": {
        "name": "HOG Random Forest 6x3",
        "file": "HOG_RandomForest_6x3.joblib",
        "type": "Random Forest",
        "tag": "6x3",
    },
    "hog_rf_8x2": {
        "name": "HOG Random Forest 8x2",
        "file": "HOG_RandomForest_8x2.joblib",
        "type": "Random Forest",
        "tag": "8x2",
    },
}


@dataclass
class LoadedModel:
    model_id: str
    name: str
    model_type: str
    tag: str
    path: Path
    model: Any
    label_encoder: Any | None
    feature_extractor: str
    hog_params: dict[str, Any]
    hsv_hist_bins: tuple[int, int, int] | None
    image_size: tuple[int, int]
    classes: list[str]


def _ensure_numpy_compat() -> None:
    """Allow joblib files saved with older NumPy import paths to load."""
    if hasattr(np, "_core"):
        return

    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.numeric"] = numpy.core.numeric


def _load_models() -> dict[str, LoadedModel]:
    _ensure_numpy_compat()
    loaded: dict[str, LoadedModel] = {}

    for model_id, meta in MODEL_REGISTRY.items():
        path = MODELS_DIR / meta["file"]
        if not path.exists():
            print(f"[models] Missing: models/{meta['file']}")
            continue

        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model" not in payload:
            print(f"[models] Unsupported payload: models/{meta['file']}")
            continue

        label_encoder = payload.get("label_encoder")
        classes = payload.get("classes")
        if classes is None and label_encoder is not None:
            classes = list(label_encoder.classes_)
        if classes is None:
            classes = [str(c) for c in getattr(payload["model"], "classes_", [])]

        model = payload["model"]
        if hasattr(model, "n_jobs"):
            try:
                model.set_params(n_jobs=1)
            except Exception:
                model.n_jobs = 1

        loaded[model_id] = LoadedModel(
            model_id=model_id,
            name=str(meta["name"]),
            model_type=str(meta["type"]),
            tag=str(meta["tag"]),
            path=path,
            model=model,
            label_encoder=label_encoder,
            feature_extractor=str(payload.get("feature_extractor", "HOG")),
            hog_params=dict(payload["hog_params"]),
            hsv_hist_bins=tuple(payload["hsv_hist_bins"]) if payload.get("hsv_hist_bins") is not None else None,
            image_size=tuple(payload.get("image_size", (128, 128))),
            classes=[str(c) for c in classes],
        )
        print(f"[models] Loaded {meta['name']} from models/{meta['file']}")

    return loaded


def _decode_image(raw_bytes: bytes) -> np.ndarray | None:
    if not raw_bytes:
        return None
    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _encode_image(image_bgr: np.ndarray, ext: str = ".png") -> str:
    ok, buffer = cv2.imencode(ext, image_bgr)
    if not ok:
        return ""
    mime = "image/png" if ext == ".png" else "image/jpeg"
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _encode_gray(gray: np.ndarray) -> str:
    gray_float = gray.astype(np.float32)
    if gray_float.max() <= 1.0:
        gray_float = gray_float * 255.0
    gray_u8 = np.clip(gray_float, 0, 255).astype(np.uint8)
    return _encode_image(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR))


def _enhance_hog_image(hog_image: np.ndarray) -> np.ndarray:
    """Create a high-contrast display image from skimage's sparse HOG render."""
    image = np.asarray(hog_image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image, 0.0)

    if float(image.max()) <= float(image.min()):
        return np.zeros((*image.shape, 3), dtype=np.uint8)

    active = image[image > 0]
    if active.size == 0:
        return np.zeros((*image.shape, 3), dtype=np.uint8)

    low = float(np.percentile(active, 5.0))
    high = float(np.percentile(active, 99.5))
    if high <= low:
        low = 0.0
        high = float(active.max())

    image = np.clip((image - low) / (high - low + 1e-8), 0.0, 1.0)
    image = np.power(image, 0.32)
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_u8 = cv2.dilate(image_u8, np.ones((2, 2), dtype=np.uint8), iterations=1)
    color = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    canvas = np.full((*image_u8.shape, 3), 246, dtype=np.uint8)
    mask = image_u8 > 10
    alpha = (image_u8.astype(np.float32) / 255.0)[..., None]
    blended = (canvas.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha)
    canvas[mask] = blended[mask].astype(np.uint8)
    return canvas


def _extract_hsv_histogram(
    image_rgb: np.ndarray,
    image_size: tuple[int, int],
    hsv_bins: tuple[int, int, int],
) -> np.ndarray:
    height, width = image_size
    rgb_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2HSV)
    h_bins, s_bins, v_bins = hsv_bins

    hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).flatten()

    color_feature = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    color_feature /= color_feature.sum() + 1e-8
    return color_feature


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = exp_values.sum()
    if total == 0:
        return np.zeros_like(exp_values)
    return exp_values / total


def _label_from_model_class(model_class: Any, loaded_model: LoadedModel) -> str:
    if loaded_model.label_encoder is not None:
        try:
            return str(loaded_model.label_encoder.inverse_transform([int(model_class)])[0])
        except Exception:
            pass
    try:
        idx = int(model_class)
        if 0 <= idx < len(loaded_model.classes):
            return loaded_model.classes[idx]
    except Exception:
        pass
    return str(model_class)


def _predict_from_feature(feature: np.ndarray, loaded_model: LoadedModel) -> dict[str, Any]:
    model = loaded_model.model
    X = feature.reshape(1, -1)
    raw_pred = model.predict(X)[0]
    label = _label_from_model_class(raw_pred, loaded_model)
    class_values = list(getattr(model, "classes_", range(len(loaded_model.classes))))

    top_scores: list[dict[str, Any]] = []
    confidence: float | None = None
    score_type = "model score"

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)[0]
            order = np.argsort(probabilities)[::-1]
            confidence = round(float(probabilities[order[0]]) * 100, 2)
            score_type = "probability"
            for idx in order[:5]:
                top_scores.append(
                    {
                        "label": _label_from_model_class(class_values[idx], loaded_model),
                        "score": round(float(probabilities[idx]) * 100, 2),
                    }
                )
        except Exception:
            top_scores = []

    if not top_scores and hasattr(model, "decision_function"):
        try:
            decision = np.asarray(model.decision_function(X)[0], dtype=np.float64)
            if decision.ndim == 0:
                decision = np.array([float(decision)])
            scaled = _softmax(decision)
            order = np.argsort(scaled)[::-1]
            confidence = round(float(scaled[order[0]]) * 100, 2)
            score_type = "softmax-scaled decision score"
            for idx in order[:5]:
                class_value = class_values[idx] if idx < len(class_values) else idx
                top_scores.append(
                    {
                        "label": _label_from_model_class(class_value, loaded_model),
                        "score": round(float(scaled[idx]) * 100, 2),
                    }
                )
        except Exception:
            top_scores = []

    return {
        "label": label,
        "confidence": confidence,
        "score_type": score_type,
        "top_scores": top_scores,
    }


def _run_model_pipeline(image_bgr: np.ndarray, loaded_model: LoadedModel) -> dict[str, Any]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    resized = resize(gray, loaded_model.image_size, anti_aliasing=True)

    hog_feature, hog_image = hog(resized, visualize=True, **loaded_model.hog_params)
    hog_image = _enhance_hog_image(hog_image)
    feature_parts = [hog_feature.astype(np.float32)]
    color_feature_length = 0

    if loaded_model.feature_extractor.upper() == "HOG_HSV":
        if loaded_model.hsv_hist_bins is None:
            raise ValueError(f"{loaded_model.name} expects HOG_HSV features but hsv_hist_bins is missing.")
        color_feature = _extract_hsv_histogram(image_rgb, loaded_model.image_size, loaded_model.hsv_hist_bins)
        color_feature_length = int(color_feature.shape[0])
        feature_parts.append(color_feature)

    feature = np.concatenate(feature_parts)
    prediction = _predict_from_feature(feature, loaded_model)

    return {
        "model_id": loaded_model.model_id,
        "name": loaded_model.name,
        "type": loaded_model.model_type,
        "tag": loaded_model.tag,
        "path": str(loaded_model.path.relative_to(ROOT_DIR)),
        "classes": loaded_model.classes,
        "image_size": list(loaded_model.image_size),
        "feature_extractor": loaded_model.feature_extractor,
        "feature_lengths": {
            "hog": int(hog_feature.shape[0]),
            "hsv": color_feature_length,
            "total": int(feature.shape[0]),
        },
        "hog_params": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in loaded_model.hog_params.items()
        },
        "hsv_hist_bins": list(loaded_model.hsv_hist_bins) if loaded_model.hsv_hist_bins is not None else None,
        "steps": {
            "original": _encode_image(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)),
            "grayscale": _encode_gray(gray),
            "resized": _encode_gray(resized),
            "hog": _encode_image(hog_image),
        },
        "prediction": prediction,
    }


def _model_metadata(loaded_model: LoadedModel) -> dict[str, Any]:
    return {
        "id": loaded_model.model_id,
        "name": loaded_model.name,
        "type": loaded_model.model_type,
        "tag": loaded_model.tag,
        "file": loaded_model.path.name,
        "classes": loaded_model.classes,
        "image_size": list(loaded_model.image_size),
        "feature_extractor": loaded_model.feature_extractor,
        "hsv_hist_bins": list(loaded_model.hsv_hist_bins) if loaded_model.hsv_hist_bins is not None else None,
        "hog_params": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in loaded_model.hog_params.items()
        },
    }


def create_app() -> FastAPI:
    loaded_models = _load_models()
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    app = FastAPI(title="HOG Traffic Sign Demo", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/api/models")
    async def list_models() -> JSONResponse:
        return JSONResponse(
            {
                "models": [_model_metadata(model) for model in loaded_models.values()],
            }
        )

    @app.post("/api/predict")
    async def predict(
        file: UploadFile = File(...),
        models: str = Form(...),
    ) -> JSONResponse:
        requested_ids = [model_id.strip() for model_id in models.split(",") if model_id.strip()]
        if not requested_ids:
            return JSONResponse({"error": "No model was selected."}, status_code=400)

        missing_ids = [model_id for model_id in requested_ids if model_id not in loaded_models]
        if missing_ids:
            return JSONResponse(
                {"error": f"Unknown or unavailable model id(s): {', '.join(missing_ids)}"},
                status_code=400,
            )

        raw_bytes = await file.read()
        image_bgr = _decode_image(raw_bytes)
        if image_bgr is None:
            return JSONResponse({"error": "Could not decode the uploaded image."}, status_code=400)

        results = [_run_model_pipeline(image_bgr, loaded_models[model_id]) for model_id in requested_ids]
        return JSONResponse(
            {
                "filename": file.filename,
                "input": {
                    "width": int(image_bgr.shape[1]),
                    "height": int(image_bgr.shape[0]),
                    "image": _encode_image(image_bgr),
                },
                "results": results,
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    print("Server running at: http://localhost:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=False)
