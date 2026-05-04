from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(data_dir, classes, img_exts=IMG_EXTS):
    data_dir = Path(data_dir)
    pairs = []
    for label, cls in enumerate(classes):
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        for path in cls_dir.rglob("*"):
            if path.suffix.lower() in img_exts:
                pairs.append((path, label))
    return pairs
