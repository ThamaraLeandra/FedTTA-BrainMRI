import argparse, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT

def _to_cv2(img_pil):
    import numpy as np
    if img_pil.mode == "L":
        return np.array(img_pil) 
    arr = np.array(img_pil)
    return arr[:, :, ::-1].copy()

def _to_pil(img_cv):
    import numpy as np
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv, mode="L")
    arr = img_cv[:, :, ::-1]
    return Image.fromarray(arr, mode="RGB")

def apply_denoise(img_pil, method="none", gaussian_ksize=3, bilateral_d=5, bilateral_sigma=50):
    if method == "none":
        return img_pil
    if not _HAS_CV2:
        print("OpenCV não disponível; ignorando denoise.", file=sys.stderr)
        return img_pil
    img_cv = _to_cv2(img_pil)
    if method == "gaussian":
        k = max(3, int(gaussian_ksize) | 1)
        den = cv2.GaussianBlur(img_cv, (k, k), 0)
        return _to_pil(den)
    elif method == "bilateral":
        d = int(bilateral_d)
        sigma = int(bilateral_sigma)
        den = cv2.bilateralFilter(img_cv, d, sigma, sigma)
        return _to_pil(den)
    else:
        return img_pil

def apply_clahe(img_pil, clip_limit=2.0, tile_grid_size=8, grayscale=False):
    if not _HAS_CV2:
        print("OpenCV não disponível; ignorando CLAHE.", file=sys.stderr)
        return img_pil
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    img_cv = _to_cv2(img_pil)
    if grayscale or img_pil.mode == "L":
        cl = clahe.apply(img_cv)
        return _to_pil(cl)
    else:
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        Lc = clahe.apply(L)
        labc = cv2.merge((Lc, A, B))
        out = cv2.cvtColor(labc, cv2.COLOR_LAB2BGR)
        return _to_pil(out)

def resize_keep_aspect(img, size, mode="stretch", pad_color=(0,0,0)):
    w, h = img.size
    if mode == "stretch":
        return img.resize((size, size), Image.BILINEAR)

    if mode == "pad":
        ratio = min(size / w, size / h)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        img_resized = img.resize((nw, nh), Image.BILINEAR)
        new_img = Image.new("RGB" if img.mode != "L" else "L", (size, size), pad_color if img.mode != "L" else 0)
        x = (size - nw) // 2
        y = (size - nh) // 2
        new_img.paste(img_resized, (x, y))
        return new_img

    if mode == "crop":
        ratio = max(size / w, size / h)  
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        img_resized = img.resize((nw, nh), Image.BILINEAR)
        left = (nw - size) // 2
        top = (nh - size) // 2
        return img_resized.crop((left, top, left + size, top + size))

    return img.resize((size, size), Image.BILINEAR)

def ensure_mode(img, grayscale=False, force_rgb=True):
    if grayscale:
        img = img.convert("L")
        if force_rgb:
            img = Image.merge("RGB", (img, img, img))
        return img
    else:
        return img.convert("RGB")

def process_one(
    src: Path, dst: Path, img_size: int,
    keep_aspect: str, pad_color,
    grayscale: bool, force_rgb: bool,
    denoise: str, gaussian_ksize: int, bilateral_d: int, bilateral_sigma: int,
    clahe: bool, clahe_clip: float, clahe_grid: int,
    out_format: str, jpg_quality: int
) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = ensure_mode(im, grayscale=grayscale, force_rgb=force_rgb)

            if denoise != "none":
                im = apply_denoise(im, method=denoise, gaussian_ksize=gaussian_ksize,
                                   bilateral_d=bilateral_d, bilateral_sigma=bilateral_sigma)

            if clahe:
                im = apply_clahe(im, clip_limit=clahe_clip, tile_grid_size=clahe_grid,
                                 grayscale=(grayscale and not force_rgb))

            im = resize_keep_aspect(im, img_size, mode=keep_aspect, pad_color=pad_color)

            suffix = ".jpg" if out_format.lower() == "jpg" else ".png"
            dst = dst.with_suffix(suffix)
            save_kwargs = {}
            if out_format.lower() == "jpg":
                save_kwargs.update(dict(format="JPEG", quality=int(jpg_quality), optimize=True))
            else:
                save_kwargs.update(dict(format="PNG", optimize=True))
            im.save(dst, **save_kwargs)
        return True
    except Exception as e:
        print(f"[ERRO] {src} -> {e}", file=sys.stderr)
        return False

def normalize_split_name(name: str) -> str:
    n = name.lower()
    if n in {"train", "training"}:  return "Training"
    if n in {"test", "testing"}:    return "Testing" 
    return name

def discover_classes(root: Path):
    """Detecta classes presentes; se não encontrar, usa as padrão."""
    found = []
    for c in CLASSES:
        if (root / c).exists():
            found.append(c)
    if found:
        return found
    if root.exists():
        subs = [p.name for p in root.iterdir() if p.is_dir()]
        if subs:
            return subs
    return CLASSES

def run_split(src_root: Path, split: str, dst_root: Path, out_train_name: str, img_size: int,
              keep_aspect: str, pad_color, grayscale: bool, force_rgb: bool,
              denoise: str, gaussian_ksize: int, bilateral_d: int, bilateral_sigma: int,
              clahe: bool, clahe_clip: float, clahe_grid: int, out_format: str, jpg_quality: int):
    norm = normalize_split_name(split)
    if norm == "Training":
        src_split = src_root / "Training"   # corrigido (aceita Training/training/etc.)
        dst_split = dst_root / out_train_name
    elif norm == "Testing":
        src_split = src_root / "Testing"    # corrigido (aceita Testing/testing/etc.)
        dst_split = dst_root / "Testing"
    else:
        src_split = src_root / split
        dst_split = dst_root / split

    if not src_split.exists():
        print(f"Split não encontrado: {src_split}", file=sys.stderr)
        return

    classes = discover_classes(src_split)
    print(f"[RUN] {src_split}  ->  {dst_split}  (size={img_size}, classes={classes}, keep_aspect={keep_aspect}, out={out_format})")

    tasks = []
    total = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for cls in classes:
            src_cls = src_split / cls
            if not src_cls.exists():
                print(f"Classe ausente no split {split}: {cls}", file=sys.stderr)
                continue
            for p in src_cls.rglob("*"):
                if p.is_file() and is_img(p):
                    rel = p.relative_to(src_split)
                    dst = dst_split / rel
                    tasks.append(ex.submit(
                        process_one, p, dst, img_size,
                        keep_aspect, pad_color,
                        grayscale, force_rgb,
                        denoise, gaussian_ksize, bilateral_d, bilateral_sigma,
                        clahe, clahe_clip, clahe_grid,
                        out_format, jpg_quality
                    ))
                    total += 1
        ok = 0
        for fut in as_completed(tasks):
            ok += 1 if fut.result() else 0
    print(f"{split}: {ok}/{total} imagens processadas -> {dst_split}")

def parse_color(color_str):
    parts = [int(x.strip()) for x in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("pad_color deve ser 'r,g,b'")
    return tuple(max(0, min(255, v)) for v in parts)

def main():
    ap = argparse.ArgumentParser(description="Pré-processamento avançado para Kaggle Brain Tumor")
    ap.add_argument("--src_root", default="dataset_kaggle", help="pasta de origem (onde estão Training/ e Testing/)")
    ap.add_argument("--dst_root", default="dataset_kaggle_preprocessed", help="pasta de destino")
    ap.add_argument("--splits", nargs="+", default=["Training", "Testing"], help="quais splits processar")
    ap.add_argument("--out_train_name", default="Train", help="nome do split de treino na saída (ex.: Train)")
    ap.add_argument("--img_size", type=int, default=224, help="lado alvo (quadrado)")

    ap.add_argument("--grayscale", type=int, default=0, help="1 para converter para tons de cinza")
    ap.add_argument("--force_rgb", type=int, default=1, help="1 para garantir RGB no final (útil para backbones 3 canais)")
    ap.add_argument("--keep_aspect", choices=["stretch", "pad", "crop"], default="stretch", help="método de redimensionamento")
    ap.add_argument("--pad_color", type=str, default="0,0,0", help="cor de padding (r,g,b), usada quando keep_aspect=pad")
    ap.add_argument("--denoise", choices=["none", "gaussian", "bilateral"], default="none", help="filtro de ruído opcional (requer OpenCV)")
    ap.add_argument("--gaussian_ksize", type=int, default=3, help="kernel do gaussiano (ímpar)")
    ap.add_argument("--bilateral_d", type=int, default=5, help="diâmetro do bilateral")
    ap.add_argument("--bilateral_sigma", type=int, default=50, help="sigma do bilateral")
    ap.add_argument("--clahe", type=int, default=0, help="1 para aplicar CLAHE (requer OpenCV)")
    ap.add_argument("--clahe_clip", type=float, default=2.0, help="clipLimit do CLAHE")
    ap.add_argument("--clahe_grid", type=int, default=8, help="tileGridSize do CLAHE (lado)")
    ap.add_argument("--out_format", choices=["jpg", "png"], default="jpg", help="formato de saída")
    ap.add_argument("--jpg_quality", type=int, default=95, help="qualidade JPEG (0-100)")

    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    try:
        pad_color = parse_color(args.pad_color)
    except Exception as e:
        print(f"[ERRO] pad_color inválido: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[START] src={src_root}  dst={dst_root}  splits={args.splits}  size={args.img_size}")
    for sp in args.splits:
        run_split(
            src_root, sp, dst_root, args.out_train_name, args.img_size,
            args.keep_aspect, pad_color,
            bool(args.grayscale), bool(args.force_rgb),
            args.denoise, args.gaussian_ksize, args.bilateral_d, args.bilateral_sigma,
            bool(args.clahe), args.clahe_clip, args.clahe_grid,
            args.out_format, args.jpg_quality
        )
    print("[DONE]")

if __name__ == "__main__":
    main()
