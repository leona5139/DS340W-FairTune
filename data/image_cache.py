"""One-time downscaled-image cache (runtime optimization, shared by every run).

PAPILA fundus JPEGs are ~2.5k px; the train pipeline (RandomResizedCrop ->
224) and eval pipeline (Resize 256 -> CenterCrop 224) discard almost all of
that, but re-decode the full file every epoch. We resize once to
`short_side` px (default 512, lossless PNG, antialiased) and reuse it.
Images already no larger than `short_side` (e.g. Harvard-GF's 200x200 PNGs)
are used as-is. Disable for a paired check with FAIRTUNE_IMG_CACHE_SHORT_SIDE=0.
"""
import hashlib
import os

from PIL import Image

_SHORT_SIDE = int(os.environ.get("FAIRTUNE_IMG_CACHE_SHORT_SIDE", "512"))
_ROOT = os.environ.get(
    "FAIRTUNE_IMG_CACHE", os.path.expanduser("~/.cache/fairtune_img_cache")
)


def cached_path(path, short_side=None):
    short_side = _SHORT_SIDE if short_side is None else short_side
    if short_side <= 0:
        return path
    key = hashlib.md5(
        f"{os.path.abspath(path)}|{os.path.getmtime(path)}|{short_side}".encode()
    ).hexdigest()
    out = os.path.join(_ROOT, key + ".png")
    if os.path.exists(out):
        return out
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if min(w, h) <= short_side:
            return path
        scale = short_side / min(w, h)
        im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        os.makedirs(_ROOT, exist_ok=True)
        tmp = out + f".tmp{os.getpid()}"
        im.save(tmp, format="PNG")
        os.replace(tmp, out)
    return out
