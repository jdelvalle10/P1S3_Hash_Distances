"""Build a small folder in the exact layout of step 2 (P1S2_Copy_Generator output) for offline tests.

    python3 tests/make_step2_sample.py OUT [N_ORIGINALS]

Synthetic 8-bit RGB images (smooth gradients plus seeded shapes), a lossless PNG reference per original,
a few benign copies, one pasted-rectangle "edit" and one not-applicable cell per original, with the
manifests and run_metadata.json fields that step 3 reads."""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COPY_FIELDS = ["image_id", "half", "cell", "stratum", "family", "op", "level", "status", "reason", "file",
               "sha256", "width", "height", "target_share", "mask_share", "changed_share", "source_ann_id",
               "donor_image_id", "donor_ann_id", "source_share", "scale", "x", "y", "attempts",
               "encoded_bytes", "seed"]
ORIG_FIELDS = ["image_id", "order", "half", "file_name", "width", "height", "license_id", "coco_url",
               "jpg_sha256", "png_sha256", "n_objects", "n_eligible_objects"]
CELLS = [("jpeg_q50", "benign", "compression", "jpeg", 50), ("resize_x0.5", "benign", "scaling", "resize", 0.5),
         ("rotate_5", "benign", "mild_geometric", "rotate", 5), ("splice_a10", "altering", "splice", "splice", 0.1),
         ("removal_a20", "altering", "removal", "removal", 0.2)]


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG", compress_level=6)
    return b.getvalue()


def make(out, n=6, seed=20261029):
    out = Path(out)
    (out / "originals").mkdir(parents=True)
    (out / "copies").mkdir()
    rng = np.random.default_rng(1)
    origs, copies, ok = [], [], 0
    for k in range(n):
        iid = 500000 + k
        h, w = 150 + 10 * k, 200
        y, x = np.mgrid[0:h, 0:w]
        arr = np.stack([(x * 255 / w), (y * 255 / h), ((x + y) * 127 / (w + h)) + 60 * k % 120], -1)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        for _ in range(6):
            x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
            d.ellipse((x0, y0, x0 + int(rng.integers(10, 40)), y0 + int(rng.integers(10, 40))),
                      fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
        a = np.asarray(im)
        data = png(a)
        (out / "originals" / f"{iid}.png").write_bytes(data)
        half = "calibration" if k % 2 == 0 else "test"
        origs.append(dict(image_id=iid, order=k, half=half, file_name=f"{iid:012d}.jpg", width=w, height=h,
                          license_id=1, coco_url="", jpg_sha256="0" * 64,
                          png_sha256=hashlib.sha256(data).hexdigest(), n_objects=1, n_eligible_objects=1))
        cdir = out / "copies" / str(iid)
        cdir.mkdir()
        for cell, stratum, family, op, level in CELLS:
            row = {f: "" for f in COPY_FIELDS}
            row.update(image_id=iid, half=half, cell=cell, stratum=stratum, family=family, op=op, level=level)
            if stratum == "altering":
                row["target_share"] = level
            if op == "removal":
                row.update(status="not_applicable", reason="no object whose dilated mask is within 0.6-1.6 x target")
                copies.append(row)
                continue
            if op == "jpeg":
                b = io.BytesIO()
                Image.fromarray(a).save(b, "JPEG", quality=level)
                c = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
            elif op == "resize":
                c = np.asarray(Image.fromarray(a).resize((round(w * level), round(h * level)), Image.BICUBIC))
            elif op == "rotate":
                c = np.asarray(Image.fromarray(a).rotate(level, resample=Image.BICUBIC, fillcolor=(0, 0, 0)))
            else:                                   # a filled rectangle covering about 10% of the image
                c = a.copy()
                ph, pw = int(h * 0.3162), int(w * 0.3162)
                c[:ph, :pw] = (250, 20, 20)
                row.update(mask_share=ph * pw / (h * w), changed_share=float((c != a).any(axis=2).mean()))
            cd = png(c)
            rel = f"copies/{iid}/{cell}.png"
            (out / rel).write_bytes(cd)
            row.update(status="ok", file=rel, sha256=hashlib.sha256(cd).hexdigest(), width=c.shape[1],
                       height=c.shape[0])
            copies.append(row)
            ok += 1
    for name, fields, rows in (("manifest_originals.csv", ORIG_FIELDS, origs),
                               ("manifest_copies.csv", COPY_FIELDS, copies)):
        with open(out / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
    meta = dict(settings=dict(script_version="2.1.0", seed=seed, limit=None, n_originals=n),
                script_sha256="test", environment={}, summary=dict(copies_ok=ok, not_applicable={}, seconds=0))
    (out / "run_metadata.json").write_text(json.dumps(meta))
    return out


if __name__ == "__main__":
    make(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
