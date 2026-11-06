# Copyright (c) 2026 Jose Luis del Valle <joseluisdelvalle@gmail.com>
# Released under the MIT License (see LICENSE).
"""Step 3 of the Distance-vs-Accuracy experiment: hash every stored image and measure distances.

Protocol steps covered: 8 (hash with aHash, dHash, pHash, wHash), 9 (distances to the original and
unrelated reference pairs), 10 (sanity checks; counts per cell and half). See protocol section 6.

Reads ONLY the output folder of step 2 (P1S2_Copy_Generator):
    IN/run_metadata.json        step-2 settings (master seed) and summary (the run must have finished)
    IN/manifest_originals.csv   one row per original: half, SHA-256 of the PNG reference
    IN/manifest_copies.csv      one row per (original, cell): status, parameters, measured area, SHA-256
    IN/originals/<id>.png       lossless 8-bit RGB references (every distance is measured against them)
    IN/copies/<id>/<cell>.png   the stored copies
Every PNG's SHA-256 is recomputed before it is hashed; the run stops on any mismatch, missing file or
unlisted file. Copies are never regenerated.

Writes to OUT (a new folder; the run refuses to write into a folder that already holds results):
    hashes.csv            one row per stored image: four 64-bit hashes as 16 hex digits
    distances.csv         one row per (original, cell, hash) for every copy with status "ok"
    reference_pairs.csv   one row per (reference pair, hash): two different originals of the same half
    cell_counts.csv       originals per (half, cell): ok, not applicable
    run_metadata.json     settings, versions, SHA-256 of the inputs, of this script and of each output,
                          sanity-check results, counts

    python3 hash_distances.py --in ~/DVA_copies --out ~/DVA_hashes
    python3 hash_distances.py --in ~/DVA_copies --out ~/DVA_hashes --expect-scipy 1.18.1   # tie to a recorded SciPy
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------- hash calls (protocol section 6)
HASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4
WHASH_IMAGE_SCALE = 128          # fixed: the library default follows each image's own size
WHASH_MODE = "haar"
WHASH_REMOVE_MAX_HAAR_LL = True  # ImageHash 4.3.2 default for the section-6 call; passed explicitly for the record
HASHES = ("ahash", "dhash", "phash", "whash")
N_BITS = HASH_SIZE * HASH_SIZE

# Versions pinned by the protocol (section 8). SciPy (required by ImageHash's pHash) is not pinned here: it is
# detected at run time and recorded; --expect-scipy ties a run to the version an earlier run recorded.
PINNED = {"ImageHash": "4.3.2", "PyWavelets": "1.8.0", "numpy": "2.5.3", "pillow": "12.3.0"}
RECORDED = ("ImageHash", "PyWavelets", "numpy", "pillow", "scipy")

HASH_CALLS = {
    "ahash": f"average_hash(img, hash_size={HASH_SIZE})",
    "dhash": f"dhash(img, hash_size={HASH_SIZE})",
    "phash": f"phash(img, hash_size={HASH_SIZE}, highfreq_factor={PHASH_HIGHFREQ_FACTOR})",
    "whash": f"whash(img, hash_size={HASH_SIZE}, image_scale={WHASH_IMAGE_SCALE}, mode=\"{WHASH_MODE}\", "
             f"remove_max_haar_ll={WHASH_REMOVE_MAX_HAAR_LL})",
}

HASH_FIELDS = ["image_id", "half", "item", "file", "sha256"] + list(HASHES)
DIST_FIELDS = ["image_id", "half", "cell", "stratum", "family", "op", "level", "target_share", "mask_share",
               "changed_share", "hash", "distance_bits"]
PAIR_FIELDS = ["pair_id", "half", "image_id_a", "image_id_b", "hash", "distance_bits"]
COUNT_FIELDS = ["half", "cell", "stratum", "n_originals", "n_ok", "n_not_applicable"]
OUTPUTS = ("hashes.csv", "distances.csv", "reference_pairs.csv", "cell_counts.csv")


class IntegrityError(RuntimeError):
    """A step-2 file or record does not match what step 2 recorded."""


# ---------------------------------------------------------------- helpers
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def derive_seed(master, *parts):
    """64-bit seed from SHA-256 of the master seed and the parts (same rule as step 2)."""
    text = "|".join(str(p) for p in (master,) + parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def hash_image(img):
    """The four section-6 hashes of one decoded 8-bit RGB image, as 16-digit hex strings."""
    return {
        "ahash": str(imagehash.average_hash(img, hash_size=HASH_SIZE)),
        "dhash": str(imagehash.dhash(img, hash_size=HASH_SIZE)),
        "phash": str(imagehash.phash(img, hash_size=HASH_SIZE, highfreq_factor=PHASH_HIGHFREQ_FACTOR)),
        "whash": str(imagehash.whash(img, hash_size=HASH_SIZE, image_scale=WHASH_IMAGE_SCALE, mode=WHASH_MODE,
                                     remove_max_haar_ll=WHASH_REMOVE_MAX_HAAR_LL)),
    }


def distance_bits(hex_a, hex_b):
    """Hamming distance in bits (0-64) between two hashes stored as hex."""
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def reference_pairs(image_ids, seed):
    """Unrelated reference pairs for one half: a seeded random cycle through the originals.

    Pair i joins the i-th and (i+1)-th original of a seeded permutation, wrapping around, so with n >= 3
    originals there are n pairs, every original appears in exactly two, no original is paired with itself
    and no pair repeats. For an even n this is the union of two disjoint perfect matchings."""
    ids = sorted(image_ids)
    n = len(ids)
    if n < 2:
        return []
    perm = [ids[i] for i in np.random.default_rng(seed).permutation(n)]
    if n == 2:
        return [(perm[0], perm[1])]
    return [(perm[i], perm[(i + 1) % n]) for i in range(n)]


def check_versions(found, pinned=PINNED):
    """Return the list of (package, pinned, found) that differ from the protocol pins."""
    return [(k, v, found.get(k)) for k, v in pinned.items() if found.get(k) != v]


def check_scipy(found, expected=None):
    """SciPy is detected at run time, never assumed. Returns a record for run_metadata.json.

    The hash computation itself never changes with the SciPy version: the same section-6 calls run in every
    environment. What adapts is the check: with no --expect-scipy the detected version is accepted and
    recorded (pilot, first run on a machine); with --expect-scipy the run stops if the detected version
    differs, so later runs are tied to the version the earlier run recorded."""
    if found is None:
        raise SystemExit("Error: SciPy is not installed; ImageHash's pHash needs it (pip install scipy).")
    return dict(detected=found, expected=expected, matches=None if expected is None else found == expected)


def package_versions():
    out = {}
    for name in RECORDED:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment():
    return dict(python=sys.version, implementation=platform.python_implementation(),
                platform=platform.platform(), machine=platform.machine(), packages=package_versions())


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# ---------------------------------------------------------------- reading and checking step 2
def load_step2(in_dir):
    """Read step 2's records and check their internal consistency. Returns (meta, originals, copies_by_id)."""
    for name in ("run_metadata.json", "manifest_originals.csv", "manifest_copies.csv"):
        if not (in_dir / name).is_file():
            raise IntegrityError(f"{in_dir / name} is missing: IN must be the output folder of step 2.")
    meta = json.loads((in_dir / "run_metadata.json").read_text())
    if "summary" not in meta:
        raise IntegrityError("step 2's run_metadata.json has no summary: the step-2 run did not finish.")
    origs = read_csv(in_dir / "manifest_originals.csv")
    copies = read_csv(in_dir / "manifest_copies.csv")
    if not origs:
        raise IntegrityError("manifest_originals.csv lists no originals.")
    ids = [o["image_id"] for o in origs]
    if len(set(ids)) != len(ids):
        raise IntegrityError("manifest_originals.csv lists an original twice.")
    half = {o["image_id"]: o["half"] for o in origs}
    if set(half.values()) - {"calibration", "test"}:
        raise IntegrityError("manifest_originals.csv has a half other than calibration/test.")
    by_id = {i: [] for i in ids}
    for r in copies:
        if r["image_id"] not in by_id:
            raise IntegrityError(f"manifest_copies.csv lists image {r['image_id']}, absent from manifest_originals.csv.")
        if r["half"] != half[r["image_id"]]:
            raise IntegrityError(f"image {r['image_id']}: half differs between the two step-2 manifests.")
        if r["status"] not in ("ok", "not_applicable"):
            raise IntegrityError(f"image {r['image_id']} cell {r['cell']}: unknown status {r['status']!r}.")
        if r["status"] == "ok" and not (r["file"] and r["sha256"]):
            raise IntegrityError(f"image {r['image_id']} cell {r['cell']}: status ok without file or SHA-256.")
        by_id[r["image_id"]].append(r)
    cells = [r["cell"] for r in by_id[ids[0]]]
    if not cells or len(set(cells)) != len(cells):
        raise IntegrityError(f"image {ids[0]}: no cells, or a cell listed twice.")
    for i in ids:
        if [r["cell"] for r in by_id[i]] != cells:
            raise IntegrityError(f"image {i}: its cells differ from the catalogue of the other originals.")
    n_ok = sum(r["status"] == "ok" for r in copies)
    if meta["summary"].get("copies_ok") not in (None, n_ok):
        raise IntegrityError(f"step 2 reported {meta['summary']['copies_ok']} copies but manifest_copies.csv "
                             f"lists {n_ok} with status ok.")
    return meta, origs, by_id


def check_no_unlisted(in_dir, origs, by_id):
    """Every PNG under originals/ and copies/ must be listed in a step-2 manifest."""
    listed = {f"originals/{o['image_id']}.png" for o in origs}
    listed |= {r["file"] for rows in by_id.values() for r in rows if r["status"] == "ok"}
    found = {p.relative_to(in_dir).as_posix() for sub in ("originals", "copies")
             for p in (in_dir / sub).rglob("*.png")}
    extra = sorted(found - listed)
    if extra:
        raise IntegrityError(f"{len(extra)} PNG file(s) not listed in the step-2 manifests, e.g. {extra[0]}.")


# ---------------------------------------------------------------- worker side
_W = {}


def _init_worker(state):
    _W.update(state)


def _read_verified(rel, expected):
    path = _W["in"] / rel
    if not path.is_file():
        raise IntegrityError(f"{rel}: listed by step 2 but missing.")
    data = path.read_bytes()
    if sha256_bytes(data) != expected:
        raise IntegrityError(f"{rel}: SHA-256 differs from the step-2 manifest.")
    return data


def _hash_bytes(data, rel):
    """Decode the verified bytes once to 8-bit RGB and hash them; hash a second time to check determinism."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    h1, h2 = hash_image(img), hash_image(img)
    if h1 != h2:
        raise IntegrityError(f"{rel}: hashing the same image twice gave different hashes.")
    return h1


def process_original(image_id):
    """Hash one original and all its stored copies. Returns (image_id, hash rows)."""
    o = _W["origs"][image_id]
    rel = f"originals/{image_id}.png"
    rows = [dict(image_id=image_id, half=o["half"], item="original", file=rel, sha256=o["png_sha256"],
                 **_hash_bytes(_read_verified(rel, o["png_sha256"]), rel))]
    for r in _W["copies"][image_id]:
        if r["status"] == "ok":
            rows.append(dict(image_id=image_id, half=o["half"], item=r["cell"], file=r["file"], sha256=r["sha256"],
                             **_hash_bytes(_read_verified(r["file"], r["sha256"]), r["file"])))
    return image_id, rows


def reread_original(image_id):
    """Sanity check: re-read the lossless reference from disk with a fresh decode and hash it again."""
    img = Image.open(_W["in"] / "originals" / f"{image_id}.png").convert("RGB")
    return image_id, hash_image(img)


# ---------------------------------------------------------------- main
def run_pool(fn, ids, workers, state, label):
    out = {}
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(state,)) as ex:
        futs = [ex.submit(fn, i) for i in ids]
        for k, f in enumerate(as_completed(futs), 1):
            iid, res = f.result()
            out[iid] = res
            if k % 100 == 0 or k == len(ids):
                print(f"  {label} {k}/{len(ids)}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 3: hash every stored image and measure distances "
                                             "(protocol steps 8-10).")
    ap.add_argument("--in", dest="in_dir", required=True, help="step-2 output folder (manifests, originals/, copies/)")
    ap.add_argument("--out", dest="out_dir", required=True, help="new output folder for hashes and distances")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes (default: all cores; "
                                                                         "the output does not depend on it)")
    ap.add_argument("--expect-scipy", metavar="VERSION",
                    help="stop unless the installed SciPy is exactly this version (default: accept and record "
                         "the version detected at run time)")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="run even if package versions differ from the protocol pins (recorded in run_metadata.json)")
    a = ap.parse_args(argv)
    t0 = time.time()
    in_dir, out = Path(a.in_dir).expanduser().resolve(), Path(a.out_dir).expanduser().resolve()

    env = environment()
    diffs = check_versions(env["packages"])
    if diffs and not a.allow_unpinned:
        raise SystemExit("Error: package versions differ from the protocol pins: "
                         + "; ".join(f"{k} {f} (pinned {v})" for k, v, f in diffs)
                         + ". Use the reference environment, or --allow-unpinned for a test run.")
    scipy_rec = check_scipy(env["packages"].get("scipy"), a.expect_scipy)
    if scipy_rec["matches"] is False and not a.allow_unpinned:
        raise SystemExit(f"Error: SciPy {scipy_rec['detected']} is installed but --expect-scipy "
                         f"{a.expect_scipy} was given.")
    print("Versions: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items())
          + f"; Python {platform.python_version()} on {platform.machine()}"
          + ("" if a.expect_scipy else "  (SciPy detected at run time and recorded)"))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Error: {out} is not empty. Use a new output folder.")
    if out == in_dir or in_dir in out.parents:
        raise SystemExit("Error: OUT must not be inside the step-2 folder.")

    try:
        step2, origs, by_id = load_step2(in_dir)
        check_no_unlisted(in_dir, origs, by_id)
    except IntegrityError as e:
        raise SystemExit(f"Error: {e}")
    seed = step2["settings"]["seed"]
    ids = [o["image_id"] for o in origs]
    orig_by_id = {o["image_id"]: o for o in origs}
    n_ok = sum(r["status"] == "ok" for rows in by_id.values() for r in rows)
    print(f"Step 3: {len(ids)} originals, {n_ok} copies, {len(HASHES)} hashes -> {out}  (workers {a.workers})")

    inputs = dict(step2_run_metadata_sha256=sha256_file(in_dir / "run_metadata.json"),
                  manifest_originals_sha256=sha256_file(in_dir / "manifest_originals.csv"),
                  manifest_copies_sha256=sha256_file(in_dir / "manifest_copies.csv"),
                  step2_script_version=step2["settings"].get("script_version"),
                  step2_script_sha256=step2.get("script_sha256"),
                  step2_environment=step2.get("environment"))

    state = {"in": in_dir, "origs": orig_by_id, "copies": by_id}
    try:
        hashed = run_pool(process_original, ids, a.workers, state, "hashed originals")
        reread = run_pool(reread_original, ids, a.workers, state, "re-read references")
    except IntegrityError as e:
        raise SystemExit(f"Error: {e} Nothing was written.")

    # Sanity check 1: the lossless reference, re-read and re-hashed, is at distance 0 for every original and hash
    reread_bad = [(i, h) for i in ids for h in HASHES if distance_bits(hashed[i][0][h], reread[i][h]) != 0]
    if reread_bad:
        raise SystemExit(f"Error: re-reading the reference of {reread_bad[0][0]} changed its {reread_bad[0][1]}. "
                         "Nothing was written.")

    # Hashes, distances, counts (originals in step-2 order, cells in catalogue order)
    hash_rows, dist_rows, counts = [], [], {}
    for i in ids:
        rows = hashed[i]
        hash_rows += rows
        ref = rows[0]
        by_cell = {r["item"]: r for r in rows[1:]}
        for c in by_id[i]:
            key = (orig_by_id[i]["half"], c["cell"])
            n = counts.setdefault(key, dict(half=key[0], cell=key[1], stratum=c["stratum"], n_originals=0,
                                            n_ok=0, n_not_applicable=0))
            n["n_originals"] += 1
            if c["status"] != "ok":
                n["n_not_applicable"] += 1
                continue
            n["n_ok"] += 1
            for h in HASHES:
                dist_rows.append(dict(image_id=i, half=c["half"], cell=c["cell"], stratum=c["stratum"],
                                      family=c["family"], op=c["op"], level=c["level"],
                                      target_share=c["target_share"], mask_share=c["mask_share"],
                                      changed_share=c["changed_share"], hash=h,
                                      distance_bits=distance_bits(ref[h], by_cell[c["cell"]][h])))
    cells = [c["cell"] for c in by_id[ids[0]]]
    count_rows = [counts[(hf, c)] for hf in ("calibration", "test") for c in cells if (hf, c) in counts]

    # Reference pairs: one seeded cycle per half
    pair_rows, n_pairs = [], {}
    for hf in ("calibration", "test"):
        members = [i for i in ids if orig_by_id[i]["half"] == hf]
        pairs = reference_pairs([int(i) for i in members], derive_seed(seed, "reference_pairs", hf))
        n_pairs[hf] = len(pairs)
        for k, (x, y) in enumerate(pairs):
            for h in HASHES:
                pair_rows.append(dict(pair_id=f"{hf[0]}{k:04d}", half=hf, image_id_a=x, image_id_b=y, hash=h,
                                      distance_bits=distance_bits(hashed[str(x)][0][h], hashed[str(y)][0][h])))

    # Write
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "hashes.csv", HASH_FIELDS, hash_rows)
    write_csv(out / "distances.csv", DIST_FIELDS, dist_rows)
    write_csv(out / "reference_pairs.csv", PAIR_FIELDS, pair_rows)
    write_csv(out / "cell_counts.csv", COUNT_FIELDS, count_rows)

    meta = dict(
        settings=dict(script_version=SCRIPT_VERSION, in_dir=str(in_dir), out_dir=str(out), seed=seed,
                      workers=a.workers, allow_unpinned=a.allow_unpinned),
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        script_sha256=sha256_file(Path(__file__)),
        environment=env, unpinned_packages=[dict(package=k, pinned=v, found=f) for k, v, f in diffs],
        scipy=scipy_rec,
        inputs=dict(**inputs, files_verified=len(hash_rows), originals=len(ids), copies_ok=n_ok,
                    unlisted_files=0, check="SHA-256 of every PNG recomputed against the step-2 manifests"),
        hash_parameters=dict(library="ImageHash", calls=HASH_CALLS, bits=N_BITS,
                             input="file bytes verified, decoded once with Pillow, converted to 8-bit RGB; "
                                   "the same image object is given to all four hashes",
                             hex="str(ImageHash): 16 hex digits", distance="Hamming distance in bits, 0-64"),
        reference_pairs=dict(rule="per half: seeded permutation of the originals sorted by id; pair i = "
                                  "(perm[i], perm[i+1 mod n]); each original in exactly two pairs, no repeats",
                             seed_derivation="derive_seed(master seed, 'reference_pairs', half)",
                             pairs_per_half=n_pairs),
        sanity=dict(reference_reread_distance_zero=dict(passed=True, originals=len(ids), hashes=list(HASHES)),
                    hash_twice_identical=dict(passed=True, images=len(hash_rows))),
        counts=dict(hash_rows=len(hash_rows), distance_rows=len(dist_rows), reference_pair_rows=len(pair_rows),
                    not_applicable_cells=sum(r["n_not_applicable"] for r in count_rows)),
        outputs_sha256={n: sha256_file(out / n) for n in OUTPUTS},
        seconds=round(time.time() - t0, 1))
    tmp = out / "run_metadata.json.part"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out / "run_metadata.json")
    print(f"Done: {len(hash_rows)} images hashed, {len(dist_rows)} distances, {len(pair_rows)} reference-pair "
          f"distances; sanity checks passed; {meta['seconds']} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
