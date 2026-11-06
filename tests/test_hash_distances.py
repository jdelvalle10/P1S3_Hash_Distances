"""Offline self-test of step 3 (no network): unit tests of the hash calls, distances and reference pairs,
and end-to-end runs on a small folder in step-2 layout, including tampered and incomplete inputs.

    python3 -m unittest discover -s tests -v
"""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "hash_distances.py"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hash_distances as hd          # noqa: E402
from make_step2_sample import make   # noqa: E402

UNPINNED = ["--allow-unpinned"]     # tests also run outside the reference environment; pins are tested below


def run(*args, check=True):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True,
                          check=check)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Units(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.img = Image.fromarray(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
        y, x = np.mgrid[0:300, 0:400]
        cls.smooth = Image.fromarray(np.stack([x % 256, y % 256, (x * y) % 256], -1).astype(np.uint8))

    def test_hash_calls_are_the_protocol_calls(self):
        for im in (self.img, self.smooth):
            h = hd.hash_image(im)
            self.assertEqual(h["ahash"], str(imagehash.average_hash(im, hash_size=8)))
            self.assertEqual(h["dhash"], str(imagehash.dhash(im, hash_size=8)))
            self.assertEqual(h["phash"], str(imagehash.phash(im, hash_size=8, highfreq_factor=4)))
            self.assertEqual(h["whash"], str(imagehash.whash(im, hash_size=8, image_scale=128, mode="haar")))
            self.assertTrue(all(len(v) == 16 for v in h.values()))

    def test_whash_scale_is_fixed_not_image_dependent(self):
        # the library default would hash a 400x300 image at 256 and its 0.5x copy at 128
        big = self.smooth
        small = big.resize((200, 150), Image.BICUBIC)
        self.assertEqual(hd.hash_image(big)["whash"],
                         str(imagehash.whash(big, hash_size=8, image_scale=128, mode="haar")))
        self.assertEqual(hd.hash_image(small)["whash"],
                         str(imagehash.whash(small, hash_size=8, image_scale=128, mode="haar")))
        # the library default follows the image: 256 for the 400x300 image, 128 for its half-size copy
        self.assertEqual(str(imagehash.whash(big)), str(imagehash.whash(big, image_scale=256)))
        self.assertEqual(str(imagehash.whash(small)), str(imagehash.whash(small, image_scale=128)))

    def test_distance_equals_imagehash_subtraction(self):
        a, b = hd.hash_image(self.img), hd.hash_image(self.smooth)
        for h in hd.HASHES:
            lib = imagehash.hex_to_hash(a[h]) - imagehash.hex_to_hash(b[h])
            self.assertEqual(hd.distance_bits(a[h], b[h]), lib)
            self.assertEqual(hd.distance_bits(a[h], a[h]), 0)
        self.assertEqual(hd.distance_bits("0" * 16, "f" * 16), 64)

    def test_reference_pairs_each_original_twice(self):
        for n in (1000, 25, 3):
            ids = list(range(10, 10 + n))
            pairs = hd.reference_pairs(ids, 99)
            self.assertEqual(len(pairs), n)
            self.assertEqual(set(Counter(x for p in pairs for x in p).values()), {2})
            self.assertTrue(all(a != b for a, b in pairs))
            self.assertEqual(len({frozenset(p) for p in pairs}), n)
            self.assertEqual(set(x for p in pairs for x in p), set(ids))

    def test_reference_pairs_seeded(self):
        ids = list(range(100))
        self.assertEqual(hd.reference_pairs(ids, 5), hd.reference_pairs(list(reversed(ids)), 5))
        self.assertNotEqual(hd.reference_pairs(ids, 5), hd.reference_pairs(ids, 6))
        self.assertEqual([set(p) for p in hd.reference_pairs([7, 8], 1)], [{7, 8}])
        self.assertEqual(hd.reference_pairs([7], 1), [])

    def test_seed_derivation_matches_step2(self):
        # values computed with P1S2_Copy_Generator v2.1.0 generate_copies.derive_seed
        self.assertEqual(hd.derive_seed(20261029, "partition"), 1267097490298986660)
        self.assertEqual(hd.derive_seed(20261029, "reference_pairs", "calibration"), 18204712903779422979)

    def test_version_check(self):
        self.assertEqual(hd.check_versions(dict(hd.PINNED)), [])
        found = dict(hd.PINNED, ImageHash="4.3.1")
        self.assertEqual(hd.check_versions(found), [("ImageHash", "4.3.2", "4.3.1")])
        self.assertEqual(hd.PINNED, {"ImageHash": "4.3.2", "PyWavelets": "1.8.0", "numpy": "2.5.3",
                                     "pillow": "12.3.0"})


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.s2 = make(t / "s2", 6)
        cls.a, cls.b = t / "a", t / "b"
        run("--in", cls.s2, "--out", cls.a, "--workers", "3", *UNPINNED)
        run("--in", cls.s2, "--out", cls.b, "--workers", "1", *UNPINNED)
        cls.meta = json.loads((cls.a / "run_metadata.json").read_text())
        cls.copies = rows(cls.s2 / "manifest_copies.csv")
        cls.n_ok = sum(r["status"] == "ok" for r in cls.copies)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def broken(self, mutate):
        """Copy the step-2 sample, apply mutate(folder), run step 3 and return (process, out folder)."""
        t = Path(tempfile.mkdtemp(dir=self.tmp.name))
        s2 = t / "s2"
        shutil.copytree(self.s2, s2)
        mutate(s2)
        out = t / "out"
        return run("--in", s2, "--out", out, "--workers", "2", *UNPINNED, check=False), out

    def test_same_output_with_any_worker_count(self):
        for name in hd.OUTPUTS:
            self.assertEqual((self.a / name).read_bytes(), (self.b / name).read_bytes(), name)

    def test_row_counts(self):
        self.assertEqual(len(rows(self.a / "hashes.csv")), 6 + self.n_ok)
        self.assertEqual(len(rows(self.a / "distances.csv")), 4 * self.n_ok)
        self.assertEqual(len(rows(self.a / "reference_pairs.csv")), 4 * 6)   # 3 per half, 4 hashes

    def test_not_applicable_counted_not_imputed(self):
        d = rows(self.a / "distances.csv")
        self.assertFalse(any(r["cell"] == "removal_a20" for r in d))
        c = {(r["half"], r["cell"]): r for r in rows(self.a / "cell_counts.csv")}
        self.assertEqual(c[("calibration", "removal_a20")]["n_not_applicable"], "3")
        self.assertEqual(c[("test", "jpeg_q50")]["n_ok"], "3")
        self.assertEqual(len(c), 2 * 5)

    def test_distances_recomputed_independently(self):
        f = {"ahash": lambda i: imagehash.average_hash(i, 8), "dhash": lambda i: imagehash.dhash(i, 8),
             "phash": lambda i: imagehash.phash(i, 8, 4), "whash": lambda i: imagehash.whash(i, 8, 128, "haar")}
        for r in rows(self.a / "distances.csv"):
            o = Image.open(self.s2 / "originals" / f"{r['image_id']}.png").convert("RGB")
            c = Image.open(self.s2 / "copies" / r["image_id"] / f"{r['cell']}.png").convert("RGB")
            self.assertEqual(f[r["hash"]](o) - f[r["hash"]](c), int(r["distance_bits"]))

    def test_pairs_stay_in_their_half(self):
        half = {r["image_id"]: r["half"] for r in rows(self.s2 / "manifest_originals.csv")}
        for r in rows(self.a / "reference_pairs.csv"):
            self.assertEqual(half[r["image_id_a"]], r["half"])
            self.assertEqual(half[r["image_id_b"]], r["half"])
            self.assertNotEqual(r["image_id_a"], r["image_id_b"])

    def test_metadata_records(self):
        m = self.meta
        self.assertEqual(m["script_sha256"], sha(SCRIPT))
        self.assertEqual(m["inputs"]["manifest_copies_sha256"], sha(self.s2 / "manifest_copies.csv"))
        self.assertEqual(m["inputs"]["manifest_originals_sha256"], sha(self.s2 / "manifest_originals.csv"))
        self.assertEqual(m["inputs"]["step2_run_metadata_sha256"], sha(self.s2 / "run_metadata.json"))
        self.assertEqual(m["inputs"]["files_verified"], 6 + self.n_ok)
        for name in hd.OUTPUTS:
            self.assertEqual(m["outputs_sha256"][name], sha(self.a / name))
        self.assertTrue(m["sanity"]["reference_reread_distance_zero"]["passed"])
        self.assertTrue(m["sanity"]["hash_twice_identical"]["passed"])
        self.assertEqual(m["settings"]["seed"], 20261029)
        self.assertIn("remove_max_haar_ll=True", m["hash_parameters"]["calls"]["whash"])
        self.assertIn("scipy", m["environment"]["packages"])

    def test_tampered_copy_stops(self):
        def mutate(s2):
            p = next((s2 / "copies").rglob("*.png"))
            b = bytearray(p.read_bytes())
            b[-20] ^= 1
            p.write_bytes(bytes(b))
        p, out = self.broken(mutate)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("SHA-256 differs", p.stderr)
        self.assertFalse(out.exists() and any(out.iterdir()))

    def test_tampered_original_stops(self):
        def mutate(s2):
            p = next((s2 / "originals").glob("*.png"))
            p.write_bytes(p.read_bytes() + b"x")
        p, out = self.broken(mutate)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("SHA-256 differs", p.stderr)

    def test_missing_file_stops(self):
        p, _ = self.broken(lambda s2: next((s2 / "copies").rglob("*.png")).unlink())
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("missing", p.stderr)

    def test_unlisted_file_stops(self):
        p, _ = self.broken(lambda s2: shutil.copy(next((s2 / "originals").glob("*.png")),
                                                  s2 / "copies" / "extra.png"))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("not listed", p.stderr)

    def test_unfinished_step2_stops(self):
        def mutate(s2):
            m = json.loads((s2 / "run_metadata.json").read_text())
            del m["summary"]
            (s2 / "run_metadata.json").write_text(json.dumps(m))
        p, _ = self.broken(mutate)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("did not finish", p.stderr)

    def test_half_mismatch_stops(self):
        def mutate(s2):
            text = (s2 / "manifest_copies.csv").read_text().splitlines()
            text[1] = text[1].replace("calibration", "test", 1)
            (s2 / "manifest_copies.csv").write_text("\n".join(text) + "\n")
        p, _ = self.broken(mutate)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("half differs", p.stderr)

    def test_refuses_non_empty_output(self):
        p = run("--in", self.s2, "--out", self.a, *UNPINNED, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("not empty", p.stderr)


if __name__ == "__main__":
    unittest.main()
