# P1S3 Hash and Distances (step 3)

Hashes every stored original and copy of the Distance-vs-Accuracy experiment with aHash, dHash, pHash and
wHash, and measures the Hamming distances the analysis uses (protocol steps 8–10: hashing, distances,
sanity checks and counts). It reads **only** the output folder of step 2 (`P1S2_Copy_Generator`) and never
regenerates a copy: every PNG's SHA-256 is recomputed against step 2's manifests before it is hashed.

## Where this script sits in the experiment

The 21 steps of the experiment protocol (final version, approved 29 October 2026). Each script reads only the output of the script before it.

```mermaid
flowchart TB
  subgraph A["A. Fix the protocol before any data exists"]
    direction LR
    S1["1. Criteria table from the literature<br/><i>document</i>"]
    S2["2. Pre-register the protocol<br/><i>document</i>"]
    S1 --> S2
  end
  subgraph B["B. Build the dataset"]
    direction LR
    S3["3. Select 2,000 originals<br/>from COCO train + val<br/><b>P1S1</b>"]
    S4["4. Partition<br/>1,000 calibration / 1,000 test<br/><b>P1S2</b>"]
    S5["5. Catalogues<br/>24 benign + 15 altering cells<br/><b>P1S2</b>"]
    S6["6. Generate and store<br/>every copy<br/><b>P1S2</b>"]
    S7["7. Visual QA<br/>contact sheets<br/><b>P1S2</b><br/><i>sheets made by P1S2;<br/>inspection by the researcher</i>"]
    S3 --> S4 --> S5 --> S6 --> S7
  end
  subgraph C["C. Hash and measure"]
    direction LR
    S8["8. Hash: aHash, dHash,<br/>pHash, wHash<br/><b>P1S3</b>"]
    S9["9. Distances to the original<br/>and unrelated pairs<br/><b>P1S3</b>"]
    S10["10. Sanity checks<br/>and counts per cell<br/><b>P1S3</b>"]
    S8 --> S9 --> S10
  end
  subgraph D["D. Phase 1, calibration half"]
    direction LR
    S11["11. Describe distributions"]
    S12["12. Separation P0<br/>Wilcoxon, AUC"]
    S13["13. Thresholds at α and β"]
    S14["14. Uncertainty<br/>bootstrap, beta-binomial"]
    S15["15. Cell check"]
    S11 --> S12 --> S13 --> S14 --> S15
  end
  subgraph E["E. Phase 2, test half"]
    direction LR
    S16["16. Validate thresholds"]
    S17["17. Sweep 2,145<br/>threshold pairs"]
    S18["18. Map: heatmaps,<br/>operating point"]
    S19["19. Break down by edit,<br/>area and cell"]
    S20["20. Compare hashes"]
    S16 --> S17 --> S18 --> S19 --> S20
  end
  subgraph F["F. Report"]
    direction LR
    S21["21. Report per hash"]
  end
  A --> B --> C --> D --> E --> F
  classDef this fill:#0b5cad,stroke:#0b5cad,color:#ffffff,stroke-width:3px
  classDef other fill:#cfe3f7,stroke:#0b5cad,color:#0b2a4a
  classDef manual fill:#eeeeee,stroke:#888888,color:#333333
  classDef pending fill:#ffffff,stroke:#999999,color:#555555,stroke-dasharray:4 3
  class S8,S9,S10 this
  class S3,S4,S5,S6,S7 other
  class S1,S2 manual
  class S11,S12,S13,S14,S15,S16,S17,S18,S19,S20,S21 pending
```

**Dark blue:** this repository (P1S3), protocol steps 8, 9, 10. **Light blue:** the finished scripts before it (P1S1 · P1S1_COCO_Dataset_Retriever, step 3; P1S2 · P1S2_Copy_Generator, steps 4–7). **Grey:** protocol documents, not scripts. **Dashed:** scripts not written yet.

## What it does

1. **Checks step 2's records.** `run_metadata.json` must hold a summary (the run finished); both manifests
   must agree on each original's half; every original must list the same cells; the number of `ok` copies
   must match step 2's summary; every PNG under `originals/` and `copies/` must be listed.
2. **Hashes (step 8).** Each original and each copy with status `ok`: the file is read, its SHA-256 compared
   with the manifest (stop on mismatch or missing file), the verified bytes decoded once to 8-bit RGB, and
   the same image given to the four ImageHash 4.3.2 calls of protocol section 6:

   | Hash | Call |
   | --- | --- |
   | aHash | `average_hash(img, hash_size=8)` |
   | dHash | `dhash(img, hash_size=8)` |
   | pHash | `phash(img, hash_size=8, highfreq_factor=4)` |
   | wHash | `whash(img, hash_size=8, image_scale=128, mode="haar", remove_max_haar_ll=True)` |

   `remove_max_haar_ll=True` is the library default for the protocol's call; it is written out so the
   record shows it (the coarsest Haar LL band is zeroed and the image reconstructed before the 8×8 low band
   is taken).
3. **Distances (step 9).** For each hash, the Hamming distance in bits (0–64) between every `ok` copy and
   its original. Not-applicable cells give no distance; they are counted. **Reference pairs:** within each
   half, the originals (sorted by id) are permuted with a seed derived from the master seed
   (`SHA-256(master seed | "reference_pairs" | half)`), and pair *i* joins the *i*-th and (*i*+1)-th
   original, wrapping around. With 1,000 originals per half this gives 1,000 pairs per half; every original
   is in exactly two pairs, never with itself, and no pair repeats.
4. **Sanity checks (step 10).** Every image is hashed twice and must give identical hashes; every lossless
   reference is re-read from disk with a fresh decode and must be at distance 0 from its first hash, for all
   four hashes. Counts of originals per (half, cell), ok and not applicable, go to `cell_counts.csv`.

If any check fails, the run stops and writes nothing.

## Output

| File | Contents |
| --- | --- |
| `hashes.csv` | One row per stored image: original/cell, file, SHA-256, four hashes as 16 hex digits |
| `distances.csv` | One row per (original, cell, hash): half, cell parameters, target and mask share, changed-pixel share, distance in bits |
| `reference_pairs.csv` | One row per (pair, hash): half, the two originals, distance in bits |
| `cell_counts.csv` | Originals per (half, cell): total, ok, not applicable |
| `run_metadata.json` | Settings, master seed, Python and package versions (including SciPy), SHA-256 of step 2's records and of this script, hash calls, pair rule, sanity results, counts, SHA-256 of every output |

## Usage

    python3 -m venv .venv && source .venv/bin/activate
    python3 -m pip install -r requirements.txt
    python3 -m unittest discover -s tests -v                        # self-test, no network
    python3 hash_distances.py --in ~/DVA_copies --out ~/DVA_hashes  # IN = step-2 output folder

Options: `--workers` (default: all cores; the output does not depend on it), `--expect-scipy VERSION`
(stop unless the installed SciPy is exactly this version; see below), `--allow-unpinned` (run
with package versions other than the protocol pins; the differences are recorded in `run_metadata.json`.
For tests only: the experiment runs in the reference environment). OUT must be a new or empty folder.

## Reproducibility

Pinned versions: numpy 2.5.3, Pillow 12.3.0, ImageHash 4.3.2, PyWavelets 1.8.0 (and SciPy 1.18.1 in
`requirements.txt`, taken from the step-3 pilot environment); the script refuses to run
with others unless `--allow-unpinned` is given. ImageHash's pHash needs SciPy (`scipy.fftpack.dct`).
SciPy is **detected at run time**: the script prints it and records it in `run_metadata.json` (`scipy`:
detected, expected, matches). The computation never changes with the version; only the check adapts. Without
`--expect-scipy` the detected version is accepted (a pilot, or the first run on a machine); with
`--expect-scipy` the run stops on any other version, so a later run can be tied to the version an earlier
run recorded. Resizing inside ImageHash uses Pillow's Lanczos
filter, so exact hashes are guaranteed only in the same software environment.

## Licence

MIT, © 2026 Jose Luis del Valle.
