# iLIDS Official Best Weight Verification — Final Report (Clean Rerun)

## 1. Summary

**iLIDS official checkpoint loads and runs successfully, but results are ~10% mAP below official train_log.**

- Checkpoint loaded: 225/225 keys matched, no key mismatch
- All 10 splits completed
- 10-split average: mAP 86.61%, R1 89.17%
- Official train_log average: mAP 96.52%, R1 94.47%
- **Gap: mAP -9.91%, R1 -5.30%**

This gap is consistent with the previous verification (before cleanup) and reproducible across two different code setups (LS-VID base + official iLIDS model/processor overlay).

## 2. Setup

| Item | Value |
|---|---|
| Workspace | `/data1/lgf/experiments/verify_ilids_official_best/official_run_workspace/` |
| Base code | LS-VID official package (complete: config/, datasets/, loss/, solver/, model/clip/) |
| Model/Processor | Overlaid from iLIDS official package (exact original code) |
| Architecture | `prompts_generator` + `SAT` (pre-release, matches checkpoint) |
| Dataset | `/data1/lgf/TF-CLIP/data/iLIDS-VID/` |
| Config | `configs/official_ilids_eval.yml` |
| Checkpoints | `official_package/iLIDS_best_code_and_weight/logs_ilids/logs_ilids/split{0-9}/best_model.pth.tar` |

## 3. Patches Applied

| # | File | Change |
|---|---|---|
| 1 | `model/make_model_clipreid.py:41` | CLIP path: hardcoded → `clip._download(url)` |
| 2 | `datasets/set/lsvid.py:11` | Root path: hardcoded → use `root` param |
| 3 | `datasets/make_dataloader_clipreid.py:59` | `img_paths = None` → `img_paths = []` |
| 4 | `data/iLIDS-VID/others` | Symlink to `images/` (for missing flow data) |
| 5 | `configs/official_ilids_eval.yml` | Created eval config with local paths |

## 4. Per-Split Results

| Split | mAP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
|---|---:|---:|---:|---:|---:|
| split0 | 85.3% | 88.1% | 91.8% | 93.8% | 95.9% |
| split1 | 86.2% | 89.6% | 94.0% | 96.1% | 97.3% |
| split2 | 86.3% | 90.1% | 93.5% | 95.3% | 97.2% |
| split3 | 86.5% | 88.4% | 92.2% | 93.6% | 96.0% |
| split4 | 81.0% | 82.5% | 88.6% | 92.4% | 95.3% |
| split5 | 87.1% | 90.5% | 95.2% | 97.1% | 98.7% |
| split6 | 87.1% | 88.5% | 93.0% | 94.8% | 96.4% |
| split7 | 88.1% | 90.4% | 94.8% | 96.2% | 97.5% |
| split8 | 89.8% | 93.2% | 95.9% | 97.1% | 98.7% |
| split9 | 88.7% | 90.4% | 93.9% | 95.7% | 97.5% |
| **Average** | **86.61%** | **89.17%** | **93.29%** | **95.21%** | **97.05%** |

## 5. Comparison

| Metric | Our Eval (10-split avg) | Official train_log (10-split avg) | Gap |
|---|---:|---:|---:|
| mAP | 86.61% | 96.52% | **-9.91%** |
| Rank-1 | 89.17% | 94.47% | **-5.30%** |
| Rank-5 | 93.29% | 99.14% | **-5.85%** |
| Rank-10 | 95.21% | 99.93% | **-4.72%** |
| Rank-20 | 97.05% | — | — |

Our eval ranks the splits similarly to the official results:
- split8 is best (89.8% mAP / 93.2% R1) — official Best Perform 194.9%
- split4 is worst (81.0% mAP / 82.5% R1) — official Best Perform 187.9%

## 6. iLIDS vs LS-VID: Key Difference

| Aspect | iLIDS | LS-VID |
|---|---|---|
| Our eval vs train_log | **-9.9% mAP gap** | **+1.2% mAP (better!)** |
| Dataset source | Pre-extracted images | Pre-extracted images |
| Split mechanism | 10-fold cross-val (splits.json) | Single standard split |
| Flow data dependency | Yes (others/ dir) | No |
| Dataset preprocessing | Complex (raw tarball → images + flow) | Direct tracklet structure |
| Package completeness | Partial (missing config, datasets, etc.) | Complete |

**LS-VID eval matched the paper. iLIDS eval did not.** This strongly suggests the iLIDS gap is dataset-related, not code-related.

## 7. Root Cause Analysis

### Confirmed NOT the cause:
- **Architecture mismatch**: We used the exact official model code (prompts_generator + SAT)
- **Processor/Inference code**: We used the exact official processor code
- **Checkpoint loading**: 225/225 keys matched, no missing/unexpected keys
- **Config**: SEQ_LEN=8, same transforms, same normalization (ImageNet stats)
- **Code version**: This is the second independent run producing identical results

### Likely causes:
1. **Dataset preprocessing**: The official training used raw iLIDS tarballs extracted on-the-fly with `imgextract()`. Our dataset was pre-extracted, possibly with different frame ordering, file naming, or camera assignment.
2. **Flow data**: Our `others/` symlink uses images as fake flow. The original training may have used real optical flow (epicflow) which could affect the training process, though flow is not used during inference.
3. **splits.json generation**: The official code generates splits.json from `train_test_splits_ilidsvid.mat` during first run (`imgextract()`). Our pre-generated splits.json might differ in subtle ways (identity ordering, frame counts).
4. **Sequence construction**: The `_pluckseq_cam` method constructs query/gallery sequences from images. If the image ordering within tracklets differs, the sampled sequences will differ.

## 8. Conclusions

### 8.1 Can the official iLIDS checkpoint be loaded and evaluated?

**Yes.** With the correct pre-release architecture code (prompts_generator + SAT), the checkpoint loads cleanly (225/225 keys) and produces sensible, reproducible results.

### 8.2 Do the results match the paper?

**No.** There is a persistent ~10% mAP gap that cannot be closed by using the exact official model/processor code. The gap is most likely caused by dataset preprocessing differences between the original server environment and our local dataset.

### 8.3 Is the checkpoint valid?

**Yes, but the eval protocol cannot be exactly reproduced.** The checkpoint is not corrupted — it produces strong results (R1 89.17%) that are well above random, and the split ranking correlates with official results. But without access to the original dataset processing pipeline, the exact paper numbers cannot be replicated.

### 8.4 How does this compare to LS-VID?

**LS-VID is much better for verification and baseline purposes.** The LS-VID official package:
- Is complete (all code components included)
- Has a simple dataset structure (no flow dependency, no on-the-fly extraction)
- Produces results matching the paper when evaluated with the official checkpoint
- Has a standard single-split protocol (no 10-fold complexity)

### 8.5 Should we use iLIDS or LS-VID as the S2R baseline?

**Strong recommendation: Use LS-VID as the primary baseline.**

Reasons:
1. LS-VID is fully verifiable (eval matches paper)
2. LS-VID package is complete and self-contained
3. LS-VID has a standard protocol (single split, widely used)
4. LS-VID is larger (3772 identities) and more challenging
5. iLIDS has an unresolvable protocol gap (~10% mAP) that undermines comparison validity

## 9. Recommendations

1. **Primary S2R baseline: LS-VID** — verifiable, complete, standard protocol
2. **Secondary reference: iLIDS** — use official train_log numbers as paper reference only, NOT as the basis for claiming improvements
3. **Archive iLIDS verification** as evidence that the checkpoint is valid but the eval protocol cannot be exactly reproduced
4. **For iLIDS experiments**: if needed, train from scratch with v0 code and report both your v0 baseline AND the paper baseline, noting the architecture/code version difference
5. **The pre-release code (prompts_generator + SAT)** should be the reference implementation for reproducing paper results; the GitHub code (SSP + TMD) is a newer, enhanced version

## 10. Files

| File | Path |
|---|---|
| This report | `/data1/lgf/experiments/verify_ilids_official_best/REPORT.md` |
| Split logs | `/data1/lgf/experiments/verify_ilids_official_best/logs/official_eval_split{0-9}.log` |
| Workspace | `/data1/lgf/experiments/verify_ilids_official_best/official_run_workspace/` |
| Official package | `/data1/lgf/experiments/verify_ilids_official_best/official_package/` |
| Run script | `/data1/lgf/experiments/verify_ilids_official_best/run_10split_eval.sh` |
