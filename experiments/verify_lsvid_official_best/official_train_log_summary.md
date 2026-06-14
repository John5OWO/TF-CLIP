# LS-VID Official Train Log Summary

## Source

- **File:** `logs_lsvid/logs_lsvid/train_log.txt`
- **Inside:** `LSVID_best_code_and_weight.zip` → `logs_lsvid.zip`

## Training Configuration

- **Config:** `configs/vit_clipreid.yml`
- **Dataset:** `lsvid`
- **ROOT_DIR (original):** `/18640539002/dataset_cc`
- **Model:** ViT-B-16
- **Stage1:** 120 epochs, lr=0.00035, batch=64
- **Stage2:** 80 epochs, lr=0.000005, batch=16, steps=[30,50,70]
- **SIE_CAMERA:** True
- **SEQ_LEN:** 8

## Official Results (from train_log)

| Metric | Value |
|---|---|
| mAP | 83.8% |
| Rank-1 | 90.4% |
| Rank-5 | 97.1% |
| Rank-10 | 98.0% |
| Rank-20 | 98.6% |

## Notes

1. **Single evaluation only.** Unlike iLIDS (10 splits), LS-VID has a single standard split. The train_log shows one evaluation run.
2. **No best-epoch info** in train_log — the log only shows the final evaluation result.
3. **ROOT_DIR is an original server path** (`/18640539002/dataset_cc`) — needs replacement for local eval.
4. **The train_log timestamp**: 2023-04-15 — this is from the pre-release code version.

## Paper Comparison

The paper result table (assets/result.jpg) reports similar numbers. The train_log mAP 83.8% / R1 90.4% is consistent with the paper's reported LS-VID performance for TF-CLIP.
