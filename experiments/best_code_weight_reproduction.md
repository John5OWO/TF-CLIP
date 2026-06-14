# Best Code And Weight Reproduction Audit

Date: 2026-06-14  
Main project: `/data1/lgf/TF-CLIP`  
Task: verify whether author-provided best code and weights can reproduce the claimed best TF-CLIP results, prioritizing iLIDS-VID.

No current TF-CLIP source code was modified. No training was started. No sudo or permission escalation was used.

## 1. Located Packages

Author-provided best archives found under `/data1/lgf`:

| Archive | Size | Status |
|---|---:|---|
| `/data1/lgf/iLIDS_best_code_and_weight.zip` | 3.6G | prioritized for this audit |
| `/data1/lgf/MARS_best_code_and_weight.zip` | 381M | found, not evaluated |
| `/data1/lgf/LSVID_best_code_and_weight.zip` | 374M | found, not evaluated |

An existing isolated verification directory was also found:

`/data1/lgf/experiments/verify_ilids_official_best`

Because this directory already contains an extracted official package, isolated workspaces, scripts, and logs, the package was not re-extracted or overwritten.

## 2. Official iLIDS Package

| Item | Value |
|---|---|
| Archive | `/data1/lgf/iLIDS_best_code_and_weight.zip` |
| Extracted official package | `/data1/lgf/experiments/verify_ilids_official_best/official_package/iLIDS_best_code_and_weight` |
| Inner archive | `logs_ilids.zip` |
| Official checkpoints | `logs_ilids/split0..split9/best_model.pth.tar` |
| Official logs | `logs_ilids/split0..split9/train_log.txt` |
| Eval entry in package | `eval_all.py` |
| Train entries in package | `train.py`, `train_main.py` |

Package completeness:

- Present: `model/`, `processor/`, `utils/`, `train.py`, `train_main.py`, `eval_all.py`, `logs_ilids.zip`.
- Missing: `README`, `requirements`, `environment`, `config/`, `configs/`, `datasets/`, `loss/`, `solver/`.

This means the official zip is not a standalone runnable repository. Exact evaluation requires reconstructing missing pieces from another TF-CLIP codebase, which introduces protocol/code drift risk.

## 3. Dataset And Config Information

Official package notes recovered from logs:

| Item | Value |
|---|---|
| Dataset | iLIDS-VID |
| Dataset name | `ilidsvidsequence` |
| Official original path in logs | `/18640539002/dataset_cc/ilidsvidsequence` |
| Local path used in verification | `/data1/lgf/TF-CLIP/data/iLIDS-VID` |
| Splits | 10 fixed iLIDS splits, `split0` to `split9` |
| Query/gallery protocol | query cam 0, gallery cam 1 |
| `SEQ_LEN` | 8 |
| `SEQ_SRD` | 4 |
| `TEST.IMS_PER_BATCH` | 1 |
| Feature norm | enabled |

Official iLIDS training config reconstructed from logs:

| Field | Official iLIDS best package | Current TF-CLIP iLIDS config |
|---|---|---|
| `SOLVER.STAGE2.MAX_EPOCHS` | 60 | 80 |
| `SOLVER.STAGE2.LARGE_FC_LR` | False | True |
| `SOLVER.STAGE2.LARGE_Prompt_LR` | False | True |
| `SOLVER.STAGE2.STEPS` | `[30, 50]` | `[30, 50, 70]` |
| Evaluation protocol | 10 split average | local baseline used split 0 only |

## 4. Environment Check

Commands were run in the normal, non-escalated user environment.

| Check | Result |
|---|---|
| `which python` | `/data1/lgf/miniconda3/bin/python` |
| system Python | `Python 3.13.13` |
| `which nvidia-smi` | `/usr/bin/nvidia-smi` |
| `nvidia-smi` | OK, 4 x RTX 4090 visible and idle |
| tfclip Python | `Python 3.8.20` |
| tfclip torch | `2.0.1+cu118` |
| `torch.cuda.is_available()` | `False` |
| torch warning | `Can't initialize NVML` |

Because CUDA is not available inside the allowed non-escalated execution environment, no new evaluation run was started in this audit. This follows the explicit constraint: if CUDA is unavailable, stop rather than using sudo or permission escalation.

## 5. Existing Isolated Verification Logs

Existing verification scripts:

| Script | Purpose |
|---|---|
| `/data1/lgf/experiments/verify_ilids_official_best/run_official_eval_ilids.sh` | evaluate 10 official split checkpoints in reconstructed official workspace |
| `/data1/lgf/experiments/verify_ilids_official_best/run_current_repo_eval_ilids.sh` | evaluate current repo split0 baseline checkpoint |
| `/data1/lgf/experiments/verify_ilids_official_best/run_current_code_official_ckpt_eval_ilids.sh` | attempt official checkpoint in current code |

Official evaluation command shape from the script:

```bash
CUDA_VISIBLE_DEVICES=0 /data1/lgf/miniconda3/envs/tfclip/bin/python eval_all.py \
  --config_file configs/official_ilids_eval.yml \
  DATASETS.SPLIT <0..9> \
  TEST.WEIGHT /data1/lgf/experiments/verify_ilids_official_best/official_package/iLIDS_best_code_and_weight/logs_ilids/split<k>/best_model.pth.tar \
  OUTPUT_DIR /data1/lgf/experiments/verify_ilids_official_best/logs/official_eval_split<k>
```

The reconstructed eval config is:

`/data1/lgf/experiments/verify_ilids_official_best/official_run_workspace/configs/official_ilids_eval.yml`

Minimal verification-only adaptations in the isolated workspace:

- CLIP path changed from `/18640539002/dataset_cc/Pretrain-models/ViT-B-16.pt` to `/data1/lgf/.cache/clip/ViT-B-16.pt`.
- iLIDS seq collate adjusted to return iterable `img_paths`, preventing the official inference logger from crashing.

These changes were isolated outside the current TF-CLIP project.

## 6. Results

| Codebase | Dataset | Weight | mAP | Rank-1 | Rank-5 | Rank-10 | Claimed Result | Reproduced? | Notes |
| -------- | ------- | ------ | --: | -----: | -----: | ------: | -------------- | ----------- | ----- |
| Official package training logs | iLIDS-VID, 10 splits | `logs_ilids/split*/best_model.pth.tar` | 96.52 | 94.47 | 99.14 | 99.93 | Paper Rank-1 94.5 / Rank-5 99.1 | Yes, from included logs | Strongest evidence because these are author-provided logs for each split. |
| Rebuilt official eval workspace | iLIDS-VID, 10 splits | official split checkpoints | 86.61 | 89.17 | 93.29 | 95.21 | Paper Rank-1 94.5 / Rank-5 99.1 | No | Official zip is partial; missing dataset/config/loss/solver/eval details prevent exact reconstruction. |
| Current TF-CLIP repo | iLIDS-VID, split0 | `logs/baseline_ilidsvid_20260613_025719/best_model.pth.tar` | 76.9 | 81.2 | 86.7 | 89.0 | Paper Rank-1 94.5 / Rank-5 99.1 | No | Matches current local baseline, but not official 10-split protocol. |
| Current TF-CLIP code + official split0 checkpoint | iLIDS-VID, split0 | official `split0/best_model.pth.tar` | n/a | n/a | n/a | n/a | n/a | No | Failed on checkpoint key mismatch: `prompts_generator.norm.weight`. |

Official split best records parsed from author logs:

| Split | Best Epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---:|---:|---:|---:|---:|---:|
| 0 | 28 | 95.2 | 92.7 | 100.0 | 100.0 |
| 1 | 14 | 97.3 | 96.0 | 98.7 | 99.3 |
| 2 | 22 | 97.5 | 96.0 | 100.0 | 100.0 |
| 3 | 28 | 94.8 | 92.0 | 98.7 | 100.0 |
| 4 | 24 | 95.2 | 92.7 | 97.3 | 100.0 |
| 5 | 30 | 97.1 | 95.3 | 98.7 | 100.0 |
| 6 | 22 | 95.5 | 93.3 | 98.0 | 100.0 |
| 7 | 48 | 97.6 | 96.0 | 100.0 | 100.0 |
| 8 | 30 | 98.2 | 96.7 | 100.0 | 100.0 |
| 9 | 26 | 96.8 | 94.0 | 100.0 | 100.0 |

Average: mAP 96.52 / Rank-1 94.47 / Rank-5 99.14 / Rank-10 99.93.

## 7. Key Differences Between Official Best Code And Current TF-CLIP

| Area | Official iLIDS best package | Current TF-CLIP repo | Impact |
|---|---|---|---|
| Release completeness | Partial package only | Full runnable repo | Exact official eval cannot be reconstructed from zip alone. |
| Protocol | 10 fixed iLIDS splits | local baseline used split0 | Single split is not comparable to paper. |
| Model temporal block | `SAT = Transformer_SP(...)` | `TMD = Temporal_Memory_Difusion(...)` | Official checkpoint cannot load into current model. |
| Prompt module | `prompts_generator` | `SSP`, plus later QATA/MEMORY branches | Key mismatch and different behavior. |
| Stage2 schedule | 60 epochs, steps `[30,50]` | 80 epochs, steps `[30,50,70]` | Material config drift. |
| LR flags | `LARGE_FC_LR=False`, `LARGE_Prompt_LR=False` | both True in current iLIDS config | Material optimization drift. |
| Eval entry | official `eval_all.py` + official processor | current eval path differs and had iLIDS logging/path issues | Direct comparison is unsafe without fixing protocol. |
| Dataset package | not included | current repo dataset loader | Hidden dataloader/protocol differences remain possible. |

## 8. Why The Current TF-CLIP Project Did Not Reproduce iLIDS

Most likely causes, ordered:

1. Current repo is not the same code as the official iLIDS best package. The official model has `prompts_generator` / `SAT`; current code has `SSP` / `TMD` / QATA / MEMORY changes.
2. Current experiment used split0 only, while author/paper evidence is a 10-split average.
3. Current iLIDS config differs from official iLIDS config: epochs, LR flags, and scheduler steps differ.
4. The official package is incomplete. Rebuilt evaluation with current repo as the missing base runs but does not match official logs, indicating hidden dataset/config/eval differences.
5. Current eval infrastructure had iLIDS-specific fragility: seq collate returned `img_paths=None`, and direct eval logging needed adjustment in the isolated copy.

## 9. Conclusions

1. The author-provided logs and weights strongly support the paper iLIDS result: Rank-1 94.47 / Rank-5 99.14 averaged over 10 splits, essentially matching Rank-1 94.5 / Rank-5 99.1.
2. The author-provided zip is not fully self-contained, so exact independent re-evaluation cannot be guaranteed from the archive alone.
3. Rebuilt evaluation did not reproduce the author logs, despite loading official checkpoints, because the missing official configs/datasets/loss/solver/eval code had to be reconstructed from the current repo.
4. Current `/data1/lgf/TF-CLIP` iLIDS baseline is not comparable to the author/paper result because it uses different code, different config, and split0 only.

## 10. Recommendations

1. Use the official best package as evidence that TF-CLIP can reach the claimed iLIDS result, but do not treat it as a clean standalone runnable baseline.
2. For future synthetic-to-real work, use a clean, explicitly documented baseline workspace instead of the current experimental branch.
3. If iLIDS is needed, restore an iLIDS-specific official-style branch: official model (`SAT`, `prompts_generator`), official stage2 schedule, and 10-split evaluation.
4. Do not force official checkpoints into the current modified model; the checkpoint keys are incompatible.
5. Verify MARS and LS-VID best packages separately before using them as cross-domain baselines.
6. For synthetic-to-real migration, prefer datasets/protocols where the baseline is cleanly reproducible. MARS is currently more trustworthy than iLIDS in the current repo.

## 11. Related Files

- Existing verification root: `/data1/lgf/experiments/verify_ilids_official_best`
- Existing official package report: `/data1/lgf/experiments/verify_ilids_official_best/REPORT.md`
- Official package notes: `/data1/lgf/experiments/verify_ilids_official_best/official_package_notes.md`
- Official/current diff summary: `/data1/lgf/experiments/verify_ilids_official_best/diff_official_vs_current.md`
- Official eval log: `/data1/lgf/experiments/verify_ilids_official_best/logs/official_eval_ilids.log`
- Current repo eval log: `/data1/lgf/experiments/verify_ilids_official_best/logs/current_repo_eval_ilids.log`
- Current code + official checkpoint failure log: `/data1/lgf/experiments/verify_ilids_official_best/logs/current_code_official_ckpt_eval_ilids.log`
