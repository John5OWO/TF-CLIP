# LSVID_best_code_and_weight.zip — Package Structure Analysis

## 1. Package Info

| Item | Value |
|---|---|
| File | `/data1/lgf/LSVID_best_code_and_weight.zip` |
| Size | 374M |
| Type | Zip archive (compression method=store) |
| Extracted to | `/data1/lgf/experiments/verify_lsvid_official_best/official_package/LSVID_best_code_and_weight/` |

## 2. Package Contents

### Core Components

| Component | Present | Files | Notes |
|---|---|---|---|
| `config/` | YES | 2 | `defaults.py`, `__init__.py` |
| `configs/` | YES | 2 | `vit_clipreid.yml` (MARS config w/ LSVID), `vit_clipreid2.yml` |
| `datasets/` | YES | 10 | Full: `set/lsvid.py`, `set/lsvid_cavit.py`, `set/mars.py`, etc. |
| `loss/` | YES | 8 | Full: triplet, softmax, arcface, center, supcontrast |
| `solver/` | YES | 7 | Full: optimizer, scheduler, lr_scheduler |
| `model/` | YES | 5 | `make_model_clipreid.py`, `TAT.py`, `Visual_Prompt.py`, `clip/` (full CLIP) |
| `processor/` | YES | 3 | Stage1 + Stage2 training/inference |
| `utils/` | YES | 12 | Full: metrics, logger, transforms, reranking, etc. |
| `eval_all.py` | YES | 1 | Standard eval entry |
| `eval_rrs.py` | YES | 1 | RRS eval (re-ranking) |
| `train.py` | YES | 1 | Training entry |
| `train_main.py` | YES | 1 | Alternative training entry |
| `logs_lsvid.zip` | YES | 389M | Contains checkpoint + train_log |

### Missing vs Expected

| Component | Status |
|---|---|
| README | NOT in package (in GitHub README) |
| requirements.txt | NOT provided |
| environment.yml | NOT provided |
| LS-VID dedicated config | NOT present (MARS config adapted for LSVID) |

### Checkpoint/Log

| Item | Path |
|---|---|
| Best checkpoint | `logs_lsvid/logs_lsvid/best_model.pth.tar` (single, no splits) |
| Train log | `logs_lsvid/logs_lsvid/train_log.txt` |

## 3. Completeness Verdict

**This is a NEARLY COMPLETE runnable repo.** Much more complete than iLIDS package.

Missing items (minor):
1. No dedicated LS-VID config (the MARS config with `DATASETS.NAMES: ('lsvid')` works)
2. No README or environment file
3. `lsvid.py` has hardcoded root path (`/home/ycy/data/LSVID`) — requires fix for local use

## 4. Key Difference from iLIDS Package

| Feature | iLIDS Package | LS-VID Package |
|---|---|---|
| `model/clip/` | MISSING | PRESENT |
| `config/` | MISSING | PRESENT |
| `configs/` | MISSING | PRESENT (2 files) |
| `datasets/` | MISSING | PRESENT (10 files) |
| `loss/` | MISSING | PRESENT (8 files) |
| `solver/` | MISSING | PRESENT (7 files) |
| Architecture | prompts_generator + SAT (pre-release) | prompts_generator + SAT (pre-release) |
| Checkpoint splits | 10 splits | Single checkpoint |
| Train log | Per-split (10 files) | Single file |

## 5. Required Patches for Local Eval

1. **`datasets/set/lsvid.py` line 11**: Change `self._root = '/home/ycy/data/LSVID'` to `self._root = root if root is not None else '/home/ycy/data/LSVID'`
2. **`configs/official_lsvid_eval.yml`**: Create eval config with local ROOT_DIR
3. **`model/make_model_clipreid.py` line 263**: Add cv_embed batch expansion fix for single-tracklet batches
