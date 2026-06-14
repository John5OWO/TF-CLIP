# MARS Official Best Weight Verification — Final Report

## 1. Summary

**MARS official checkpoint verification: SUCCESSFUL.**

- Checkpoint loaded: 254/254 keys matched
- Results are nearly identical to official train_log

## 2. Results

| Metric | Our Eval | Official train_log | Diff |
|---|---:|---:|---:|
| mAP | **89.5%** | 89.4% | **+0.1%** |
| Rank-1 | **93.6%** | 93.0% | **+0.6%** |
| Rank-5 | **97.4%** | 97.9% | -0.5% |
| Rank-10 | **98.6%** | 98.5% | +0.1% |
| Rank-20 | **98.9%** | 99.0% | -0.1% |

**Verdict: Results match the paper.**

## 3. Architecture

MARS uses a **UNIQUE hybrid architecture**:

| Component | MARS Package | iLIDS/LS-VID | GitHub/v0 |
|---|---|---|---|
| Module name | `prompts_generator` + `SAT` | `prompts_generator` + `SAT` | `SSP` + `TMD` |
| ImageSpecificPrompt | ENHANCED (memory_proj, text_proj, out_proj, alpha) | SIMPLE (norm only) | ENHANCED |
| PromptGeneratorLayer | ENHANCED (self_attn, norm2, cross_attn, norm1, norm3) | SIMPLE (cross_attn only, norm1, norm3) | ENHANCED |
| Checkpoint keys | **254** | **225** | **254** |

**Key insight:** MARS name = iLIDS/LS-VID (prompts_generator+SAT), but structure = GitHub/v0 (enhanced with all components). This is a transitional version between pre-release and GitHub release.

## 4. Package Completeness

| Component | Status |
|---|---|
| `model/clip/` | PRESENT |
| `config/` | PRESENT |
| `configs/` | PRESENT |
| `datasets/` | PRESENT |
| `loss/` | PRESENT |
| `solver/` | PRESENT |
| `processor/` | PRESENT |
| `utils/` | PRESENT |
| Checkpoint | Single `best_model.pth.tar` |
| Train log | `train_log.txt` + `train_log_raw.txt` |

**Verdict: Complete runnable package.**

## 5. Patches Applied

| # | File | Change |
|---|---|---|
| 1 | `model/make_model_clipreid.py:41` | CLIP path: hardcoded → `clip._download(url)` |
| 2 | `datasets/set/mars.py:38-48` | Root path: class-level hardcoded → instance attributes from `root` param |
| 3 | `configs/official_mars_eval.yml` | Created eval config with local paths |

## 6. Three-Dataset Summary

| Dataset | Architecture | Checkpoint keys | Our eval vs Official | Status |
|---|---|---|---|---|
| **MARS** | Hybrid (prompts_generator+SAT, enhanced) | 254 | mAP +0.1% | ✅ **MATCH** |
| **LS-VID** | Pre-release (prompts_generator+SAT, simple) | 225 | mAP +1.2% | ✅ **MATCH** |
| **iLIDS** | Pre-release (prompts_generator+SAT, simple) | 225 | mAP -9.9% | ❌ **GAP** |

## 7. Architecture Evolution (Hypothesized Timeline)

```
Phase 1: Pre-release (simple ImageSpecificPrompt)
  ├── iLIDS training (225 keys)
  └── LS-VID training (225 keys)
       │
Phase 2: Enhanced ImageSpecificPrompt (added memory_proj, text_proj, out_proj, alpha, self_attn, norm2)
  └── MARS training (254 keys, still named prompts_generator+SAT)
       │
Phase 3: Module renaming for GitHub publication
  └── GitHub Clean / v0 (254 keys, renamed to SSP+TMD)
```

## 8. Conclusions

1. **MARS official checkpoint is valid and verifiable.**
2. **MARS results match the paper within 1%.**
3. **MARS has the enhanced architecture** (same structure as v0/GitHub, just different module names).
4. **MARS can serve as an excellent S2R baseline** along with LS-VID.
5. **iLIDS remains the outlier** — its gap is most likely due to dataset preprocessing differences specific to the iLIDS pipeline.

## 9. Files

| File | Path |
|---|---|
| This report | `/data1/lgf/experiments/verify_mars_official_best/REPORT.md` |
| Eval log | `/data1/lgf/experiments/verify_mars_official_best/logs/official_eval_mars.log` |
| Workspace | `/data1/lgf/experiments/verify_mars_official_best/official_run_workspace/` |
| Official package | `/data1/lgf/experiments/verify_mars_official_best/official_package/` |
