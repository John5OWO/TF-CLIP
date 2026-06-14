# LS-VID Architecture Comparison

## 1. Code Sources Compared

| Label | Path | Description |
|---|---|---|
| **LS-VID Official** | `official_package/LSVID_best_code_and_weight/model/` | Code from Baidu disk zip |
| **iLIDS Official** | `verify_ilids_official_best/official_run_workspace/model/` | Code from iLIDS Baidu disk zip |
| **GitHub Clean** | `tfclip_official_clean/model/` | Upstream GitHub (`AsuradaYuci/TF-CLIP`) |
| **v0 Repo** | `TF-CLIP/model/` | Current working repo at v0 |

## 2. Module Name Comparison

| Module Role | LS-VID Official | iLIDS Official | GitHub Clean | v0 Repo |
|---|---|---|---|---|
| Temporal module attr | `self.SAT` | `self.SAT` | `self.TMD` | `self.TMD` |
| Temporal module class | `Transformer_SP` | `Transformer_SP` | `Temporal_Memory_Difusion` | `Temporal_Memory_Difusion` |
| Prompt generator attr | `self.prompts_generator` | `self.prompts_generator` | `self.SSP` | `self.SSP` |
| Prompt generator class | `ImageSpecificPrompt` | `ImageSpecificPrompt` | `ImageSpecificPrompt` | `ImageSpecificPrompt` |

## 3. ImageSpecificPrompt Structure

| Component | LS-VID Official | iLIDS Official | GitHub Clean / v0 |
|---|---|---|---|
| `norm` (LayerNorm) | YES | YES | YES |
| `memory_proj` (3-layer) | **NO** | **NO** | **YES** |
| `text_proj` (2-layer) | **NO** | **NO** | **YES** |
| `out_proj` (2-layer) | **NO** | **NO** | **YES** |
| `alpha` (Parameter) | **NO** | **NO** | **YES** |
| `decoder` (2× PromptGeneratorLayer) | YES | YES | YES |

### Forward Pass

**LS-VID / iLIDS Official (simpler):**
```python
visual = self.norm(visual)
for layer in self.decoder:
    text = layer(text, visual)
return text
```

**GitHub Clean / v0 (enhanced):**
```python
visual = self.memory_proj(visual)    # NEW
text = self.text_proj(text)           # NEW
for layer in self.decoder:
    text = layer(text, visual)        # self_attn + cross_attn + mlp
text = self.out_proj(text)            # NEW
return text
```

## 4. PromptGeneratorLayer Structure

| Component | LS-VID Official | iLIDS Official | GitHub Clean / v0 |
|---|---|---|---|
| `self_attn` | **NO** | **NO** | **YES** |
| `cross_attn` | YES | YES | YES |
| `norm1` | YES | YES | YES |
| `norm2` | **NO** | **NO** | **YES** |
| `norm3` | YES | YES | YES |
| `mlp` | YES | YES | YES |

### Forward Pass

**LS-VID / iLIDS Official:**
```python
q = k = v = self.norm1(x)
x = x + self.cross_attn(q, visual, visual)    # NO self_attn
x = x + self.dropout(self.mlp(self.norm3(x)))  # NO norm2
return x
```

**GitHub Clean / v0:**
```python
q = k = v = self.norm1(x)
x = x + self.self_attn(q, k, v)               # NEW
q = self.norm2(x)                              # NEW
x = x + self.cross_attn(q, visual, visual)
x = x + self.dropout(self.mlp(self.norm3(x)))
return x
```

## 5. Temporal Module (SAT vs TMD)

| | LS-VID / iLIDS Official | GitHub Clean / v0 |
|---|---|---|
| Class | `Transformer_SP` | `Temporal_Memory_Difusion` |
| Key prefix | `SAT.resblocks.*` | `TMD.resblocks.*` |
| Internal structure | **Identical** (attn, message_attn, message_fc, ln_1, ln_2, mlp) | **Identical** |
| Key count | 20 | 20 |

→ **Pure rename.** No structural difference in the temporal module.

## 6. Checkpoint Key Analysis

### LS-VID Checkpoint (`best_model.pth.tar`)

| Category | Count |
|---|---|
| Total keys | 225 |
| `prompts_generator.*` keys | 28 |
| `SAT.*` keys | 20 |
| `image_encoder.*` keys | 145 |
| Other (bottleneck, classifier, cv_embed) | 32 |

### Compatibility Matrix

| Checkpoint → Code | Compatible? | Issue |
|---|---|---|
| LS-VID ckpt → LS-VID Official code | **YES** | Exact match after path fix |
| LS-VID ckpt → iLIDS Official code | **YES** | Same architecture (prompts_generator+SAT) |
| LS-VID ckpt → GitHub Clean code | **NO** | SSP vs prompts_generator (28 missing + 57 extra keys) |
| LS-VID ckpt → v0 code | **NO** | Same as GitHub Clean |

### Detailed Key Mismatch (LS-VID ckpt vs v0/clean model)

**Keys in checkpoint but NOT in v0/clean model (48 keys):**
- `SAT.resblocks.0.*` (20 keys)
- `prompts_generator.norm.*` (2 keys)
- `prompts_generator.decoder.{0,1}.cross_attn.*` (10 keys)
- `prompts_generator.decoder.{0,1}.mlp.*` (8 keys)
- `prompts_generator.decoder.{0,1}.norm1.*` (4 keys)
- `prompts_generator.decoder.{0,1}.norm3.*` (4 keys)

**Keys in v0/clean model but NOT in checkpoint (77 keys):**
- `TMD.resblocks.0.*` (20 keys) — renamed from SAT
- `SSP.alpha` (1 key)
- `SSP.norm.*` (2 keys)
- `SSP.memory_proj.*` (6 keys)
- `SSP.text_proj.*` (4 keys)
- `SSP.out_proj.*` (4 keys)
- `SSP.decoder.{0,1}.self_attn.*` (10 keys) — NEW
- `SSP.decoder.{0,1}.cross_attn.*` (10 keys) — renamed
- `SSP.decoder.{0,1}.mlp.*` (8 keys) — renamed
- `SSP.decoder.{0,1}.norm1.*` (4 keys) — renamed
- `SSP.decoder.{0,1}.norm2.*` (4 keys) — NEW
- `SSP.decoder.{0,1}.norm3.*` (4 keys) — renamed

## 7. Key Findings

1. **LS-VID official code = iLIDS official code (architecture-wise)**: Both use `prompts_generator` + `SAT` with the simpler `ImageSpecificPrompt` (no memory_proj, no self_attn, no norm2).

2. **LS-VID and iLIDS checkpoints belong to the same pre-release code version**: 225 keys each, identical structure, different weights.

3. **GitHub Clean / v0 is a LATER, enhanced version**: Adds self_attn, norm2, memory_proj, text_proj, out_proj, alpha to the prompt generator. Renames `prompts_generator`→`SSP` and `SAT`→`TMD`.

4. **LS-VID official package is more complete than iLIDS package**: Has full `model/clip/`, `config/`, `configs/`, `datasets/`, `loss/`, `solver/`. Can run as a standalone workspace with minimal fixes.

5. **Cross-compatibility**: LS-VID checkpoint CAN be loaded by iLIDS official code, and vice versa (same architecture). Neither can be loaded by GitHub Clean / v0.

## 8. Version Timeline (Hypothesized)

```
Pre-release code (prompts_generator + SAT)
  ├── Trained iLIDS (10 splits) → iLIDS_best_code_and_weight.zip
  └── Trained LS-VID (single)   → LSVID_best_code_and_weight.zip
       │
       ▼ (refactoring before GitHub publication)
       │
GitHub release (SSP + TMD, enhanced ImageSpecificPrompt)
  └── AsuradaYuci/TF-CLIP.git (current main)
       │
       ▼ (forked)
       │
John5OWO/TF-CLIP.git (v0 + later modifications)
```
