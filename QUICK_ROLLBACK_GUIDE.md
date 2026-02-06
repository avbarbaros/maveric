# Quick Rollback Guide - HuggingFace Transformers

## TL;DR - One Command Solutions

### Option 1: Downgrade to Last Stable Version (RECOMMENDED)

```bash
pip install transformers==4.46.3 --force-reinstall
```

**Result**: Returns to pytorch_model.bin format (no BaseModelOutputWithPooling issues)

---

### Option 2: Keep Current Fix with Version Pin

```bash
# Already done in requirements.txt!
# Just reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Result**: Won't auto-upgrade to transformers 5.x in future

---

## Verification

Check your transformers version:

```bash
pip show transformers | grep Version
```

**Expected output**:
- ✅ Good: `Version: 4.46.3` (or any 4.x version)
- ⚠️ New: `Version: 5.0.0` (requires compatibility fix - already applied)

---

## What Changed?

| Aspect | Transformers 4.x | Transformers 5.x |
|--------|------------------|------------------|
| Model file | pytorch_model.bin | model.safetensors (via PR #66) |
| File size | 605 MB | 605 MB (same) |
| Return type | `torch.Tensor` | `BaseModelOutputWithPooling` |
| MAVERIC status | ✅ Native support | ✅ Fixed with compatibility layer |

---

## Decision Matrix

### Use 4.46.3 if you want:
- ✅ Maximum stability
- ✅ No surprises
- ✅ Smaller downloads
- ✅ Proven configuration

### Use 5.0.0+ if you want:
- ✅ Latest features
- ✅ Security updates
- ✅ Future-proofing
- ⚠️ Larger initial download

---

## Emergency Rollback (Complete)

If everything breaks and you need to start fresh:

```bash
# 1. Remove everything
pip uninstall transformers torch torchvision -y

# 2. Clear caches
rm -rf ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32

# 3. Install specific versions
pip install torch==2.1.0 torchvision==0.16.0
pip install transformers==4.46.3

# 4. Verify
python -c "from transformers import CLIPModel; print('OK')"
```

---

## Current Code Status

✅ **All 5 locations fixed** to handle both versions:
1. `evaluation.py` - `_create_text_classifier_with_templates()`
2. `evaluation.py` - `evaluate()`
3. `evaluation.py` - `evaluate_detailed()`
4. `training.py` - Training loop
5. `model_customizer.py` - `encode_text()`

✅ **Requirements.txt updated** with version pin:
```
transformers>=4.20.0,<5.0.0
```

---

## Final Recommendation

**For immediate use**: Downgrade to 4.46.3
```bash
pip install transformers==4.46.3 --force-reinstall
```

**For future**: Keep the version pin in requirements.txt (already done)

**For testing**: Current code works with both versions - your choice!

---

## Support

- Full guide: See `TRANSFORMERS_VERSION_GUIDE.md`
- Code changes: See `CLAUDE.md` (February 4, 2026 entry)
- Git history: `git log --oneline | grep -i transform`
