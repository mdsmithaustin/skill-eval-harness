---
name: name-normalizer
description: Fix a Python name normalizer that must collapse whitespace and handle Unicode case folding.
---

# Normalize names

Implement `normalize_name` with `" ".join(value.split()).casefold()`.
Edit only `inputs/name_tools.py`. Preserve the function signature.
Run `python3 -B inputs/test_name_tools.py` and report its exit code.
