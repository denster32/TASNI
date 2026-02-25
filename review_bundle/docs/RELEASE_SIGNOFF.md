# TASNI Release Signoff Report

**Date:** 2026-02-20
**Scope:** Publication hardening execution signoff
**Mode:** Staged rerun + clean-environment smoke verification

---

## Executed Verification Steps

1. **Baseline snapshot (strict)**
   - Command: `python scripts/generate_release_baseline.py --strict`
   - Output: `output/release/baseline_snapshot.json`
   - Result: PASS

2. **Release manifest regeneration**
   - Command: `python src/tasni/utils/data_manager.py --manifest`
   - Output: `output/release/RELEASE_MANIFEST.json`
   - Result: PASS

3. **Checksum + manifest hash verification**
   - Command: `python scripts/verify_release_checksums.py`
   - Result: PASS (0 failures)

4. **Publication figure regeneration**
   - Command: `python src/tasni/generation/generate_publication_figures.py ... --output reports/figures`
   - Result: PASS with one non-blocking warning
   - Note: fading lightcurve panel generation requires `data/processed/neowise_epochs.parquet`, not present in current workspace.

5. **Focused hardening test suite**
   - Command:
     - `pytest tests/unit/test_periodogram.py`
     - `pytest tests/unit/test_ml_scoring.py`
     - `pytest tests/unit/test_selection_function.py`
     - `pytest tests/unit/test_bayesian_selection.py`
     - `pytest tests/unit/test_release_integrity.py`
     - `pytest tests/unit/test_tier_vetoes.py tests/test_new_vetoes.py`
     - `pytest tests/unit/test_config.py tests/unit/test_imports.py`
   - Aggregate result: PASS (`82 passed, 1 skipped`)

6. **CLI clean-run smoke**
   - Command: `python -m tasni info`
   - Result: PASS

---

## Gate Status

| Gate | Description | Status |
|---|---|---|
| A | Statistical integrity (multiple testing + alias handling in periodograms) | PASS |
| B | ML integrity (no circular proxy-label training path) | PASS |
| C | Governance/compliance (manifest, checksums, external model attribution bundle) | PASS |
| D | Reproducibility (deterministic tests + baseline snapshot + smoke rerun) | PASS (staged) |
| E | Reliability/CI (consolidated workflow + release gates) | PASS |
| F | Publication consistency (metadata/doc path alignment) | PASS |

---

## Signoff Notes

- This signoff is based on a **staged regeneration** of release artifacts available in the workspace.
- Full survey-scale recomputation is intentionally separated from release hardening due compute/runtime constraints and should be executed as an independent production run when required.
- Current release package now has:
  - deterministic seed policy,
  - release manifest with SHA-256 digests and provenance,
  - checksum verification tooling,
  - consolidated CI release gates,
  - updated metadata/documentation consistency.
