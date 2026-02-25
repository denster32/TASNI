# TASNI Claim-to-Artifact Ledger

**Purpose:** Establish a release-control ledger mapping public claims to concrete artifacts, generation paths, and verification gates.

**Owner:** Publication hardening phase
**Baseline snapshot generator:** `scripts/generate_release_baseline.py`

---

## Claim Verification Table

| Claim Category | Public Claim | Artifact(s) | Generation Path | Verification Gate | Current State |
|---|---|---|---|---|---|
| Sample size | Golden sample contains 100 candidates | `data/processed/final/golden_improved.parquet`, `data/processed/final/golden_improved.csv` | `tasni validate` / `src/tasni/pipeline/validation.py` | row-count check in release verification | Pending hardening rerun |
| Fading sources | Three confirmed fading thermal orphans (+ one LMC source discussed separately) | `golden_improved.*`, manuscript tables, figure products | variability + periodogram + manuscript pipeline | statistical-integrity gate + manuscript consistency gate | Pending corrected stats rerun |
| Parallax results | Parallax/distances reported for release subset | `golden_improved_parallax.*`, `output/final/golden_parallax.csv` | `src/tasni/analysis/extract_neowise_parallax.py` + validation merge | schema + consistency checks | Pending end-to-end consistency check |
| Periodicity claims | Significant periodicity reporting after alias controls | periodogram outputs and summary tables | `src/tasni/analysis/periodogram_analysis.py` | multiple-testing + alias gate | Failing until science hardening applied |
| Reproducibility | Third party can reproduce key artifacts | docs + scripts + CI checks | `docs/reproducibility.md`, `docs/REPRODUCIBILITY_QUICKSTART.md`, CI workflows | clean-room reproduction gate | Failing until reproducibility hardening applied |
| Data governance | Public release includes complete manifest + checksums + licensing | release manifest, checksums, external-data license bundle | data manager / release scripts | governance & compliance gate | Partially complete |
| Citation metadata | Repo URL, DOI, and citation metadata are mutually consistent | `README.md`, `CITATION.cff`, `pyproject.toml`, `.zenodo.json` | docs/release metadata | publication-consistency gate | Failing consistency check |

---

## Baseline Snapshot Procedure

1. Generate baseline snapshot:
   ```bash
   python scripts/generate_release_baseline.py --strict
   ```
2. Archive output JSON:
   - `output/release/baseline_snapshot.json`
3. Store hash for traceability:
   ```bash
   sha256sum output/release/baseline_snapshot.json
   ```

---

## Release Decision Rules

- A claim is **publishable** only if all linked artifacts pass their gate checks.
- Any change to methods/statistics invalidates prior claim verification until re-run.
- `docs/PUBLICATION_STATUS.md` must reference this ledger and hard-gate outcomes before stating readiness.
