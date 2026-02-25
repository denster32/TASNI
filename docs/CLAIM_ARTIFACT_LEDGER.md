# TASNI Claim-to-Artifact Ledger

**Purpose:** Establish a release-control ledger mapping public claims to concrete artifacts, generation paths, and verification gates.

**Owner:** Publication hardening phase
**Last updated:** 2026-02-23
**Status:** All 7 claims RESOLVED
**Baseline snapshot generator:** `scripts/generate_release_baseline.py`

---

## Claim Verification Table

| Claim Category | Public Claim | Artifact(s) | Generation Path | Verification Gate | Current State |
|---|---|---|---|---|---|
| Sample size | Golden sample contains 100 candidates | `data/processed/final/golden_improved.parquet`, `data/processed/final/golden_improved.csv` | `tasni validate` / `src/tasni/pipeline/validation.py` | row-count check in release verification | RESOLVED (Session 2: validation.py fixed top-150 to top-100; 100 rows confirmed, checksums verified) |
| Fading sources | Three confirmed fading thermal orphans (+ one LMC source discussed separately) | `golden_improved.*`, manuscript tables, figure products | variability + periodogram + manuscript pipeline | statistical-integrity gate + manuscript consistency gate | RESOLVED (Sessions 5-6: W1-W2 color tension documented, ATMO 2020 SED fitting added, blend MC analysis confirms sources are real, manuscript consistent with data) |
| Parallax results | Parallax/distances reported for release subset | `golden_improved_parallax.*`, `output/final/golden_parallax.csv` | `src/tasni/analysis/extract_neowise_parallax.py` + validation merge | schema + consistency checks | RESOLVED (Session 3: bogus est_parallax_mas removed, real NEOWISE parallax merged; Session 6: injection-recovery validation added; 67 sources >5 mas, 44 SNR>3) |
| Periodicity claims | Significant periodicity reporting after alias controls | periodogram outputs and summary tables | `src/tasni/analysis/periodogram_analysis.py` | multiple-testing + alias gate | RESOLVED (Session 1: 90-180 day periodicities honestly acknowledged as likely NEOWISE cadence aliases in manuscript; no false claims of astrophysical signal remain) |
| Reproducibility | Third party can reproduce key artifacts | docs + scripts + CI checks | `docs/reproducibility.md`, `docs/REPRODUCIBILITY_QUICKSTART.md`, CI workflows | clean-room reproduction gate | RESOLVED (Session 7: Makefile targets fixed, pytest config added, CI passes lint+typecheck+test on Python 3.11/3.12; full docs in place) |
| Data governance | Public release includes complete manifest + checksums + licensing | release manifest, checksums, external-data license bundle | data manager / release scripts | governance & compliance gate | RESOLVED (Session 7: 5 SHA256 checksums added and verified; Session 9: all checksums re-verified by data-agent; Sonora Cholla license bundle present) |
| Citation metadata | Repo URL, DOI, and citation metadata are mutually consistent | `README.md`, `CITATION.cff`, `pyproject.toml`, `.zenodo.json` | docs/release metadata | publication-consistency gate | RESOLVED (Session 9: all files aligned on dpalucki/tasni URL, version 1.0.0, MIT license, ORCID, release date 2026-02-23; DOI placeholder remains for author to fill post-Zenodo deposit) |

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
