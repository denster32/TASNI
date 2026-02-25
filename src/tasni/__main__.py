#!/usr/bin/env python3
"""
TASNI CLI Entry Point
======================
Run with: python -m tasni --help
       or: poetry run tasni --help

Commands:
  tasni tier1-vetoes    Run Tier1 veto cross-matches (UKIDSS/VHS/CatWISE)
  tasni ml-scoring      Run ML ensemble scoring on features
  tasni validate        Run golden validation filters
  tasni pipeline        Run full or partial pipeline
  tasni figures         Generate publication figures
  tasni info            Show project information
"""

from pathlib import Path

import pandas as pd
from typer import Argument, Exit, Option, Typer

from tasni.pipeline.tier_vetoes import VetoConfig, run_tier_vetoes

app = Typer(
    help="TASNI - Thermal Anomaly Search for Non-communicating Intelligence", add_completion=False
)


@app.command()
def tier1_vetoes(
    input: Path = Argument(
        "data/interim/checkpoints/tier1/orphans.parquet", help="Tier1 input parquet"
    ),
    output: Path = Argument(
        "data/interim/checkpoints/tier1_improved/tier1_vetoes.parquet", help="Output parquet"
    ),
    radius: float = Option(3.0, "--radius", "-r", help="Match radius in arcsec"),
    batch_size: int = Option(50, "--batch-size", "-b", help="Batch size for API queries"),
    workers: int = Option(8, "--workers", "-w", help="Parallel query workers per batch"),
    test: bool = Option(False, "--test", "-t", help="Test mode (100 sources only)"),
):
    """Run Tier1 vetoes (UKIDSS/VHS/CatWISE cross-match)."""
    config = VetoConfig(radius_arcsec=radius, batch_size=batch_size, max_workers=workers)
    try:
        run_tier_vetoes(input, output, config, test)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise Exit(code=1)


@app.command()
def ml_scoring(
    input: Path = Argument("data/processed/features/tier5_features.parquet", help="Features input"),
    output: Path = Argument("data/processed/ml/ranked_tier5.parquet", help="Ranked output"),
    contamination: float = Option(
        0.1, "--contamination", "-c", help="Isolation Forest contamination"
    ),
    label_col: str | None = Option(
        None, "--label-col", help="Optional external binary label column for supervised models"
    ),
    no_blend_existing: bool = Option(
        False,
        "--no-blend-existing",
        help="Disable blending with existing composite score columns",
    ),
    test: bool = Option(False, "--test", "-t", help="Test mode (1000 samples)"),
):
    """Run ML scoring (unsupervised by default, optional supervised labels)."""
    from tasni.pipeline.ml_scoring import ensemble_scores, load_features

    if not input.exists():
        print(f"Error: Input file not found: {input}")
        raise Exit(code=1)

    print(f"Loading features from {input}...")
    df, feature_cols = load_features(input)

    if test:
        df = df.head(1000)
        print("TEST MODE: Limited to 1000 samples")

    try:
        df = ensemble_scores(
            df,
            feature_cols,
            label_col=label_col,
            contamination=contamination,
            blend_with_existing=not no_blend_existing,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        raise Exit(code=1)

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    print("\nML Scoring complete:")
    print(f"  Total candidates: {len(df)}")
    print(f"  Top score: {df['improved_composite_score'].max():.4f}")
    print(f"  Saved to: {output}")


@app.command()
def validate(
    input: Path = Argument("data/processed/ml/ranked_tier5.parquet", help="Ranked input"),
    output_dir: Path = Argument("data/processed/final", help="Output directory"),
    top_n: int = Option(100, "--top-n", "-n", help="Number of golden candidates"),
):
    """Run golden validation filters and select final candidates."""
    from tasni.pipeline.validation import apply_filters

    if not input.exists():
        print(f"Error: Input file not found: {input}")
        raise Exit(code=1)

    print(f"Loading ranked candidates from {input}...")
    df = pd.read_parquet(input)
    if "improved_composite_score" not in df.columns:
        print("Error: input is missing 'improved_composite_score' column")
        raise Exit(code=1)
    if "designation" not in df.columns:
        print("Error: input is missing 'designation' column")
        raise Exit(code=1)
    df = df.sort_values("improved_composite_score", ascending=False)

    # Apply validation filters
    golden, kin, eros, par, bay = apply_filters(df)

    # Limit to top N
    golden = golden.head(top_n)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    golden.to_csv(output_dir / "golden_improved.csv", index=False)
    golden.to_parquet(output_dir / "golden_improved.parquet", index=False)

    kin.to_csv(output_dir / "golden_improved_kinematics.csv", index=False)
    kin.to_parquet(output_dir / "golden_improved_kinematics.parquet", index=False)

    eros.to_csv(output_dir / "golden_improved_erosita.csv", index=False)
    eros.to_parquet(output_dir / "golden_improved_erosita.parquet", index=False)

    par.to_csv(output_dir / "golden_improved_parallax.csv", index=False)
    par.to_parquet(output_dir / "golden_improved_parallax.parquet", index=False)

    bay.to_csv(output_dir / "golden_improved_bayesian.csv", index=False)
    bay.to_parquet(output_dir / "golden_improved_bayesian.parquet", index=False)

    print("\nValidation complete:")
    print(f"  Golden candidates: {len(golden)}")
    print(f"  Kinematics PASS: {len(kin)}")
    print(f"  eROSITA quiet: {len(eros)}")
    print(f"  Parallax positive: {len(par)}")
    print(f"  Bayesian low FP: {len(bay)}")
    print(f"  Top score: {golden['improved_composite_score'].iloc[0]:.4f}")
    print(f"  Output directory: {output_dir}")


@app.command()
def pipeline(
    phase: str = Argument("all", help="Phase: tier1, ml, validate, all"),
    test: bool = Option(False, "--test", "-t", help="Test mode"),
):
    """Run full or partial TASNI pipeline.

    Phases:
    - tier1: Run UKIDSS/VHS/CatWISE vetoes
    - ml: Run ML ensemble scoring
    - validate: Select golden candidates
    - all: Run complete pipeline
    """
    phases_to_run = []

    if phase == "all":
        phases_to_run = ["tier1", "ml", "validate"]
    elif phase in ["tier1", "ml", "validate"]:
        phases_to_run = [phase]
    else:
        print(f"Error: Unknown phase '{phase}'. Use: tier1, ml, validate, or all")
        raise Exit(code=1)

    print(f"TASNI Pipeline - Running phases: {phases_to_run}")
    print("=" * 60)

    if "tier1" in phases_to_run:
        print("\n[Phase 1] Running Tier1 vetoes...")
        tier1_vetoes(
            input=Path("data/interim/checkpoints/tier1/orphans.parquet"),
            output=Path("data/interim/checkpoints/tier1_improved/tier1_vetoes.parquet"),
            test=test,
        )

    if "ml" in phases_to_run:
        print("\n[Phase 2] Running ML scoring...")
        ml_scoring(
            input=Path("data/processed/features/tier5_features.parquet"),
            output=Path("data/processed/ml/ranked_tier5.parquet"),
            test=test,
        )

    if "validate" in phases_to_run:
        print("\n[Phase 3] Running validation...")
        validate(
            input=Path("data/processed/ml/ranked_tier5.parquet"),
            output_dir=Path("data/processed/final"),
        )

    print("\n" + "=" * 60)
    print("Pipeline complete!")


@app.command()
def figures(
    output_dir: Path = Option("reports/figures", "--output", "-o", help="Output directory"),
    format: str = Option("png", "--format", "-f", help="Output format: png, pdf, or both"),
):
    """Generate publication figures."""
    print(f"Generating publication figures in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # This would call the figure generation module
    # For now, indicate the command exists
    print("Figure generation requires running the full analysis pipeline first.")
    print("Run: python src/tasni/generation/generate_publication_figures.py")


@app.command()
def info():
    """Show TASNI project information."""
    import tasni

    print("=" * 60)
    print("TASNI - Thermal Anomaly Search for Non-communicating Intelligence")
    print("=" * 60)
    print(f"Version: {tasni.__version__}")
    print()
    print("Key Discoveries:")
    print("  - 4 fading thermal orphans (205-466 K)")
    print("  - 100 golden candidates for follow-up")
    print("  - Significant periodicity (P=40-400 days)")
    print()
    print("Data Products:")
    print("  - Golden sample: data/processed/final/golden_improved.parquet")
    print("  - Parallax: data/processed/final/golden_improved_parallax.parquet")
    print("  - ML ranked: data/processed/ml/ranked_tier5.parquet")
    print()
    print("Documentation:")
    print("  - README: README.md")
    print("  - Manuscript: tasni_paper_final/manuscript.tex")
    print("  - API reference: docs/api_reference.md")
    print("=" * 60)


if __name__ == "__main__":
    app(prog_name="tasni")
