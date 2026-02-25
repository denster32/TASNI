#!/usr/bin/env python3
"""
Enhanced Bayesian Population Inference for TASNI

Provides:
1. Hierarchical population models using PyMC
2. Bootstrap confidence intervals for all metrics
3. Proper error propagation with full posteriors
4. Selection function modeling
5. Space density estimation

This replaces the simplified Bayesian model in bayesian_selection.py with
a comprehensive framework for population-level inference.
"""

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

# Bayesian inference
import pymc as pm
import pytensor.tensor as pt

from tasni.core.config import OUTPUT_DIR

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class PopulationParameters:
    """Container for population parameters with uncertainties."""

    space_density: float
    space_density_ci: tuple[float, float]
    temperature_mean: float
    temperature_std: float
    distance_mean: float
    distance_std: float
    selection_completeness: float
    n_simulated: int
    n_observed: int


class BayesianPopulationInference:
    """
    Bayesian inference for brown dwarf population properties.

    Uses hierarchical models to:
    1. Estimate space density with proper uncertainties
    2. Model selection function completeness
    3. Compare observed vs expected distributions
    4. Propagate measurement errors through all analyses
    """

    def __init__(self, random_seed: int = 42, n_samples: int = 2000, n_tune: int = 1000):
        self.random_seed = random_seed
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.trace = None

    def build_space_density_model(
        self,
        distances: np.ndarray,
        distance_errors: np.ndarray | None = None,
        volume: float = 100.0,  # pc^3
        prior_density: float = 1e-3,  # Expected density from literature
    ) -> pm.Model:
        """
        Build hierarchical model for space density estimation.

        Args:
            distances: Distance measurements (pc)
            distance_errors: Distance uncertainties (pc)
            volume: Survey volume (pc^3)
            prior_density: Prior on space density from literature

        Returns:
            PyMC model
        """
        n_objects = len(distances)

        if distance_errors is None:
            distance_errors = np.full(n_objects, 0.1 * distances)

        with pm.Model() as model:
            # Prior on space density (log-normal to ensure positivity)
            log_density = pm.Normal("log_density", mu=np.log(prior_density), sigma=1.0)
            density = pm.Deterministic("density", pt.exp(log_density))

            # Expected number of objects
            expected_n = pm.Deterministic("expected_n", density * volume)

            # Observed number (Poisson likelihood)
            n_obs = pm.Poisson("n_obs", mu=expected_n, observed=n_objects)

            # Hierarchical distance model
            # True distances follow a power-law distribution (d^2 for uniform in volume)
            # But we observe with error
            true_distances = pm.TruncatedNormal(
                "true_distances", mu=distances, sigma=distance_errors, lower=0, shape=n_objects
            )

            # Distance scale parameter
            distance_scale = pm.HalfNormal("distance_scale", sigma=50.0)

            # Likelihood for distances (exponential distribution for uniform in volume)
            pm.Exponential("distance_obs", lam=1.0 / distance_scale, observed=true_distances)

        return model

    def build_temperature_distribution_model(
        self, temperatures: np.ndarray, temperature_errors: np.ndarray | None = None
    ) -> pm.Model:
        """
        Build model for temperature distribution.

        Args:
            temperatures: Temperature measurements (K)
            temperature_errors: Temperature uncertainties (K)

        Returns:
            PyMC model
        """
        n_objects = len(temperatures)

        if temperature_errors is None:
            temperature_errors = np.full(n_objects, 50.0)  # 50K default uncertainty

        with pm.Model() as model:
            # Population mean and std
            mu = pm.Normal("temperature_mean", mu=300, sigma=100)
            sigma = pm.HalfNormal("temperature_std", sigma=100)

            # True temperatures (latent)
            true_temps = pm.TruncatedNormal(
                "true_temperatures",
                mu=mu,
                sigma=sigma,
                lower=100,  # Minimum reasonable temperature
                upper=1000,  # Maximum for brown dwarfs
                shape=n_objects,
            )

            # Observed temperatures with measurement error
            pm.Normal(
                "temperature_obs", mu=true_temps, sigma=temperature_errors, observed=temperatures
            )

            # Fraction below 300K (Y-dwarf threshold)
            frac_cold = pm.Deterministic("frac_below_300k", pt.mean(true_temps < 300))

        return model

    def build_selection_function_model(
        self, magnitudes: np.ndarray, detected: np.ndarray, detection_limit: float = 16.0
    ) -> pm.Model:
        """
        Build model for selection function (detection completeness).

        Args:
            magnitudes: Source magnitudes
            detected: Boolean array of detection status
            detection_limit: Survey magnitude limit

        Returns:
            PyMC model
        """
        n_total = len(magnitudes)
        n_detected = detected.sum()

        with pm.Model() as model:
            # 50% completeness magnitude (with uncertainty)
            m50 = pm.Normal("m50", mu=detection_limit - 0.5, sigma=0.5)

            # Sharpness of completeness curve
            alpha = pm.HalfNormal("alpha", sigma=2.0)

            # Detection probability (logistic function)
            p_detect = pm.Deterministic(
                "p_detect", 1.0 / (1.0 + pt.exp(alpha * (magnitudes - m50)))
            )

            # Likelihood
            pm.Bernoulli("detection", p=p_detect, observed=detected.astype(int))

            # Completeness at different magnitude bins
            completeness_15 = pm.Deterministic(
                "completeness_at_15", 1.0 / (1.0 + pt.exp(alpha * (15.0 - m50)))
            )
            completeness_16 = pm.Deterministic(
                "completeness_at_16", 1.0 / (1.0 + pt.exp(alpha * (16.0 - m50)))
            )

        return model

    def fit_model(self, model: pm.Model, target_accept: float = 0.9) -> az.InferenceData:
        """
        Fit a PyMC model.

        Args:
            model: PyMC model
            target_accept: Target acceptance rate

        Returns:
            ArviZ InferenceData
        """
        log.info(f"Fitting model with {self.n_samples} samples, {self.n_tune} tune...")

        with model:
            trace = pm.sample(
                self.n_samples,
                tune=self.n_tune,
                random_seed=self.random_seed,
                target_accept=target_accept,
                progressbar=True,
                return_inferencedata=True,
            )

        self.trace = trace
        return trace

    def summarize_results(
        self, trace: az.InferenceData | None = None, var_names: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Summarize posterior distributions.

        Args:
            trace: ArviZ InferenceData
            var_names: Variables to summarize

        Returns:
            DataFrame with summary statistics
        """
        if trace is None:
            trace = self.trace

        if trace is None:
            raise ValueError("No trace available. Run fit_model first.")

        summary = az.summary(trace, var_names=var_names, hdi_prob=0.95)
        return summary


class BootstrapAnalyzer:
    """
    Bootstrap analysis for confidence intervals.
    """

    def __init__(self, n_bootstrap: int = 1000, random_seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed

    def bootstrap_mean_ci(
        self, data: np.ndarray, confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """
        Bootstrap confidence interval for mean.

        Args:
            data: Data array
            confidence: Confidence level

        Returns:
            Tuple of (mean, lower_ci, upper_ci)
        """
        np.random.seed(self.random_seed)

        n = len(data)
        bootstrap_means = []

        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return np.mean(data), lower, upper

    def bootstrap_median_ci(
        self, data: np.ndarray, confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """
        Bootstrap confidence interval for median.

        Args:
            data: Data array
            confidence: Confidence level

        Returns:
            Tuple of (median, lower_ci, upper_ci)
        """
        np.random.seed(self.random_seed)

        n = len(data)
        bootstrap_medians = []

        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_medians.append(np.median(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_medians, alpha / 2 * 100)
        upper = np.percentile(bootstrap_medians, (1 - alpha / 2) * 100)

        return np.median(data), lower, upper

    def bootstrap_space_density(
        self, distances: np.ndarray, max_distance: float = 100.0, confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """
        Bootstrap space density estimate with CI.

        Args:
            distances: Distance measurements (pc)
            max_distance: Maximum distance for volume calculation
            confidence: Confidence level

        Returns:
            Tuple of (density, lower_ci, upper_ci) in pc^-3
        """
        np.random.seed(self.random_seed)

        # Volume of sphere
        volume = (4 / 3) * np.pi * max_distance**3

        n = len(distances)
        bootstrap_densities = []

        for _ in range(self.n_bootstrap):
            sample = np.random.choice(distances, size=n, replace=True)
            n_in_volume = (sample <= max_distance).sum()
            density = n_in_volume / volume
            bootstrap_densities.append(density)

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_densities, alpha / 2 * 100)
        upper = np.percentile(bootstrap_densities, (1 - alpha / 2) * 100)

        # Point estimate
        n_observed = (distances <= max_distance).sum()
        density_estimate = n_observed / volume

        return density_estimate, lower, upper

    def bootstrap_completeness(
        self, detected: np.ndarray, confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """
        Bootstrap completeness (detection fraction).

        Args:
            detected: Boolean array of detection status
            confidence: Confidence level

        Returns:
            Tuple of (completeness, lower_ci, upper_ci)
        """
        np.random.seed(self.random_seed)

        n = len(detected)
        bootstrap_completeness = []

        for _ in range(self.n_bootstrap):
            sample = np.random.choice(detected, size=n, replace=True)
            bootstrap_completeness.append(sample.mean())

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_completeness, alpha / 2 * 100)
        upper = np.percentile(bootstrap_completeness, (1 - alpha / 2) * 100)

        return detected.mean(), lower, upper


def run_full_population_analysis(
    candidates: pd.DataFrame, output_dir: Path | None = None
) -> dict[str, Any]:
    """
    Run comprehensive population analysis.

    Args:
        candidates: DataFrame with candidate properties
        output_dir: Directory for output files

    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Bootstrap analyzer
    bootstrap = BootstrapAnalyzer(n_bootstrap=1000)

    # 1. Distance statistics
    if "distance_pc" in candidates.columns or "est_distance_pc" in candidates.columns:
        dist_col = "distance_pc" if "distance_pc" in candidates.columns else "est_distance_pc"
        distances = candidates[dist_col].dropna().values

        if len(distances) > 0:
            mean_dist, mean_lo, mean_hi = bootstrap.bootstrap_mean_ci(distances)
            median_dist, med_lo, med_hi = bootstrap.bootstrap_median_ci(distances)
            density, dens_lo, dens_hi = bootstrap.bootstrap_space_density(
                distances, max_distance=50.0
            )

            results["distance"] = {
                "mean": mean_dist,
                "mean_ci": (mean_lo, mean_hi),
                "median": median_dist,
                "median_ci": (med_lo, med_hi),
                "n_measured": len(distances),
            }
            results["space_density"] = {
                "estimate": density,
                "ci": (dens_lo, dens_hi),
                "volume_radius_pc": 50.0,
            }

            log.info(f"Distance: mean={mean_dist:.1f} pc ({mean_lo:.1f}-{mean_hi:.1f})")
            log.info(f"Space density: {density:.2e} pc^-3 ({dens_lo:.2e}-{dens_hi:.2e})")

    # 2. Temperature statistics
    if "T_eff_K" in candidates.columns:
        temps = candidates["T_eff_K"].dropna().values
        temps = temps[(temps > 0) & (temps < 1000)]  # Filter unphysical values

        if len(temps) > 0:
            mean_temp, temp_lo, temp_hi = bootstrap.bootstrap_mean_ci(temps)
            frac_below_300 = (temps < 300).mean()

            results["temperature"] = {
                "mean": mean_temp,
                "mean_ci": (temp_lo, temp_hi),
                "frac_below_300k": frac_below_300,
                "n_measured": len(temps),
            }

            log.info(f"Temperature: mean={mean_temp:.1f} K ({temp_lo:.1f}-{temp_hi:.1f})")
            log.info(f"Fraction below 300K: {frac_below_300:.2%}")

    # 3. Proper motion statistics
    if "pm_total" in candidates.columns:
        pm = candidates["pm_total"].dropna().values
        pm = pm[pm > 0]  # Filter zero PM

        if len(pm) > 0:
            mean_pm, pm_lo, pm_hi = bootstrap.bootstrap_mean_ci(pm)
            median_pm, med_pm_lo, med_pm_hi = bootstrap.bootstrap_median_ci(pm)

            results["proper_motion"] = {
                "mean_mas_yr": mean_pm,
                "mean_ci": (pm_lo, pm_hi),
                "median_mas_yr": median_pm,
                "median_ci": (med_pm_lo, med_pm_hi),
                "n_measured": len(pm),
            }

            log.info(f"Proper motion: median={median_pm:.1f} mas/yr")

    # 4. Save results
    results_file = output_dir / "population_analysis_results.json"
    with open(results_file, "w") as f:
        # Convert tuples to lists for JSON
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, tuple):
                        json_results[key][k] = list(v)
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value

        json.dump(json_results, f, indent=2)

    log.info(f"Results saved to {results_file}")

    return results


def main():
    """Test the Bayesian population inference."""
    print("=" * 70)
    print("TASNI Bayesian Population Inference Test")
    print("=" * 70)

    # Load golden targets
    golden_file = OUTPUT_DIR / "golden_targets.csv"
    if not golden_file.exists():
        golden_file = OUTPUT_DIR / "final" / "golden_targets.csv"

    if golden_file.exists():
        print(f"\nLoading golden targets from {golden_file}")
        golden = pd.read_csv(golden_file)
        print(f"Loaded {len(golden)} candidates")

        # Run population analysis
        results = run_full_population_analysis(golden)

        print("\n" + "=" * 70)
        print("Population Analysis Results")
        print("=" * 70)
        for key, value in results.items():
            print(f"\n{key.upper()}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
    else:
        print(f"\nGolden targets file not found: {golden_file}")
        print("Run the full pipeline first to generate candidates.")


if __name__ == "__main__":
    main()
