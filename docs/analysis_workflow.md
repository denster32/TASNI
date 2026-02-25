# TASNI Analysis Workflow Guide

**Date:** February 2, 2025
**Purpose:** Step-by-step guide for analyzing TASNI data and results

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Exploration](#data-exploration)
3. [Feature Extraction](#feature-extraction)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Candidate Selection](#candidate-selection)
6. [Spectroscopy Planning](#spectroscopy-planning)
7. [Publication Preparation](#publication-preparation)

---

## Quick Start

### Prerequisites

```bash
# Recommended: use Poetry (see docs/REPRODUCIBILITY_QUICKSTART.md)
poetry install

# Alternative: pip
pip install -r requirements.txt

# Test installation
python -c "import pandas; import sklearn; import astropy; print('✓ All imports successful')"
```

### Basic Workflow

```bash
# 1. Load and analyze golden sample
poetry run python -c "
import pandas as pd
golden = pd.read_parquet('data/processed/final/golden_improved.parquet')
print(f'Golden sample: {len(golden)} sources')
"

# 2. Run periodogram analysis
poetry run python -m tasni.analysis.periodogram_analysis

# 3. Generate publication figures
poetry run python -m tasni.generation.generate_publication_figures

# 4. Review golden sample
head -5 data/processed/final/golden_improved.csv
```

---

## Data Exploration

### 1. Load Data

```python
import pandas as pd

# Load golden sample
golden = pd.read_parquet('data/processed/final/golden_improved.parquet')
print(f"Golden sample: {len(golden)} sources")

# Load parallax subset
parallax = pd.read_parquet('data/processed/final/golden_improved_parallax.parquet')
print(f"Sources with parallax: {len(parallax)}")
```

### 2. Basic Statistics

```python
# Temperature distribution
print(golden['T_eff_K'].describe())

# Proper motion distribution
print(golden['pm_total'].describe())

# Color distribution
print(golden['w1_w2_color'].describe())
```

### 3. Plot Distributions

```python
import matplotlib.pyplot as plt

# Temperature histogram
plt.hist(golden['T_eff_K'], bins=20)
plt.xlabel('Temperature (K)')
plt.ylabel('Count')
plt.title('Golden Targets: Temperature Distribution')
plt.savefig('reports/figures/temp_distribution.png')
```

### 4. Color-Color Diagram

```python
# W1-W2 vs W2-W3
plt.scatter(golden['w1_w2_color'], golden['w2_w3_color'])
plt.xlabel('W1 - W2')
plt.ylabel('W2 - W3')
plt.title('Golden Targets: Color-Color Diagram')
plt.savefig('reports/figures/color_color.png')
```

---

## Feature Extraction

### 1. Extract Features

```bash
# ML feature extraction (requires tier5 and NEOWISE epoch data)
poetry run python -m tasni.ml.extract_features
```

### 2. Explore Features

```python
# Load features
features = pd.read_parquet('data/processed/features/tier5_features.parquet')
print(f"Features: {len(features.columns)}")

# Feature summary
print(features.describe())

# Check for missing values
missing = features.isnull().sum()
print(f"Missing values:\n{missing[missing > 0]}")
```

### 3. Feature Selection

```python
# Select only numeric features
numeric_features = features.select_dtypes(include=[np.number])

# Remove constant features
constant_cols = numeric_features.columns[numeric_features.nunique() <= 1]
print(f"Removing {len(constant_cols)} constant features")

numeric_features = numeric_features.drop(columns=constant_cols)
print(f"Remaining features: {len(numeric_features.columns)}")
```

---

## Machine Learning Pipeline

### 1. Train Models

```bash
poetry run python -m tasni.ml.train_classifier
```

### 2. Evaluate Models

```python
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score

# Load models
with open('data/processed/ml/models/random_forest.pkl', 'rb') as f:
    rf = pickle.load(f)

# Test on golden targets
X_golden = features.loc[golden['designation']]
y_golden = [1] * len(golden)

# Predict
y_pred = rf.predict(X_golden)
y_prob = rf.predict_proba(X_golden)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_golden, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_golden, y_prob):.3f}")
```

### 3. Feature Importance

```python
# Get feature importance
importance = rf.feature_importances_
feature_names = X_golden.columns

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Print top 20
print(importance_df.head(20))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:20], importance_df['importance'][:20])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest: Top 20 Features')
plt.tight_layout()
plt.savefig('reports/figures/feature_importance.png')
```

---

## Candidate Selection

### 1. Predict Scores

```bash
poetry run python -m tasni.ml.predict_tier5
```

### 2. Load Ranked Results

```python
# Load ranked sources
ranked = pd.read_parquet('data/processed/ml/ranked_tier5.parquet')
print(f"Ranked sources: {len(ranked)}")

# Load top candidates
top_candidates = pd.read_csv('data/processed/ml/ranked_tier5_top10000.csv')
print(f"Top candidates: {len(top_candidates)}")
```

### 3. Filter Candidates

```python
# Filter by temperature (200-400 K)
temp_filtered = top_candidates[
    (top_candidates['T_eff_K'] >= 200) &
    (top_candidates['T_eff_K'] <= 400)
]

# Filter by proper motion (>50 mas/yr)
pm_filtered = temp_filtered[temp_filtered['pm_total'] > 50]

# Filter by composite score (>0.7)
score_filtered = pm_filtered[pm_filtered['weighted_score'] > 0.7]

print(f"Filtered candidates: {len(score_filtered)}")
```

### 4. Visualize Candidates

```python
# Sky map
plt.scatter(score_filtered['ra'], score_filtered['dec'],
            c=score_filtered['T_eff_K'], cmap='viridis')
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
plt.title('Top Candidates: Sky Map')
plt.colorbar(label='Temperature (K)')
plt.savefig('reports/figures/candidate_sky_map.png')
```

---

## Spectroscopy Planning

### 1. Plan Observations

```bash
poetry run python -m tasni.generation.prepare_spectroscopy_targets
```

### 2. Load Observation Plan

```python
# Load plan
plan = pd.read_csv('data/processed/spectroscopy/observation_plan.csv')
print(plan[['designation', 'best_telescope', 'total_time_minutes', 'T_eff_K']])
```

### 3. Create Schedule

```bash
python src/tasni/analysis/spectroscopy_planner.py \
    --targets data/processed/final/golden_targets.csv \
    --output data/processed/spectroscopy/observation_plan.csv \
    --schedule keck_nires
```

### 4. Review Schedule

```python
# Load schedule
schedule = pd.read_csv('data/processed/spectroscopy/observation_plan_keck_nires_schedule.csv')
print(schedule[['designation', 'start_hour', 'end_hour', 'total_time_minutes']])
```

---

## Publication Preparation

### 1. Generate Figures

```bash
# Publication figures (see Makefile)
make figures

# Or directly
poetry run python -m tasni.generation.generate_publication_figures
```

### 2. Prepare Tables

```python
# Table 1: Golden targets
golden[['designation', 'ra', 'dec', 'T_eff_K', 'pm_total']].to_csv('output/paper/table1_golden.csv', index=False)

# Table 2: Top ML candidates
top_candidates.head(100).to_csv('output/paper/table2_ml_candidates.csv', index=False)
```

### 3. Write Paper

```python
# Use template from docs/
# Follow guidelines in CONTRIBUTING.md
```

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'pandas'**
```bash
# Install pandas
conda install pandas
```

**2. Out of Memory**
```bash
# Process in chunks
# Reduce batch size
# Use smaller subset for testing
```

**3. Slow Performance**
```bash
# Use GPU acceleration (if available)
# Increase number of workers
# Use Dask for distributed computing
```

**4. Missing Features**
```bash
# Check feature extraction
# Ensure all source data is present
# Handle missing values appropriately
```

---

## Resources

### Documentation
- [Reproducibility Quickstart](REPRODUCIBILITY_QUICKSTART.md)
- [Data Availability](DATA_AVAILABILITY.md)

### Tools
- [Data Manager](src/tasni/utils/data_manager.py)
- [Security Auditor](src/tasni/utils/security_audit.py)
- [Health Checker](src/tasni/utils/health_check.py)

### Research
- [Research Opportunities](RESEARCH_OPPORTUNITIES.md)
- [Roadmap](ROADMAP.md)

---

**Workflow Version:** 1.0
**Last Updated:** February 2, 2025
**Status:** Active Development
