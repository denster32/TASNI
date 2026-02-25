# TASNI Research Opportunities & Computational Tasks

**Date:** February 2, 2025
**Status:** Research & Development Planning

---

## Executive Summary

TASNI has achieved a groundbreaking discovery (4 fading thermal orphans) and established a complete analysis pipeline. However, numerous research opportunities and computational tasks remain to expand the scientific impact and discover additional anomalies.

**Key Insight:** We've analyzed **810,000 tier5 radio-silent sources** from the full **747 million source AllWISE catalog**. **99.9% of the catalog remains unexplored** for systematic anomaly detection.

---

## Table of Contents

1. [Research Opportunities](#research-opportunities)
2. [Computational Tasks](#computational-tasks)
3. [Priority Matrix](#priority-matrix)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Resource Requirements](#resource-requirements)

---

## Research Opportunities

### 🔴 High Priority (Critical Scientific Impact)

#### 1. Full AllWISE Catalog Analysis

**Current Status:**
- Analyzed: 810,000 sources (tier5 radio-silent)
- Total AllWISE: 747,000,000 sources
- **Unexplored: 746,190,000 sources (99.89%)**

**Research Goal:**
Systematically search the entire AllWISE catalog for thermal anomalies using the proven TASNI pipeline.

**Expected Discoveries:**
- Additional fading thermal orphans (estimated: 10-50)
- Extremely cold brown dwarfs (T < 200 K)
- Rare subpopulations (unusual colors, variability patterns)
- New astrophysical phenomena

**Computational Requirements:**
- CPU: 10,000+ core-hours
- Memory: 500GB+ RAM
- Storage: 2TB+ intermediate
- GPU: Optional (for ML acceleration)

**Research Impact:** ⭐⭐⭐⭐⭐ (Transformative)

---

#### 2. Machine Learning Classification of 810K Tier5 Sources

**Current Status:**
- 100 golden targets (ranked by human criteria)
- 810,000 tier5 sources (unclassified)
- No ML models deployed

**Research Goal:**
Train machine learning models to classify all 810,000 tier5 sources and identify high-priority candidates.

**Approaches:**
1. **Supervised Classification:**
   - Train on golden sample (100 targets)
   - Features: colors, variability, proper motion, coordinates
   - Models: Random Forest, XGBoost, Neural Networks
   - Output: Probability scores for all 810K sources

2. **Unsupervised Clustering:**
   - UMAP + HDBSCAN clustering
   - Identify rare subpopulations
   - Discover novel anomaly types
   - Output: Cluster assignments, outlier scores

3. **Semi-Supervised Learning:**
   - Combine labeled (golden targets) + unlabeled (tier5)
   - Pseudo-labeling, self-training
   - Improve classification with limited labels

**Expected Outcomes:**
- Ranked list of all 810,000 sources
- Identification of 1,000-10,000 high-priority candidates
- Discovery of 50-200 additional fading orphans
- Characterization of distinct subpopulations

**Computational Requirements:**
- CPU: 500+ core-hours (training)
- GPU: 100+ core-hours (neural networks)
- Memory: 128GB RAM
- Storage: 10GB+ models + features

**Research Impact:** ⭐⭐⭐⭐⭐ (High)

---

#### 3. Extended Multi-Wavelength Cross-Correlation

**Current Status:**
- Crossmatched with: Gaia (optical), 2MASS (NIR), Pan-STARRS (optical), Legacy (optical), NVSS (radio)
- Missing surveys: UKIDSS, VISTA, Herschel, SCUBA-2, VLASS, LOFAR, eROSITA, Chandra, ZTF, LSST (future)

**Research Goal:**
Incorporate all available surveys to create the most comprehensive multi-wavelength anomaly catalog.

**Surveys to Add:**

| Survey | Wavelength | Sources | Priority |
|---------|-------------|----------|----------|
| **UKIDSS** | 0.8-2.4 μm | 1B+ | 🔴 Critical |
| **VISTA** | 0.9-2.4 μm | 1B+ | 🔴 Critical |
| **Herschel** | 70-500 μm | 1M+ | 🟡 Important |
| **SCUBA-2** | 450 μm, 850 μm | 1M+ | 🟡 Important |
| **VLASS** | 3 GHz | 10M+ | 🟡 Important |
| **LOFAR** | 150 MHz | 10M+ | 🟢 Nice to Have |
| **eROSITA** | 0.2-10 keV | 1M+ | 🟡 Important |
| **Chandra/XMM** | 0.1-10 keV | 1M+ | 🟢 Nice to Have |
| **ZTF** | 0.5-1.0 μm | 1B+ | 🔴 Critical |
| **Rubin/LSST** | 0.3-1.0 μm | 20B+ | 🔴 Critical |

**Expected Discoveries:**
- Objects visible in additional wavelengths (new classifications)
- Hidden infrared sources (no optical at all)
- Extreme outliers (unexpected multi-wavelength properties)
- Time-domain anomalies (ZTF, LSST)

**Computational Requirements:**
- CPU: 5,000+ core-hours (crossmatching)
- Memory: 1TB+ RAM (large catalogs)
- Storage: 5TB+ (downloaded catalogs, crossmatches)
- GPU: Not required

**Research Impact:** ⭐⭐⭐⭐⭐ (Transformative)

---

#### 4. NEOWISE Extended Variability Analysis

**Current Status:**
- Analyzed 810K tier5 sources for variability
- 4 fading thermal orphans discovered
- Limited time-series analysis (10 years)

**Research Goal:**
Perform comprehensive time-domain analysis on NEOWISE light curves to discover additional variability patterns.

**Analysis Types:**

1. **Linear Trend Detection:**
   - Fit linear models to light curves
   - Identify sources with significant brightening/dimming
   - Search for additional fading orphans
   - Expected: 10-50 new fading sources

2. **Periodic Variability:**
   - Lomb-Scargle periodogram analysis
   - Identify rotating brown dwarfs (periods: hours to days)
   - Detect eclipsing binaries
   - Expected: 1,000+ periodic variables

3. **Stochastic Variability:**
   - Calculate structure function
   - Identify flaring sources
   - Detect accretion events
   - Expected: 5,000+ stochastic variables

4. **Multi-band Correlation:**
   - W1 vs W2 correlation analysis
   - Identify chromatic variability
   - Search for wavelength-dependent trends
   - Expected: Novel discovery space

5. **Machine Learning for Light Curves:**
   - CNN for pattern recognition
   - RNN (LSTM/GRU) for time-series classification
   - Unsupervised anomaly detection in light curves
   - Expected: 100+ rare patterns

**Expected Outcomes:**
- Complete variability catalog for 810K sources
- 10-50 new fading thermal orphans
- 1,000+ periodic variables
- 5,000+ stochastic variables
- 100+ novel variability patterns

**Computational Requirements:**
- CPU: 2,000+ core-hours (periodogram)
- GPU: 500+ core-hours (CNN/RNN)
- Memory: 64GB RAM
- Storage: 500GB (intermediate light curve data)

**Research Impact:** ⭐⭐⭐⭐⭐ (High)

---

### 🟡 Medium Priority (Important Scientific Impact)

#### 5. Population Synthesis & Statistical Analysis

**Current Status:**
- 4 fading thermal orphans discovered
- 100 golden targets identified
- No statistical characterization

**Research Goal:**
Perform comprehensive population analysis to understand the distribution of thermal anomalies and place constraints on astrophysical models.

**Analysis Tasks:**

1. **Luminosity Function:**
   - Calculate space density of thermal anomalies
   - Derive luminosity function
   - Compare with brown dwarf models
   - Constrain cooling curves

2. **Spatial Distribution:**
   - Map sky distribution of anomalies
   - Test for Galactic latitude/longitude dependence
   - Identify clustering or avoidance regions
   - Search for non-uniform distributions

3. **Color Distribution:**
   - Construct color-color diagrams (W1-W2, W2-W3, etc.)
   - Identify distinct populations
   - Compare with known brown dwarf sequences
   - Discover new sequences

4. **Kinematics Analysis:**
   - Proper motion statistics
   - Velocity distribution
   - Age estimates from kinematics
   - Galactic orbit analysis

5. **Discovery Rate Estimation:**
   - Calculate expected number of fading orphans
   - Compare with observed number (4)
   - Estimate total population
   - Predict future discoveries

**Expected Outcomes:**
- Complete statistical characterization
- Constraints on brown dwarf evolution
- Predictions for future searches
- Publication-ready statistical analysis

**Computational Requirements:**
- CPU: 200+ core-hours
- Memory: 32GB RAM
- Storage: Minimal
- GPU: Not required

**Research Impact:** ⭐⭐⭐⭐ (Medium-High)

---

#### 6. JWST Observations Planning

**Current Status:**
- 4 fading thermal orphans identified
- No JWST observations scheduled
- No observation planning tools

**Research Goal:**
Prepare for and execute JWST observations to obtain mid-IR spectra of fading thermal orphans.

**Tasks:**

1. **Observation Planning:**
   - Calculate visibility windows
   - Estimate exposure times
   - Optimize instrument configuration
   - Design observation sequences

2. **Spectral Modeling:**
   - Generate synthetic spectra (Sonora, Exo-REM models)
   - Predict molecular bands (CH₄, H₂O, NH₃)
   - Identify diagnostic features
   - Estimate signal-to-noise

3. **Proposal Writing:**
   - Write JWST Cycle proposals
   - Prepare figures and tables
   - Justify scientific significance
   - Submit to TAC

4. **Data Analysis Pipeline:**
   - Prepare MIRI data reduction pipeline
   - Develop spectral fitting tools
   - Create atmospheric retrieval software
   - Plan publication workflow

**Targets:**
- J143046.35-025927.8 (T_eff=293 K)
- J231029.40-060547.3 (T_eff=258 K)
- J193547.43+601201.5 (T_eff=251 K)
- J060501.01-545944.5 (T_eff=253 K)

**Expected Outcomes:**
- First mid-IR spectra of fading orphans
- Atmospheric composition measurements
- Temperature validation
- Publication of JWST results

**Computational Requirements:**
- CPU: 100+ core-hours (modeling)
- Memory: 16GB RAM
- Storage: 10GB (models, spectra)
- GPU: Not required

**Research Impact:** ⭐⭐⭐⭐⭐ (Transformative)

---

#### 7. Parallax Measurements & Distance Determination

**Current Status:**
- 4 fading orphans with proper motion
- No distance measurements
- No absolute luminosities

**Research Goal:**
Measure distances to fading thermal orphans using parallax observations.

**Methods:**

1. **Gaia DR4 (Expected 2025):**
   - Use Gaia astrometry
   - Measure parallaxes (if detectable)
   - Expected precision: ~0.1 mas (10% at 10 pc)

2. **HST Astrometry:**
   - Hubble Space Telescope observations
   - High-precision astrometry
   - Target: ~0.05 mas precision

3. **Ground-based Astrometry:**
   - Keck NIRC2 adaptive optics
   - VLT NAOMI/GRAVITY
   - Target: ~0.1 mas precision

4. **Photometric Distance Estimates:**
   - Use absolute magnitude - T_eff relation
   - Compare with brown dwarf models
   - Uncertainty: ~50%

**Expected Outcomes:**
- Distance measurements (±10%)
- Absolute luminosities
- Mass-age constraints
- Brown dwarf model validation

**Computational Requirements:**
- CPU: 50+ core-hours (astrometry)
- Memory: 16GB RAM
- Storage: Minimal
- GPU: Not required

**Research Impact:** ⭐⭐⭐⭐ (High)

---

### 🟢 Low Priority (Exploratory Scientific Impact)

#### 8. Alternative Search Strategies

**Current Status:**
- Single search method (thermal colors + multi-wavelength veto)
- No alternative approaches tested

**Research Goal:**
Develop and test alternative anomaly detection strategies.

**Approaches:**

1. **Color-Color Space:**
   - Explore unusual color combinations
   - Use unsupervised clustering
   - Identify outliers in multi-dimensional space

2. **Time-Domain Searches:**
   - Focus on variability rather than thermal colors
   - Search for transient infrared sources
   - Detect突然出现的 events

3. **Spatial Clustering:**
   - Search for spatial concentrations
   - Identify unusual sky patterns
   - Test for non-uniform distributions

4. **Machine Learning Anomaly Detection:**
   - Isolation Forest
   - One-Class SVM
   - Autoencoder reconstruction error
   - Deep learning approaches

5. **Multi-Messenger Searches:**
   - Correlate with neutrino events
   - Cross-match with gravitational wave events
   - Search for radio transients
   - Investigate gamma-ray sources

**Expected Outcomes:**
- Additional discovery methods
- Complementary candidate lists
- New types of anomalies
- Improved detection efficiency

**Computational Requirements:**
- CPU: 1,000+ core-hours
- GPU: 500+ core-hours (ML)
- Memory: 128GB RAM
- Storage: 100GB+

**Research Impact:** ⭐⭐⭐ (Medium)

---

#### 9. Theoretical Modeling & Signatures

**Current Status:**
- Limited theoretical framework
- No comprehensive modeling

**Research Goal:**
Develop theoretical models for various astrophysical scenarios.

**Models to Develop:**

1. **Dyson Sphere Signatures:**
   - Waste heat spectra (blackbody, partial coverage)
   - Temporal evolution (construction, maintenance)
   - Multi-wavelength predictions
   - Observational constraints

2. **Brown Dwarf Evolution:**
   - Cooling curves for ultra-cold objects (T < 300 K)
   - Atmospheric chemistry (CH₄, H₂O, NH₃)
   - Cloud formation models
   - Comparison with observations

3. **Exoplanet Scenarios:**
   - Free-floating planets (rogue planets)
   - Wide-orbit planets ( >100 AU)
   - Mass-luminosity relations
   - Formation pathways

4. **Artificial Source Models:**
   - Engineered emission spectra
   - Temporal modulation (pulsing, periodic)
   - Directional emission patterns
   - Expected detection rates

**Expected Outcomes:**
- Comprehensive theoretical framework
- Predictive signatures for observations
- Constraint calculations for various scenarios
- Publication-ready theoretical results

**Computational Requirements:**
- CPU: 500+ core-hours (modeling)
- GPU: Not required
- Memory: 32GB RAM
- Storage: 10GB+ (model outputs)

**Research Impact:** ⭐⭐⭐ (Medium)

---

## Computational Tasks

### 🔴 High Impact (Major Scientific Advances)

#### 1. Full Pipeline Execution on AllWISE

**Task:** Run TASNI pipeline on all 747 million AllWISE sources

**Current:**
- Processed: 810,000 sources (tier5 radio-silent)
- **Remaining: 746,190,000 sources (99.89%)**

**Implementation:**

```python
# Full AllWISE pipeline
from scripts.optimized.optimized_pipeline import OptimizedPipeline

# Initialize
pipeline = OptimizedPipeline(
    n_workers=128,  # Massive parallelization
    use_gpu=True,   # GPU acceleration
    batch_size=100000  # Large batches
)

# Process full catalog
results = pipeline.run_full_allwise()

# Expected output:
# - 746M sources processed
# - 1M+ thermal anomalies identified
# - 10-50 fading orphans discovered
# - Complete catalog generated
```

**Optimization Strategies:**
- **GPU Acceleration:** Use cupy/cudf for vectorized operations
- **Parallel Processing:** Distribute across multiple nodes
- **Streaming:** Process in chunks, avoid loading full catalog
- **Caching:** Cache intermediate results for resume capability
- **Progress Tracking:** Checkpoint every 10M sources

**Expected Runtime:**
- CPU-only (128 cores): ~3 months
- GPU-accelerated (4x A100): ~2-3 weeks
- Distributed (10 nodes): ~1-2 weeks

**Output:**
- Complete thermal anomaly catalog (1M+ sources)
- All fading thermal orphans (estimated: 10-50)
- Statistical characterization
- Publication-ready results

**Impact:** ⭐⭐⭐⭐⭐ (Transformative)

---

#### 2. Machine Learning Pipeline Development

**Task:** Build end-to-end ML pipeline for classification and anomaly detection

**Components:**

```python
# ML Pipeline Architecture
class TASNIMLPipeline:
    """Complete ML pipeline for TASNI"""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.supervised_model = SupervisedClassifier()
        self.unsupervised_model = UnsupervisedClusterer()
        self.anomaly_detector = AnomalyDetector()

    def extract_features(self, sources):
        """Extract 500+ features from sources"""
        features = {
            # Photometric (100)
            'colors': compute_all_colors(),
            'magnitudes': extract_magnitudes(),

            # Variability (200)
            'trend': fit_linear_trend(),
            'period': compute_periodogram(),
            'stochastic': compute_structure_function(),

            # Kinematics (50)
            'pm_total': calculate_pm(),
            'pm_angle': calculate_pm_angle(),
            'galactic_coords': convert_to_galactic(),

            # Multi-wavelength (100)
            'detection_flags': get_detection_flags(),
            'upper_limits': get_upper_limits(),
            'seds': fit_sed(),
        }
        return features

    def train_supervised(self, golden_targets, tier5):
        """Train supervised classifier"""
        # Extract features
        X_golden = self.extract_features(golden_targets)
        X_tier5 = self.extract_features(tier5)

        # Train models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=1000),
            'xgboost': XGBClassifier(n_estimators=1000),
            'neural_net': MLPClassifier(hidden_layers=(512, 256, 128)),
        }

        # Ensemble
        ensemble = VotingClassifier(models)
        ensemble.fit(X_golden, y_golden)

        # Predict on all tier5
        probabilities = ensemble.predict_proba(X_tier5)

        return probabilities

    def train_unsupervised(self, tier5):
        """Train unsupervised models"""
        # UMAP embedding
        umap_model = UMAP(n_components=50)
        embedding = umap_model.fit_transform(X_tier5)

        # HDBSCAN clustering
        clusterer = HDBSCAN(min_cluster_size=100)
        clusters = clusterer.fit_predict(embedding)

        # Isolation Forest for anomalies
        iso_forest = IsolationForest(contamination=0.01)
        anomaly_scores = iso_forest.fit_predict(X_tier5)

        return clusters, anomaly_scores

    def run_pipeline(self, tier5_sources):
        """Complete ML pipeline"""
        # Extract features
        features = self.extract_features(tier5_sources)

        # Supervised classification
        supervised_scores = self.train_supervised(
            golden_targets, tier5_sources
        )

        # Unsupervised clustering
        clusters, anomaly_scores = self.train_unsupervised(tier5_sources)

        # Combine scores
        final_scores = {
            'supervised': supervised_scores,
            'clusters': clusters,
            'anomaly': anomaly_scores,
            'combined': combine_scores(supervised_scores, anomaly_scores)
        }

        # Rank sources
        ranked = rank_sources(final_scores)

        return ranked
```

**Features to Extract (500+):**

| Category | Features | Count |
|----------|-----------|--------|
| **Photometric** | W1, W2, W3, W4, colors, errors | 100 |
| **Variability** | Trend, period, amplitude, stochastic, correlations | 200 |
| **Kinematics** | PM, PM angle, galactic coords, velocities | 50 |
| **Multi-wavelength** | Detection flags, upper limits, SED parameters | 100 |
| **Spatial** | Galactic latitude, longitude, clustering | 30 |
| **Statistical** | Percentiles, moments, quantiles | 20 |

**Models to Train:**

1. **Supervised:**
   - Random Forest (1000 trees)
   - XGBoost (1000 estimators)
   - Neural Network (MLP: 512-256-128)
   - Gradient Boosting
   - Support Vector Machine

2. **Unsupervised:**
   - UMAP (50 components)
   - HDBSCAN (min_cluster_size=100)
   - Isolation Forest (contamination=0.01)
   - Local Outlier Factor
   - One-Class SVM

3. **Deep Learning:**
   - CNN (for light curve images)
   - RNN/LSTM (for time series)
   - Autoencoder (for anomaly detection)
   - Transformer (for multi-modal data)

**Expected Output:**
- Ranked list of all 810K tier5 sources
- Probability scores for each source
- Cluster assignments (10-100 clusters)
- Anomaly scores (outlier detection)
- 1,000-10,000 high-priority candidates
- 50-200 additional fading orphans

**Runtime:**
- Feature extraction: 1,000+ core-hours
- Model training: 500+ core-hours
- Prediction: 100+ core-hours
- Total: 1,600+ core-hours

**Impact:** ⭐⭐⭐⭐⭐ (Transformative)

---

#### 3. GPU-Accelerated Crossmatching

**Task:** Implement GPU-accelerated crossmatching for massive catalogs

**Implementation:**

```python
# GPU Crossmatch using RAPIDS
import cudf
from cuml.neighbors import NearestNeighbors

class GPUCrossmatcher:
    """GPU-accelerated crossmatcher"""

    def __init__(self, match_radius_arcsec=3.0):
        self.match_radius = match_radius_arcsec

    def crossmatch_wise_gaia(self, wise_df, gaia_df):
        """Crossmatch WISE and Gaia on GPU"""

        # Convert to cuDF (GPU DataFrames)
        wise_gpu = cudf.DataFrame.from_pandas(wise_df)
        gaia_gpu = cudf.DataFrame.from_pandas(gaia_df)

        # Extract coordinates
        wise_coords = wise_gpu[['ra', 'dec']].to_gpu()
        gaia_coords = gaia_gpu[['ra', 'dec']].to_gpu()

        # Create nearest neighbors model on GPU
        nn = NearestNeighbors(metric='haversine')
        nn.fit(gaia_coords)

        # Find nearest neighbors within radius
        distances, indices = nn.kneighbors(
            wise_coords,
            n_neighbors=1,
            return_distance=True
        )

        # Filter by radius
        mask = distances <= (self.match_radius / 3600.0)  # deg

        # Create matches
        matches = cudf.DataFrame({
            'wise_index': wise_gpu.index[mask],
            'gaia_index': gaia_gpu.index[indices[mask]],
            'separation_arcsec': distances[mask] * 3600.0
        })

        # Merge back to original data
        results = matches.merge(
            wise_gpu,
            left_on='wise_index',
            right_index=True
        ).merge(
            gaia_gpu,
            left_on='gaia_index',
            right_index=True,
            suffixes=('_wise', '_gaia')
        )

        return results

    def crossmatch_massive_catalogs(self, wise_path, gaia_path):
        """Crossmatch massive catalogs (747M × 1.8B)"""

        # Process in chunks
        chunk_size = 1_000_000  # 1M sources per chunk
        all_results = []

        for chunk in pd.read_parquet(wise_path, chunksize=chunk_size):
            # Crossmatch on GPU
            matches = self.crossmatch_wise_gaia(chunk, gaia_df)

            # Save results
            all_results.append(matches)

        # Concatenate all results
        final_results = cudf.concat(all_results)

        return final_results
```

**Performance Improvements:**

| Operation | CPU (8 cores) | GPU (A100) | Speedup |
|-----------|----------------|--------------|----------|
| Load catalog (10M) | 60s | 10s | 6x |
| Nearest neighbors (10M) | 300s | 30s | 10x |
| Merge results | 60s | 20s | 3x |
| **Total (10M)** | **420s (7 min)** | **60s (1 min)** | **7x** |

**Expected Speedup:**
- 1M sources: 7x faster
- 10M sources: 10x faster
- 100M sources: 15x faster
- Full AllWISE (747M): 20x faster

**Hardware Requirements:**
- GPU: NVIDIA A100 (80GB) or RTX 4090 (24GB)
- RAM: 128GB (for CPU data loading)
- Storage: 10TB+ (for catalogs, intermediate)

**Impact:** ⭐⭐⭐⭐⭐ (Transformative)

---

#### 4. Deep Learning for Light Curve Classification

**Task:** Train deep neural networks to classify NEOWISE light curves

**Architecture:**

```python
# CNN for Light Curve Classification
import torch
import torch.nn as nn

class LightCurveCNN(nn.Module):
    """Convolutional Neural Network for light curves"""

    def __init__(self, input_length=100, num_classes=4):
        super().__init__()

        # 1D CNN layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5)

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Fully connected
        self.fc1 = nn.Linear(256 * 11, 512)  # Adjust based on input size
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch, 2, 100) - (W1, W2, time)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

# RNN for Time Series
class LightCurveRNN(nn.Module):
    """Recurrent Neural Network for light curves"""

    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_classes=4):
        super().__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Fully connected
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, 100, 2) - (time, W1, W2)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last time step
        out = lstm_out[:, -1, :]

        # FC
        out = self.fc(out)

        return out

# Training Pipeline
def train_light_curve_model(
    light_curves,  # Shape: (N, 100, 2)
    labels,         # Shape: (N, 4) - [fading, periodic, stochastic, normal]
    epochs=100,
    batch_size=64
):
    """Train deep learning model"""

    # Split data
    train_lc, test_lc, train_labels, test_labels = train_test_split(
        light_curves, labels, test_size=0.2
    )

    # Initialize model
    model = LightCurveCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()

        for batch in DataLoader(train_lc, batch_size=batch_size):
            optimizer.zero_grad()

            # Forward
            outputs = model(batch['light_curve'])
            loss = criterion(outputs, batch['labels'])

            # Backward
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(test_lc)
            val_accuracy = accuracy_score(
                test_labels,
                val_outputs.argmax(axis=1)
            )

        print(f"Epoch {epoch}, Val Acc: {val_accuracy:.4f}")

    return model
```

**Model Types:**

1. **CNN:** Convolutional Neural Network for pattern recognition
   - Input: Light curve images (W1, W2 vs time)
   - Output: Classification probabilities
   - Best for: Fixed-length patterns

2. **RNN/LSTM:** Recurrent Neural Network for time series
   - Input: Sequential light curve data
   - Output: Classification probabilities
   - Best for: Temporal dependencies

3. **Transformer:** Attention-based model
   - Input: Sequential light curve data
   - Output: Classification probabilities
   - Best for: Long-range dependencies

4. **Autoencoder:** Unsupervised anomaly detection
   - Input: Light curve data
   - Output: Reconstructed light curve
   - Best for: Anomaly detection (high reconstruction error = anomaly)

**Expected Accuracy:**
- Classification: 85-95% accuracy
- Anomaly detection: 90-98% precision
- Fading orphans: 80-90% recall

**Runtime:**
- Training: 100+ GPU-hours (A100)
- Prediction: 10+ GPU-hours (for 810K sources)

**Impact:** ⭐⭐⭐⭐ (High)

---

### 🟡 Medium Impact (Significant Scientific Advances)

#### 5. Distributed Computing Framework

**Task:** Set up distributed computing for massive-scale analysis

**Implementation:**

```python
# Dask distributed computing
import dask
from dask.distributed import Client
import dask.dataframe as dd

class DistributedTASNI:
    """Distributed TASNI pipeline"""

    def __init__(self, n_workers=128):
        # Initialize Dask client
        self.client = Client(n_workers=n_workers)

    def process_full_allwise(self, allwise_path):
        """Process full AllWISE catalog distributed"""

        # Load catalog (distributed)
        allwise_df = dd.read_parquet(allwise_path)

        # Crossmatch with Gaia (distributed)
        gaia_df = dd.read_parquet('data/gaia/gaia_*.parquet')
        crossmatched = crossmatch_distributed(allwise_df, gaia_df)

        # Filter orphans (distributed)
        orphans = crossmatched[~crossmatched['matched']]

        # Apply filters (distributed)
        thermal = orphans[orphans['w1_w2'] > 0.5]
        filtered = filter_multiwavelength_distributed(thermal)

        # Compute results (distributed)
        results = filtered.compute()  # Triggers distributed computation

        return results

    def process_parallel_healpix(self):
        """Process HEALPix tiles in parallel"""

        # Create tasks for each HEALPix tile
        tasks = []
        for hpix in range(12288):  # NSIDE=32
            task = self.client.submit(
                process_healpix_tile,
                hpix
            )
            tasks.append(task)

        # Collect results
        results = self.client.gather(tasks)

        return results
```

**Distributed Framework Options:**

1. **Dask:** Python-native, easy to use
   - Pros: Python integration, dynamic scheduling
   - Cons: Higher overhead for small tasks

2. **Ray:** High-performance distributed computing
   - Pros: Low overhead, actor model
   - Cons: Steeper learning curve

3. **Spark:** Big data processing
   - Pros: Mature ecosystem, scale to petabytes
   - Cons: Java-based, less Pythonic

4. **Kubernetes (K8s):** Container orchestration
   - Pros: Industry standard, auto-scaling
   - Cons: Complex setup

**Expected Performance:**

| Framework | 10 nodes | 100 nodes | 1000 nodes |
|-----------|-----------|------------|-------------|
| **Dask** | 3 months → 10 days | 3 months → 1 day | 3 months → 2 hours |
| **Ray** | 3 months → 8 days | 3 months → 12 hours | 3 months → 1.5 hours |
| **Spark** | 3 months → 12 days | 3 months → 1.5 days | 3 months → 3 hours |

**Hardware Requirements:**
- 10-1000 nodes (depending on scale)
- 128GB RAM per node
- 10TB storage per node
- High-speed network (10GbE+)

**Impact:** ⭐⭐⭐⭐ (High)

---

#### 6. Interactive Visualization Dashboard

**Task:** Build interactive web dashboard for TASNI data exploration

**Implementation:**

```python
# Streamlit Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px

def tasni_dashboard():
    """Interactive TASNI dashboard"""

    st.title("TASNI - Thermal Anomaly Search")
    st.sidebar.title("Filters")

    # Load data
    golden = pd.read_csv('data/processed/final/golden_targets.csv')
    tier5 = pd.read_parquet('data/processed/final/tier5_radio_silent.parquet')

    # Filters
    min_temp = st.sidebar.slider("Min T_eff (K)", 0, 500, 200)
    min_pm = st.sidebar.slider("Min PM (mas/yr)", 0, 500, 50)
    min_score = st.sidebar.slider("Min Score", 0, 100, 50)

    # Apply filters
    filtered = golden[
        (golden['T_eff_K'] >= min_temp) &
        (golden['pm_total'] >= min_pm) &
        (golden['composite_score'] >= min_score)
    ]

    # Display
    st.subheader(f"Showing {len(filtered)} sources")

    # Map
    fig_map = px.scatter_mapbox(
        filtered,
        lat='dec',
        lon='ra',
        color='T_eff_K',
        hover_name='designation',
        mapbox_style='dark'
    )
    st.plotly_chart(fig_map)

    # Color-color diagram
    fig_cc = px.scatter(
        filtered,
        x='w1_w2_color',
        y='w2_w3_color',
        color='T_eff_K',
        hover_name='designation'
    )
    st.plotly_chart(fig_cc)

    # Light curves
    source = st.selectbox("Select Source", filtered['designation'])
    plot_light_curve(source)

# Run dashboard
if __name__ == '__main__':
    tasni_dashboard()
```

**Dashboard Features:**

1. **Interactive Maps:**
   - Sky map of all sources
   - Color-coded by temperature
   - Zoom/pan functionality
   - Click for details

2. **Color-Color Diagrams:**
   - W1-W2 vs W2-W3
   - Color distributions
   - Highlight golden targets
   - Compare with known sequences

3. **Light Curve Viewer:**
   - NEOWISE epoch data
   - Linear trend fitting
   - Periodogram display
   - Variability metrics

4. **Filter Controls:**
   - Temperature range
   - Proper motion range
   - Score threshold
   - Multi-wavelength flags

5. **Download Options:**
   - Export filtered sources
   - Download figures
   - Generate reports
   - API access

**Technologies:**
- Frontend: Streamlit, Dash, Plotly
- Backend: Flask, FastAPI
- Database: PostgreSQL, DuckDB
- Deployment: Docker, Kubernetes

**Impact:** ⭐⭐⭐ (Medium)

---

## Priority Matrix

| Research/Task | Impact | Effort | Priority |
|---------------|--------|---------|----------|
| **Full AllWISE Analysis** | ⭐⭐⭐⭐⭐ | 🔴 Very High | 🔴 Critical |
| **ML Classification** | ⭐⭐⭐⭐⭐ | 🟡 Medium | 🔴 Critical |
| **Multi-wavelength Cross-Correlation** | ⭐⭐⭐⭐⭐ | 🔴 Very High | 🔴 Critical |
| **NEOWISE Variability Analysis** | ⭐⭐⭐⭐⭐ | 🟡 Medium | 🔴 Critical |
| **GPU-Accelerated Crossmatch** | ⭐⭐⭐⭐⭐ | 🟡 Medium | 🟡 Important |
| **Deep Learning Light Curves** | ⭐⭐⭐⭐ | 🟡 Medium | 🟡 Important |
| **Population Synthesis** | ⭐⭐⭐⭐ | 🟢 Low | 🟡 Important |
| **JWST Planning** | ⭐⭐⭐⭐⭐ | 🟢 Low | 🟡 Important |
| **Parallax Measurements** | ⭐⭐⭐⭐ | 🟢 Low | 🟡 Important |
| **Distributed Computing** | ⭐⭐⭐⭐ | 🔴 Very High | 🟢 Nice to Have |
| **Interactive Dashboard** | ⭐⭐⭐ | 🟡 Medium | 🟢 Nice to Have |
| **Alternative Search Methods** | ⭐⭐⭐ | 🟡 Medium | 🟢 Nice to Have |
| **Theoretical Modeling** | ⭐⭐⭐ | 🟡 Medium | 🟢 Nice to Have |

---

## Implementation Roadmap

### Phase 1: Immediate (Next 1-3 months)

**Goal:** Extend current analysis to 810K tier5 sources

| Task | Weeks | Resources | Deliverables |
|------|--------|-----------|--------------|
| ML Feature Extraction | 2 | 1 GPU | 500 features extracted |
| Supervised Model Training | 1 | 1 GPU | Classification model |
| Unsupervised Clustering | 1 | 1 GPU | Cluster assignments |
| Rank All 810K Sources | 1 | 1 GPU | Ranked candidate list |
| Variability Analysis | 4 | 8 CPU | Extended variability catalog |
| Paper on 810K Analysis | 4 | 1 writer | Draft paper |

**Total:** 12 weeks (3 months)
**Deliverables:** Ranked list of 810K sources, 50-200 additional fading orphans

---

### Phase 2: Short-Term (Next 3-6 months)

**Goal:** Incorporate additional surveys and improve methods

| Task | Weeks | Resources | Deliverables |
|------|--------|-----------|--------------|
| Download UKIDSS/VISTA | 2 | 1 node | Full catalogs |
| Crossmatch UKIDSS/VISTA | 2 | 4 CPUs | Matched catalog |
| Download ZTF | 4 | 1 node | Light curves |
| GPU-Accelerated Crossmatch | 4 | 1 GPU | 10x faster crossmatch |
| Deep Learning Light Curves | 4 | 1 GPU | Classification model |
| JWST Proposal Writing | 4 | 1 writer | Ready for submission |
| Parallax Observations | 8 | Telescope | Distance measurements |

**Total:** 24 weeks (6 months)
**Deliverables:** Multi-wavelength catalog, GPU crossmatch, JWST proposals, distances

---

### Phase 3: Medium-Term (Next 6-12 months)

**Goal:** Execute full AllWISE catalog analysis

| Task | Months | Resources | Deliverables |
|------|---------|-----------|--------------|
| Distributed Computing Setup | 1 | Dev team | 10-node cluster |
| Full AllWISE Pipeline Run | 3 | 10 nodes | Complete catalog |
| Population Synthesis | 2 | 1 CPU | Statistical analysis |
| JWST Observations | 6 | JWST time | Mid-IR spectra |
| Publication of Results | 2 | 1 writer | Published paper |

**Total:** 12 months (1 year)
**Deliverables:** Full AllWISE analysis, population statistics, JWST spectra, publications

---

### Phase 4: Long-Term (Next 12-24 months)

**Goal:** Community catalog and LSST integration

| Task | Months | Resources | Deliverables |
|------|---------|-----------|--------------|
| Community Web Portal | 6 | Dev team | Public access |
| LSST Integration | 6 | Dev team | Real-time processing |
| Automated Alert System | 3 | Dev team | Live notifications |
| Extended Search Methods | 6 | Dev team | Novel techniques |
| Theoretical Framework | 6 | Research team | Published theory |

**Total:** 18-24 months (1.5-2 years)
**Deliverables:** Public catalog, LSST integration, automated alerts, novel methods

---

## Resource Requirements

### Computational Resources

| Task | CPU | GPU | RAM | Storage | Runtime |
|------|------|------|------|---------|----------|
| **ML Feature Extraction (810K)** | 500 cores | 1 A100 | 128GB | 1,000 hrs |
| **ML Training (810K)** | 100 cores | 1 A100 | 64GB | 500 hrs |
| **ML Prediction (810K)** | 50 cores | 1 A100 | 64GB | 100 hrs |
| **GPU Crossmatch (AllWISE)** | 128 cores | 4 A100 | 1TB | 4 weeks |
| **Full AllWISE Pipeline** | 10,000 cores | 10 A100 | 10TB | 12 weeks |
| **Deep Learning Training** | 100 cores | 2 A100 | 128GB | 200 GPU-hrs |
| **Distributed (100 nodes)** | 10,000 cores | 10 A100 | 10TB | 1 week |

### Financial Resources

| Resource | Cost (Low) | Cost (High) | Notes |
|-----------|-------------|--------------|-------|
| **GPU Compute (A100)** | $5K | $20K | 1-3 months |
| **CPU Compute (Cloud)** | $10K | $100K | 3-12 months |
| **On-Prem Cluster** | $100K | $500K | One-time |
| **JWST Time** | $50K | $200K | Cycle proposals |
| **Ground-Based Time** | $10K | $50K | Keck, VLT |
| **Personnel (Postdoc)** | $100K | $200K | 1 year |
| **Personnel (Software)** | $150K | $300K | 1 year |
| **Total (1 Year)** | $425K | $1.37M | - |

### Human Resources

| Role | FTE | Expertise | Tasks |
|------|------|-----------|-------|
| **Research Scientist** | 1.0 | Astronomy, ML | Analysis, writing |
| **Machine Learning Engineer** | 1.0 | ML, GPU | Model development |
| **Software Engineer** | 0.5 | Python, Dask | Pipeline optimization |
| **Graduate Student** | 1.0 | Astronomy | Data analysis |
| **Undergraduate Student** | 0.5 | Computing | Data processing |
| **Postdoctoral Fellow** | 1.0 | Astronomy | Research lead |

---

## Conclusion

### Summary of Opportunities

**Research:**
- 🔴 **Critical:** Full AllWISE analysis, ML classification, multi-wavelength correlation, extended variability
- 🟡 **Important:** Population synthesis, JWST planning, parallax measurements
- 🟢 **Exploratory:** Alternative methods, theoretical modeling

**Computational:**
- 🔴 **High Impact:** Full AllWISE pipeline, ML pipeline, GPU crossmatch, deep learning
- 🟡 **Medium Impact:** Distributed computing, interactive dashboard
- 🟢 **Nice to Have:** Visualization tools, automated alerts

### Immediate Next Steps

1. **Week 1-4:** ML feature extraction, model training
2. **Week 5-8:** Rank 810K sources, variability analysis
3. **Week 9-12:** Write paper on 810K analysis
4. **Month 4-6:** GPU crossmatch, deep learning models
5. **Month 7-12:** Distributed computing, full AllWISE pipeline

### Expected Discoveries

- **Short-term (3 months):** 50-200 additional fading orphans
- **Medium-term (12 months):** 10-50 total fading orphans (full catalog)
- **Long-term (24 months):** 100-1000 novel anomalies

### Publication Potential

- **Year 1:** 3-5 papers (810K analysis, ML methods, variability)
- **Year 2:** 2-3 papers (full AllWISE, population synthesis)
- **Year 3:** 2-3 papers (JWST results, theoretical work)

**Total Potential:** 7-11 high-impact publications

---

**Last Updated:** February 2, 2025
**Status:** Active Development
**Next Review:** Monthly (updated with progress)
