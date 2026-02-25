# TASNI System Architecture

## Overview

TASNI (Thermal Anomaly Search for Non-communicating Intelligence) is a multi-stage astronomical data pipeline designed to identify non-communicating intelligence signatures through thermal anomalies.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT DATA LAYERS                      │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│   WISE   │   Gaia   │  2MASS   │ Spitzer  │  Secondary  │
│ (Mid-IR) │ (Optical)│  (NIR)   │(Mid-IR) │   Catalogs  │
│  747M   │  1.8B    │   470M   │   300K   │     ...     │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────────────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DOWNLOAD LAYER                              │
│  - HEALPix-tiling (NSIDE=32, 12,288 tiles)            │
│  - TAP async queries                                      │
│  - Streaming parquet I/O                                  │
│  - Checkpoint-based resume                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CROSSMATCH LAYER                           │
│  - GPU-accelerated spatial indexing (BallTree/cKDTree)    │
│  - HEALPix-based filtering                                │
│  - 3-arcsec matching radius                               │
│  - Parallel processing                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FILTERING LAYER                            │
│  - Optical veto (Gaia DR3, 46% removed)                  │
│  - Thermal colors (W1-W2 > 0.5, 99% removed)          │
│  - NIR veto (2MASS, 94% removed)                         │
│  - Deep optical veto (Pan-STARRS, 38% removed)           │
│  - Legacy Survey veto (<0.1% removed)                     │
│  - Radio veto (NVSS, 89% removed)                         │
│  - Temperature filtering (T_eff < 500K)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SCORING LAYER                           │
│  - Multi-wavelength anomaly scoring                         │
│  - Proper motion analysis                                  │
│  - LAMOST spectroscopy cross-check                        │
│  - Isolation scoring                                     │
│  - Solar system rejection                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VARIABILITY LAYER                           │
│  - NEOWISE 10-year light curves                           │
│  - Periodogram analysis                                     │
│  - Stetson J index                                        │
│  - Trend detection (Numba-JIT compiled)                   │
│  - Fade rate calculation                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                             │
│  - Golden sample (100 top targets)                          │
│  - Tier 5 radio-silent candidates (810K sources)          │
│  - Spectroscopic target lists                             │
│  - Publication figures                                     │
│  - Data release packages                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Technologies

### Core Computing
| Component | Technology | Purpose |
|-----------|-------------|---------|
| Numerical Computing | NumPy, SciPy | Array operations, statistics |
| Data Manipulation | Pandas, PyArrow | DataFrame operations, parquet I/O |
| Astronomy | Astropy, Astroquery, HEALPy | Celestial coordinates, TAP queries |
| Database | DuckDB | In-memory analytical queries |

### Spatial Indexing
| Algorithm | Library | Speedup |
|-----------|----------|----------|
| BallTree | scikit-learn | 2-4x |
| cKDTree | SciPy | 4x |
| Naive O(N×M) | Astropy | baseline |

### GPU Acceleration
| Hardware | Library | Use Case |
|----------|----------|-----------|
| NVIDIA RTX 3060 | cuDF, Numba | Crossmatch, variability |
| Intel Arc A770 | PyTorch XPU | ML classification |

### Performance Optimizations
| Component | Technique | Speedup |
|-----------|-----------|----------|
| Crossmatch | BallTree spatial indexing | 4-100x |
| Variability | Numba-JIT compilation | 12-60x |
| Downloads | Async I/O with aiohttp | 10-50x |
| I/O | Streaming parquet | 2-5x |
| Overall | Optimized pipeline | ~100x |

## Data Flow

### Stage 1: Download
1. Define HEALPix grid (NSIDE=32)
2. Query TAP services asynchronously
3. Stream results to parquet files
4. Verify checksums

### Stage 2: Crossmatch
1. Convert RA/Dec to 3D Cartesian coordinates
2. Build spatial index on larger catalog
3. Query nearest neighbors for each source
4. Filter by separation threshold (3")

### Stage 3: Filtering
1. Apply optical veto (Gaia DR3)
2. Filter by thermal colors (W1-W2 > 0.5)
3. Apply NIR veto (2MASS)
4. Check deep optical surveys (Pan-STARRS, Legacy)
5. Veto radio sources (NVSS)
6. Filter by temperature (T_eff < 500K)

### Stage 4: Scoring
1. Calculate isolation scores
2. Cross-check with LAMOST spectra
3. Analyze proper motion
4. Apply multi-wavelength scoring
5. Rank by composite score

### Stage 5: Variability
1. Query NEOWISE for light curves
2. Compute periodograms
3. Calculate RMS, chi-squared, Stetson J
4. Detect trends (fade/brighten)
5. Classify variability type

### Stage 6: Output Generation
1. Select top 100 candidates
2. Generate figures
3. Create spectroscopy target lists
4. Compile publication materials

## Directory Structure

```
tasni/
├── src/tasni/              # Pipeline components
│   ├── core/           # Configuration, logging
│   ├── download/       # Data acquisition
│   ├── crossmatch/     # Spatial matching
│   ├── analysis/       # Data analysis
│   ├── filtering/      # Anomaly detection
│   ├── generation/     # Output generation
│   ├── optimized/      # Performance versions
│   ├── ml/            # Machine learning
│   ├── utils/         # Utilities
│   ├── checks/        # Validation
│   ├── misc/          # Miscellaneous
│   └── legacy/        # Superseded code
│
├── data/               # Input catalogs (83GB)
│   ├── wise/          # WISE tiles
│   ├── gaia/          # Gaia DR3 tiles
│   ├── crossmatch/    # Matched sources
│   └── ...            # Other catalogs
│
├── output/             # Analysis outputs (42MB)
│   ├── final/         # Golden targets
│   ├── figures/       # Generated plots
│   ├── cutouts/       # Image cutouts
│   ├── spectroscopy/  # Target lists
│   └── periodogram/   # Variability analysis
│
├── archive/            # Intermediate files (24GB)
├── logs/               # Processing logs (3.7MB)
├── data/interim/checkpoints/        # Resume checkpoints
├── paper/              # Publication materials
├── docs/               # Documentation
└── tests/              # Test suite
```

## Performance Metrics

### Processing Times (Full Pipeline)
| Stage | Time (CPU) | Time (GPU) | Notes |
|--------|------------|-------------|--------|
| Download WISE | 8h | 8h | Network-limited |
| Download Gaia | 12h | 12h | Network-limited |
| Crossmatch | 24h | 2h | 12x GPU speedup |
| Filtering | 2h | 1h | Moderate speedup |
| Scoring | 4h | 2h | 2x GPU speedup |
| Variability | 48h | 4h | 12x speedup |
| **Total** | **98h** | **29h** | 3.4x overall |

### Throughput
| Operation | Rate | Notes |
|------------|-------|-------|
| Crossmatch | 414K sources/s | GPU, cKDTree |
| Variability | 11K sources/s | Numba-JIT |
| Downloads | 10MB/s | Async TAP |

### Memory Usage
| Stage | RAM | GPU Memory |
|-------|------|-------------|
| Download | 4GB | N/A |
| Crossmatch | 16GB | 8GB |
| Filtering | 8GB | 4GB |
| Scoring | 12GB | 6GB |
| Variability | 8GB | 4GB |

## Scalability

### Supported Catalog Sizes
| Catalog | Size | Status |
|---------|-------|--------|
| AllWISE | 747M | ✓ Tested |
| Gaia DR3 | 1.8B | ✓ Tested |
| 2MASS | 470M | ✓ Tested |
| Legacy Survey | 3B | ✓ Compatible |
| Future catalogs | >10B | ✓ Scalable |

### Scaling Strategy
- **HEALPix tiling**: Parallel by geographic region
- **Streaming I/O**: Process chunks without full load
- **Spatial indexing**: O(N log M) instead of O(N×M)
- **GPU acceleration**: Compute-intensive operations
- **Checkpoints**: Resume capability for long runs

## Extension Points

### Adding New Catalogs
1. Implement download script in `src/tasni/download/`
2. Add crossmatch script in `src/tasni/crossmatch/`
3. Update scoring in `src/tasni/filtering/multi_wavelength_scoring.py`
4. Add catalog info to `src/tasni/core/config.py`

### Adding New Filters
1. Implement filter in `src/tasni/filtering/`
2. Update pipeline in `src/tasni/optimized/optimized_pipeline.py`
3. Add tests in `tests/`

### Adding New Analysis
1. Implement analysis in `src/tasni/analysis/`
2. Generate output in `src/tasni/generation/`
3. Update documentation

## Security Considerations

### Data Privacy
- No personal data processed
- Catalog data is public domain
- Checkpoints contain no sensitive information

### API Security
- TAP services use read-only access
- No authentication required for public catalogs
- Rate limiting implemented in downloader

### Code Security
- No hardcoded credentials
- Environment-based configuration
- Pre-commit security checks
- Bandit static analysis

## Monitoring

### Logging
- Structured logging with python's logging module
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log rotation: 7 days retention, 30 day compression
- Log files: `logs/*.log`

### Progress Tracking
- HEALPix tile progress: `data/interim/checkpoints/download_*.json`
- Variability progress: `data/interim/checkpoints/tier5_variability_checkpoint.json`
- Status command: `python src/tasni/optimized_pipeline.py --status`

### Performance Monitoring
- Timing for each stage
- Memory usage tracking
- GPU utilization (when available)
- Benchmark mode: `--benchmark`

## Maintenance

### Regular Tasks
- **Daily**: Check pipeline status, review logs
- **Weekly**: Run data cleanup (`make data-cleanup`)
- **Monthly**: Review archive, compress old logs
- **Quarterly**: Update dependencies, review security

### Cleanup Procedures
- **Log rotation**: Automatic after 30 days
- **Archive cleanup**: Remove intermediate files older than 90 days
- **Cache cleanup**: Remove Python cache, numba cache
- **Data manifest**: Generate with `make data-manifest`

### Backup Strategy
- Git repository: GitHub (code, documentation)
- Data catalogs: External HDD (offline backup)
- Golden targets: Cloud storage (research backup)
- Paper materials: Multiple locations

---

**For detailed implementation guides, see:**
- `docs/pipeline.md` - Pipeline execution guide
- `docs/DATA_SOURCES.md` - Catalog documentation
- `docs/OPTIMIZATIONS.md` - Performance optimization guide
