.PHONY: help install install-dev test test-unit test-integration test-coverage \
        lint format clean clean-cache clean-archive clean-all compress-logs \
        docs paper pipeline-status run-pipeline golden-targets figures \
        data-cleanup data-manifest data-cleanup-dry security-audit \
        benchmark profile docker-build docker-run pre-commit

# Python interpreter for tests (bge-env has the required dependencies)
PYTEST := PYTHONPATH="" VIRTUAL_ENV="" /home/server/miniconda3/envs/bge-env/bin/python -m pytest

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-25s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# INSTALLATION
# =============================================================================

install: ## Install production dependencies
	pip install poetry
	poetry install --no-interaction
	@echo "Production dependencies installed"

install-dev: ## Install development dependencies
	pip install poetry
	poetry install --no-interaction --with dev
	pre-commit install
	@echo "Development dependencies installed"
	@echo "Pre-commit hooks installed"

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all tests
	$(PYTEST) tests/ -v --cov=src/tasni --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v

test-coverage: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=src/tasni --cov-report=html --cov-report=xml
	@echo "Coverage report: htmlcov/index.html"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run linters (black, isort, flake8)
	@echo "Running black..."
	black --check src/tasni/ tests/
	@echo "Running isort..."
	isort --check-only src/tasni/ tests/
	@echo "Running flake8..."
	flake8 src/tasni/ tests/ --max-line-length=100
	@echo "✓ Linting passed"

format: ## Format code with black and isort
	@echo "Formatting with black..."
	black src/tasni/ tests/
	@echo "Sorting imports with isort..."
	isort src/tasni/ tests/
	@echo "✓ Code formatted"

pre-commit: ## Run all pre-commit checks
	@echo "Running pre-commit..."
	pre-commit run --all-files

# =============================================================================
# CLEANUP
# =============================================================================

clean: ## Clean Python cache and temporary files
	@echo "Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.nbc" -delete 2>/dev/null || true
	find . -name "*.nbi" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*.so" -delete 2>/dev/null || true
	@echo "✓ Python cache cleaned"

clean-cache: ## Clean additional cache files
	@echo "Cleaning cache files..."
	rm -rf .pytest_cache/ .coverage htmlcov/ 2>/dev/null || true
	rm -rf .mypy_cache/ 2>/dev/null || true
	rm -rf *.egg-info/ build/ dist/ 2>/dev/null || true
	@echo "✓ Additional cache cleaned"

clean-archive: ## Clean archive directory (USE WITH CAUTION!)
	@echo "⚠️  This will delete intermediate files in archive/"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
		rm -f archive/*_progress_batch*.parquet; \
		rm -f archive/anomalies_*.parquet; \
		rm -f archive/tier2_*.parquet archive/tier3_*.parquet; \
		rm -f archive/*_old.parquet; \
		echo "✓ Archive cleaned"; \
	else \
		echo "Aborted"; \
	fi

clean-all: clean clean-cache clean-archive ## Run all cleanup operations
	@echo "✓ All cleanup complete"

compress-logs: ## Compress logs older than 30 days
	@echo "Compressing old logs..."
	find logs/ -name "*.log" -mtime +30 -exec gzip {} \; 2>/dev/null || true
	@echo "✓ Old logs compressed"

# =============================================================================
# PIPELINE OPERATIONS
# =============================================================================

pipeline-status: ## Check pipeline status
	python src/tasni/optimized/optimized_pipeline.py --status

run-pipeline: ## Run full pipeline (CPU)
	python src/tasni/optimized/optimized_pipeline.py --phase all --workers 16

run-pipeline-gpu: ## Run full pipeline (GPU)
	python src/tasni/optimized/optimized_pipeline.py --phase all --workers 16 --gpu

golden-targets: ## Generate golden target list
	python src/tasni/generation/generate_golden_list.py
	@echo "✓ Golden targets generated: data/processed/final/golden_improved.parquet"

figures: ## Generate publication figures
	python src/tasni/generation/generate_publication_figures.py
	@echo "✓ Figures generated: reports/figures/"

variability: ## Run variability analysis
	python src/tasni/analysis/compute_ir_variability.py
	@echo "✓ Variability analysis complete"

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

data-cleanup: ## Run data cleanup cycle
	python src/tasni/utils/data_manager.py
	@echo "✓ Data cleanup complete"

data-manifest: ## Generate data manifest
	python src/tasni/utils/data_manager.py --manifest
	@echo "✓ Data manifest generated"

data-cleanup-dry: ## Preview data cleanup without making changes
	python src/tasni/utils/data_manager.py --dry-run

# =============================================================================
# SECURITY
# =============================================================================

security-audit: ## Run security audit
	python src/tasni/utils/security_audit.py

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Build documentation
	@echo "Documentation available in docs/ directory"
	@ls -lh docs/*.md

paper: ## Print LaTeX compilation instructions
	@echo "LaTeX compilation (texlive not installed in this environment):"
	@echo "  cd tasni_paper_final && latexmk -pdf manuscript.tex"
	@echo ""
	@echo "Or manually:"
	@echo "  cd tasni_paper_final"
	@echo "  pdflatex manuscript.tex"
	@echo "  bibtex manuscript"
	@echo "  pdflatex manuscript.tex"
	@echo "  pdflatex manuscript.tex"

# =============================================================================
# PROFILING & BENCHMARKING
# =============================================================================

benchmark: ## Run pipeline benchmarks
	python src/tasni/optimized/optimized_pipeline.py --benchmark

profile: ## Profile crossmatch performance
	python -m cProfile -o profile.stats src/tasni/crossmatch/crossmatch_full.py --hpix 1234
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build Docker image
	docker build -t tasni:latest .
	@echo "✓ Docker image built: tasni:latest"

docker-run: ## Run Docker container
	docker run -v /mnt/data:/data tasni:latest python -c "print('TASNI container ready')"

docker-shell: ## Open shell in Docker container
	docker run -it -v /mnt/data:/data tasni:latest /bin/bash

# =============================================================================
# QUICK COMMANDS
# =============================================================================

download-wise: ## Download WISE catalog
	python src/tasni/download/download_wise_full.py

download-gaia: ## Download Gaia catalog
	python src/tasni/download/download_gaia_full.py --parallel

crossmatch-cpu: ## Run crossmatch (CPU)
	python src/tasni/crossmatch/crossmatch_full.py --workers 16

crossmatch-gpu: ## Run crossmatch (GPU)
	python src/tasni/crossmatch/gpu_crossmatch.py --batch-size 100000

filter-anomalies: ## Filter anomalies
	python src/tasni/filtering/filter_anomalies_full.py

analyze-variability: ## Analyze variability
	python src/tasni/analysis/compute_ir_variability.py

prepare-spectroscopy: ## Prepare spectroscopy targets
	python src/tasni/generation/prepare_spectroscopy_targets.py
# TASNI Machine Learning Targets

.PHONY: ml-features ml-train ml-predict ml-all ml-clean

ml-features: ## Extract features from tier5 sources
	python src/tasni/ml/extract_features.py \
		--tier5 data/processed/final/tier5_radio_silent.parquet \
		--neowise data/processed/final/neowise_epochs.parquet \
		--output data/processed/features/tier5_features.parquet
	@echo "✓ Feature extraction complete"

ml-train: ml-features ## Train ML models (requires features)
	python src/tasni/ml/train_classifier.py \
		--features data/processed/features/tier5_features.parquet \
		--golden data/processed/final/golden_improved.csv \
		--output data/processed/ml/models/ \
		--train
	@echo "✓ ML training complete"

ml-predict: ## Predict scores for tier5 sources (requires trained models)
	python src/tasni/ml/predict_tier5.py \
		--features data/processed/features/tier5_features.parquet \
		--models data/processed/ml/models/ \
		--output data/processed/ml/ranked_tier5.parquet \
		--top 10000
	@echo "✓ ML prediction complete"

ml-all: ml-features ml-train ml-predict ## Run complete ML pipeline
	@echo "✓ Complete ML pipeline finished"

ml-clean: ## Clean ML outputs
	rm -rf data/processed/features/*
	rm -rf data/processed/ml/models/*
	rm -rf data/processed/ml/results/*
	@echo "✓ ML outputs cleaned"

ml-top: ## Show top ML candidates (prints head of ranked output)
	@echo "Top 20 ML-ranked candidates:"
	@head -21 data/processed/ml/ranked_tier5_top10000.csv 2>/dev/null || \
		echo "No ranked output found. Run 'make ml-predict' first."


# Analysis Commands
plan-spectroscopy: ## Plan spectroscopic observations
	python src/tasni/generation/prepare_spectroscopy_targets.py \
		--targets data/processed/final/golden_improved.csv \
		--output output/spectroscopy/
	@echo "✓ Spectroscopy planning complete"

visualize-lightcurve: ## Visualize light curve for a source
	python src/tasni/analysis/light_curve_visualizer.py \
		--designation $(DESIGNATION) \
		--neowise data/processed/final/neowise_epochs.parquet \
		--output reports/figures/light_curves/
	@echo "✓ Light curve visualization complete"

# Makefile commands
