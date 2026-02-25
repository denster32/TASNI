# TASNI Dockerfile
# Thermal Anomaly Search for Non-communicating Intelligence
#
# Build: docker build -t tasni .
# Run:   docker run -v /path/to/data:/data tasni python -m tasni

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libhdf5-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-interaction --no-ansi --without dev

# Copy application code
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/

# Create necessary directories
RUN mkdir -p /app/output /app/data /app/logs /app/checkpoints

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TASNI_DATA_ROOT=/data

# Default command
CMD ["python", "-c", "print('TASNI container ready. Run with: docker run -v /data:/data tasni python -m tasni')"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import astropy; import pandas; print('OK')" || exit 1
