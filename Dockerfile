# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install uv package manager for faster dependency installation
RUN pip install --upgrade pip && \
    pip install uv

# Install Python dependencies using uv
RUN uv pip install --system --no-cache \
    dash>=3.2.0 \
    joblib>=1.5.2 \
    matplotlib>=3.10.7 \
    numpy>=2.3.4 \
    pandas>=2.3.3 \
    plotly>=6.4.0 \
    psutil>=7.1.2 \
    scikit-learn>=1.7.2 \
    scipy>=1.16.2 \
    seaborn>=0.13.2 \
    statsmodels>=0.14.5 \
    tqdm>=4.67.1 \
    xgboost>=3.1.1 \
    dash-bootstrap-components

# Install PyTorch with CUDA 13.0 support (CPU version for lighter image)
# For GPU support, uncomment the line below and comment out the CPU version
# RUN uv pip install --system --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
RUN uv pip install --system --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY app.py model_inference_example.py ./
COPY models/ ./models/

# Note: datasets/ directory is mounted as a volume in docker-compose.yml
# This keeps the Docker image small and allows easy data updates

# Expose the port that the app runs on
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Run the application
CMD ["python", "app.py"]
