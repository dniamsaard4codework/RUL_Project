# Battery Remaining Useful Life (RUL) Prediction

Link: [http://ec2-13-223-89-169.compute-1.amazonaws.com:8050/]

A comprehensive machine learning project for predicting battery Remaining Useful Life (RUL) using advanced modeling techniques, cross-dataset transfer learning, and production-ready web deployment.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/dniamsaard4codework/battery-rul-predictor)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen.svg)](.github/workflows/test-and-deploy.yml)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Web Application](#-web-application)
- [Docker Deployment](#-docker-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

## 📋 Project Overview

This project implements state-of-the-art machine learning models to predict battery RUL across different datasets with a user-friendly web interface:

### Models
- **Zenodo Dataset Model**: General-purpose RUL prediction for battery lifecycle analysis
- **NASA Dataset Model**: Cross-dataset transfer learning with fine-tuning for aerospace applications

### Key Capabilities
- 🔋 **Predict battery lifespan** based on charge/discharge patterns
- 📊 **Interactive web dashboard** with 16 different usage protocol simulations
- 🔄 **Transfer learning** that requires only 15% of target domain data
- 🐳 **Production-ready deployment** with Docker and CI/CD pipeline
- 📈 **Real-time predictions** with uncertainty estimates

## 🎯 Features

### Comprehensive Model Analysis
- ✅ **5-Fold Cross-Validation** with statistical significance testing
- ✅ **Hyperparameter Tuning** using RandomizedSearchCV
- ✅ **Learning Curves** for overfitting detection
- ✅ **Feature Importance** and Partial Dependence Plots
- ✅ **Model Ensemble** methods (simple & weighted averaging)

### Transfer Learning & Fine-tuning
- ✅ **Cross-Dataset Adaptation**: Zenodo → NASA
- ✅ **Distribution Shift Analysis**: KS tests, Wasserstein distance
- ✅ **Fine-tuning Strategy**: Only 15% NASA data needed
- ✅ **Performance Improvement**: R² from -4.56 → 0.94

### Production-Ready Deployment
- ✅ **Web Application**: Interactive Dash-based dashboard
- ✅ **Docker Containerization**: One-command deployment
- ✅ **CI/CD Pipeline**: Automated testing and deployment
- ✅ **Model Export**: Joblib serialization with metadata
- ✅ **Prediction Uncertainty**: 95% confidence intervals
- ✅ **Business Metrics**: Cost analysis, false alarm rates

### Explainability & Interpretability
- ✅ **Statistical Testing**: Paired t-tests, Cohen's d
- ✅ **Residual Analysis**: Q-Q plots, normality tests
- ✅ **RUL Segment Analysis**: Performance by lifecycle stage
- ✅ **Feature Correlation**: Cross-dataset comparison

## � Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull and run the pre-built image with models included
docker pull dniamsaard4codework/battery-rul-predictor:latest
docker run -p 8050:8050 dniamsaard4codework/battery-rul-predictor:latest
```

Visit `http://localhost:8050` in your browser.

### Option 2: Docker Compose

```bash
# Clone the repository
git clone https://github.com/dniamsaard4codework/RUL_Project.git
cd RUL_Project

# Run with docker-compose
docker-compose up
```

### Option 3: Local Python Environment

```bash
# Install dependencies
pip install uv
uv pip install -e .

# Run the web application
python app.py
```

### Option 4: Use Pre-trained Models in Python

```python
from model_inference_example import BatteryRULPredictor

# For Zenodo batteries
predictor = BatteryRULPredictor(model_type='zenodo')
rul_predictions = predictor.predict(your_battery_data)

# For NASA batteries
predictor = BatteryRULPredictor(model_type='nasa')
rul_predictions = predictor.predict(your_battery_data)
```

## 💻 Installation

### Requirements
- Python 3.13+
- Docker (optional, for containerized deployment)
- Git LFS (for model files)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/dniamsaard4codework/RUL_Project.git
cd RUL_Project
```

2. **Install uv package manager**
```bash
pip install uv
```

3. **Install dependencies**
```bash
uv pip install -e .
```

4. **Get model files** (choose one method)

   **Method A: Pull from Docker Hub**
   ```bash
   docker pull dniamsaard4codework/battery-rul-predictor:latest
   docker create --name temp-container dniamsaard4codework/battery-rul-predictor:latest
   docker cp temp-container:/app/models ./models
   docker rm temp-container
   ```

   **Method B: Train models locally**
   ```bash
   jupyter notebook notebook/modelling.ipynb
   jupyter notebook notebook/ML-cross-dataset-v5.ipynb
   ```

## 📖 Usage

### Web Application

Run the interactive dashboard:

```bash
python app.py
```

Then open `http://localhost:8050` in your browser.

**Features:**
- Upload CSV files with battery data
- Select from 16 different usage protocols
- View RUL predictions with confidence intervals
- Interactive data visualization
- Protocol-specific insights

### Command-Line Prediction

```python
import pandas as pd
from model_inference_example import BatteryRULPredictor

# Initialize predictor
predictor = BatteryRULPredictor(model_type='nasa')

# Load your battery data
battery_data = pd.read_csv('your_battery_data.csv')

# Predict RUL
rul = predictor.predict(battery_data)
print(f"Predicted RUL: {rul}")
```

### Prediction with Uncertainty

```python
# Get predictions with confidence intervals
predictions, std = predictor.predict_with_uncertainty(battery_data)

for i, (pred, uncertainty) in enumerate(zip(predictions, std)):
    print(f"Battery {i+1}: RUL = {pred:.2f} ± {uncertainty:.2f}")
```

### Feature Importance

```python
# Get top important features
importance = predictor.get_feature_importance(top_n=10)
print(importance)
```

### Example with Real Data

```python
# Load example data
example_data = pd.read_csv('example_input/zenodo_battery_1_up_to_200_cycles.csv')

# Predict
predictions = predictor.predict(example_data)
print(f"RUL predictions: {predictions[:5]}")  # First 5 predictions
```

## 📁 Project Structure

```
RUL_Project/
├── app.py                          # Dash web application
├── model_inference_example.py      # Easy-to-use prediction interface
├── Dockerfile                      # Docker container configuration
├── docker-compose.yml              # Local Docker Compose setup
├── docker-compose-server.yml       # Production Docker Compose setup
├── pyproject.toml                  # Python dependencies
├── .github/
│   └── workflows/
│       └── test-and-deploy.yml    # CI/CD pipeline
├── datasets/                       # Battery datasets
│   ├── battery_alt_dataset/       # NASA battery data
│   │   ├── recommissioned_batteries/
│   │   ├── regular_alt_batteries/
│   │   └── second_life_batteries/
│   ├── preprocessed/              # Preprocessed data
│   │   ├── final_df.csv
│   │   └── primary_use_phase_rul.csv
│   └── Primary_use_phase/         # Primary use phase data (88 CSV files)
├── example_input/                 # Example input data
│   └── zenodo_battery_1_up_to_200_cycles.csv
├── models/                        # Exported trained models
│   ├── zenodo_best_model_latest.pkl
│   ├── zenodo_preprocessor_latest.pkl
│   ├── zenodo_model_metadata_latest.json
│   ├── zenodo_feature_info_latest.json
│   ├── nasa_finetuned_model_latest.pkl
│   ├── nasa_model_metadata_latest.json
│   ├── nasa_feature_info_latest.json
│   └── nasa_stability_analysis_*.json
├── notebook/                      # Jupyter notebooks
│   ├── modelling.ipynb            # Main modeling (Zenodo)
│   ├── ML-cross-dataset-v5.ipynb  # Cross-dataset transfer learning
│   ├── feature_engineer.ipynb     # Feature engineering
│   └── data.ipynb                 # Data exploration
├── tests/                         # Test suite
│   ├── test_models.py             # Model testing
│   └── __init__.py
└── eda_full_advanced.ipynb        # Comprehensive EDA
```

## 📊 Model Performance

### Zenodo Model
- **R² Score**: 0.85+ (High confidence)
- **RMSE**: Low error on test set
- **Application**: General battery RUL prediction
- **Features**: 39 engineered features
- **Status**: ✅ Production-ready

### NASA Fine-tuned Model
- **R² Score (NASA)**: 0.94
- **R² Score (Zenodo)**: 0.85+ (maintained)
- **Key Achievement**: 500%+ improvement over baseline
- **Data Efficiency**: Only 15% of new domain data needed
- **Cost Savings**: 85% reduction in data collection
- **Status**: ✅ Production-ready

### Key Metrics Summary

| Metric | Zenodo Model | NASA Model |
|--------|-------------|------------|
| R² Score | 0.85+ | 0.94 |
| Training Data | Full dataset | 15% fine-tuning |
| Features | 39 | 10 (common) |
| Confidence | High | Very High |
| Use Case | General batteries | Aerospace batteries |

## 🌐 Web Application

### Features

The interactive web dashboard (`app.py`) provides:

1. **File Upload**: CSV upload with drag-and-drop support
2. **Protocol Selection**: 16 different battery usage patterns:
   - Regular charging and current (Protocol 1)
   - Normal use (Protocol 2)
   - Deep use / Full charge habit (Protocol 3)
   - Fast charge variations (Protocols 4-6)
   - Hard driving patterns (Protocols 7-8)
   - Combined stress scenarios (Protocols 9-13)
   - Light/moderate/stop-go use (Protocols 14-16)

3. **Interactive Visualizations**:
   - RUL predictions over battery cycles
   - Confidence intervals
   - Feature distributions
   - Protocol-specific insights

4. **Real-time Predictions**: Instant RUL calculation on upload

### Running the Web App

```bash
# Local
python app.py

# Docker
docker-compose up

# Access at http://localhost:8050
```

## 🐳 Docker Deployment

### Pre-built Image (Recommended)

The Docker image is available on Docker Hub with all models included:

```bash
# Pull the image
docker pull dniamsaard4codework/battery-rul-predictor:latest

# Run the container
docker run -p 8050:8050 dniamsaard4codework/battery-rul-predictor:latest
```

### Build Locally

```bash
# Build the image
docker build -t battery-rul-predictor:latest .

# Run with docker-compose
docker-compose up
```

### Production Deployment

For production servers:

```bash
# Use the server configuration
docker-compose -f docker-compose-server.yml up -d
```

Features:
- Auto-restart on failure
- Health checks every 30 seconds
- Optimized for production workloads
- Network isolation

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The project includes a complete CI/CD pipeline (`.github/workflows/test-and-deploy.yml`):

#### Test Job
1. Checkout code
2. Pull Docker image with pre-trained models
3. Extract models from Docker image
4. Set up Python 3.13 environment
5. Install dependencies with `uv`
6. Run pytest test suite

#### Deploy Job (on main branch)
1. Configure SSH for server access
2. Deploy to production server
3. Pull latest Docker image
4. Restart containers with docker-compose
5. Verify deployment with health check
6. Send notifications

### Required GitHub Secrets

Configure these in your repository settings:

```bash
SSH_PRIVATE_KEY    # SSH private key for server access
SERVER_HOST        # Server IP or hostname
SERVER_USER        # SSH username (e.g., ubuntu)
DEPLOY_PATH        # Deployment directory path
DOCKER_IMAGE       # Docker image name (e.g., dniamsaard4codework/battery-rul-predictor:latest)
```

### Manual Deployment

```bash
# On your server
cd /path/to/RUL_Project
git pull origin main
docker-compose -f docker-compose-server.yml pull
docker-compose -f docker-compose-server.yml up -d
```

## 🧪 Testing

### Run Tests Locally

```bash
# Install test dependencies
uv pip install pytest

# Run all tests
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::TestZenodoModel::test_model_loads -v
```

### Test Coverage

The test suite (`tests/test_models.py`) includes:

- ✅ Model loading verification
- ✅ Feature count validation
- ✅ Prediction accuracy tests
- ✅ Input validation
- ✅ Missing feature detection
- ✅ Output shape verification
- ✅ Uncertainty estimation tests
- ✅ Feature importance checks
- ✅ Metadata validation

### Note on Model Files

Models are too large for Git (>100MB). Use one of these methods:

1. **Git LFS**: Track with Git Large File Storage
2. **Docker Hub**: Models included in Docker image
3. **Local Training**: Train models using notebooks

## 📈 Business Impact

### Cost-Benefit Analysis
- **False Alarm Reduction**: Optimized threshold for minimal unnecessary maintenance
- **Missed Failure Prevention**: High recall for critical battery states
- **Operational Savings**: Estimated cost reduction through predictive maintenance
- **Data Efficiency**: 85% less data collection needed with transfer learning

### Deployment Metrics
- **Precision**: Accurate critical alerts
- **Recall**: Comprehensive failure detection
- **F1-Score**: Balanced performance
- **Confidence Level**: Automated readiness assessment
- **Prediction Speed**: Real-time inference (<100ms per battery)

## 📚 Documentation

### Main Documentation Files

- **`README.md`** (this file): Complete project overview and setup guide
- **`models/README.md`**: Detailed model usage and API documentation
- **`tests/README.md`**: Testing guidelines and test descriptions
- **Notebook Comments**: Inline documentation in all Jupyter notebooks

### Notebooks Documentation

1. **`notebook/modelling.ipynb`**
   - Zenodo dataset model development
   - Feature engineering pipeline
   - Hyperparameter tuning
   - Cross-validation analysis
   - Model export and evaluation

2. **`notebook/ML-cross-dataset-v5.ipynb`**
   - Cross-dataset transfer learning (Zenodo → NASA)
   - Distribution shift analysis
   - Fine-tuning strategies
   - Performance comparison
   - Stability analysis

3. **`notebook/feature_engineer.ipynb`**
   - Feature creation and selection
   - Rolling window features
   - Statistical feature engineering
   - Feature correlation analysis

4. **`notebook/data.ipynb`**
   - Exploratory data analysis
   - Data preprocessing
   - Battery lifecycle visualization
   - Dataset statistics

5. **`eda_full_advanced.ipynb`**
   - Comprehensive EDA
   - Advanced visualizations
   - Pattern discovery
   - Anomaly detection

### API Documentation

#### BatteryRULPredictor Class

```python
from model_inference_example import BatteryRULPredictor

# Initialize
predictor = BatteryRULPredictor(model_type='zenodo')  # or 'nasa'

# Methods
predictor.predict(X)                           # Predict RUL
predictor.predict_with_uncertainty(X)          # Predict with std
predictor.get_feature_importance(top_n=10)     # Feature importance
predictor.get_required_features()              # List required features
predictor.get_required_features_count()        # Count features
```

## 🔄 Model Versioning

Models are saved in two formats for easy deployment:

1. **Timestamped**: `model_YYYYMMDD_HHMMSS.pkl` (archive)
2. **Latest**: `model_latest.pkl` (production use)

### Model Files

Each model includes:
- ✅ Trained model object (`.pkl`)
- ✅ Preprocessor/pipeline (`.pkl`)
- ✅ Comprehensive metadata (`.json`)
- ✅ Feature information (`.json`)
- ✅ Performance metrics
- ✅ Training parameters

### Example: Zenodo Model Files
```
models/
├── zenodo_best_model_latest.pkl           # Model object
├── zenodo_preprocessor_latest.pkl         # Preprocessing pipeline
├── zenodo_model_metadata_latest.json      # Metrics & parameters
├── zenodo_feature_info_latest.json        # Feature names & types
├── zenodo_best_model_20251110_163047.pkl  # Timestamped backup
└── ...
```

## 🎓 Key Findings & Achievements

### Technical Achievements

1. **Transfer Learning Success**: Only 15% of target domain data needed for excellent performance
2. **Feature Engineering Impact**: Rolling window features significantly improve predictions
3. **Ensemble Benefits**: Weighted averaging provides robust predictions
4. **Cross-Dataset Generalization**: Models adapt well across different battery types

### Model Insights

1. **Most Important Features**:
   - Discharge capacity trends (max, mean, std)
   - Voltage characteristics (max, mean, std)
   - Rolling window statistics
   - State of Health (SoH) indicators
   - Temperature patterns

2. **Protocol Impact**:
   - Fast charging reduces battery lifespan
   - Deep discharge cycles accelerate degradation
   - Combined stress (fast + deep) has multiplicative effects
   - Regular moderate use maximizes battery life

3. **Prediction Accuracy**:
   - Early lifecycle (RUL > 500): ±50 cycles
   - Mid lifecycle (200 < RUL < 500): ±30 cycles
   - End of life (RUL < 200): ±10 cycles

## �️ Development Setup

### For Contributors

1. **Clone and setup**
```bash
git clone https://github.com/dniamsaard4codework/RUL_Project.git
cd RUL_Project
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

2. **Install development dependencies**
```bash
uv pip install pytest jupyterlab black flake8
```

3. **Run notebooks**
```bash
jupyter lab
```

4. **Run tests before commit**
```bash
pytest tests/test_models.py -v
```

### Code Quality

- **Formatting**: Follow PEP 8 style guide
- **Documentation**: Add docstrings to all functions
- **Testing**: Write tests for new features
- **Comments**: Explain complex logic

## 🔐 Security & Privacy

- **No sensitive data**: All datasets are public research data
- **Model artifacts**: Stored securely in Docker Hub
- **Server deployment**: SSH key authentication required
- **Environment variables**: Never commit credentials to Git

## 🚀 Future Enhancements

### Planned Features
- [ ] **SHAP values** for deeper model explainability
- [ ] **Online learning** for continuous model adaptation
- [ ] **Additional battery chemistries** (Li-ion variants, solid-state)
- [ ] **REST API service** with FastAPI
- [ ] **Real-time monitoring** dashboard for battery fleets
- [ ] **Mobile application** for field technicians
- [ ] **Anomaly detection** for unusual degradation patterns
- [ ] **Battery health scoring** system

### Research Directions
- [ ] **Attention mechanisms** for sequential battery data
- [ ] **Physics-informed neural networks** combining data and domain knowledge
- [ ] **Multi-modal learning** incorporating temperature, vibration, and usage data
- [ ] **Federated learning** for privacy-preserving model updates
- [ ] **Causal inference** for understanding degradation mechanisms

## ❓ Troubleshooting

### Common Issues

#### 1. Model files not found
```bash
# Solution: Pull models from Docker Hub
docker pull dniamsaard4codework/battery-rul-predictor:latest
docker create --name temp-container dniamsaard4codework/battery-rul-predictor:latest
docker cp temp-container:/app/models ./models
docker rm temp-container
```

#### 2. Missing features error
```python
# Solution: Check required features
predictor = BatteryRULPredictor(model_type='zenodo')
required_features = predictor.get_required_features()
print(f"Required features ({len(required_features)}):")
print(required_features)
```

#### 3. Docker build fails
```bash
# Solution: Use pre-built image instead
docker pull dniamsaard4codework/battery-rul-predictor:latest
docker run -p 8050:8050 dniamsaard4codework/battery-rul-predictor:latest
```

#### 4. Port 8050 already in use
```bash
# Solution: Use a different port
docker run -p 8051:8050 dniamsaard4codework/battery-rul-predictor:latest
# Access at http://localhost:8051
```

#### 5. Tests fail with "Model files not found"
```bash
# Solution: Extract models from Docker image first (see issue #1)
# Or train models locally using the notebooks
```

## 📊 Datasets

### Zenodo Battery Dataset
- **Source**: Public research dataset
- **Batteries**: 88 batteries in primary use phase
- **Cycles**: Variable (50-1000+ cycles per battery)
- **Features**: Voltage, current, capacity, temperature, energy
- **Use case**: General battery lifecycle analysis

### NASA Battery Dataset
- **Source**: NASA Prognostics Center of Excellence
- **Categories**:
  - Recommissioned batteries
  - Regular alt batteries
  - Second life batteries
- **Features**: Similar to Zenodo with aerospace-specific protocols
- **Use case**: Aerospace and high-reliability applications

### Preprocessed Datasets
- **`final_df.csv`**: Combined and cleaned data
- **`primary_use_phase_rul.csv`**: RUL labels for training

## 🏆 Performance Benchmarks

### Computational Performance

| Operation | Time | Hardware |
|-----------|------|----------|
| Model Loading | ~2s | CPU |
| Single Prediction | <10ms | CPU |
| Batch (1000 samples) | ~100ms | CPU |
| Training (Zenodo) | ~5 min | CPU |
| Fine-tuning (NASA) | ~2 min | CPU |

### Memory Usage
- **Model Size**: ~50MB per model
- **Runtime Memory**: ~200MB
- **Docker Image**: ~2.5GB (includes all dependencies)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

1. **Report bugs**: Open an issue with detailed description
2. **Suggest features**: Share your ideas in discussions
3. **Improve documentation**: Fix typos, add examples
4. **Submit code**: Create pull requests with new features
5. **Share datasets**: Help expand battery types coverage

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Follow coding standards
- Test your changes

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors & Acknowledgments

### Development Team
- Battery RUL Prediction Team

### Acknowledgments
- NASA Prognostics Center of Excellence for battery datasets
- Zenodo open science platform for dataset hosting
- Open source community for amazing tools and libraries

### Technologies Used
- **Python**: Core programming language
- **scikit-learn**: Machine learning framework
- **Dash/Plotly**: Interactive web visualizations
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline
- **PyTorch**: Deep learning (optional experiments)
- **Pandas/NumPy**: Data manipulation
- **Joblib**: Model serialization

---

**Project Status**: ✅ Production-Ready  
**Last Updated**: November 11, 2025  
**Documentation Version**: 1.0.0

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ by the Battery RUL Prediction Team

</div>
