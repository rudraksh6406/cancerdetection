# Cancer Patient Survival Prediction

[![Build Status](https://img.shields.io/github/workflow/status/yourusername/cancerpatients/CI)](https://github.com/yourusername/cancerpatients/actions)
[![Code Quality](https://img.shields.io/github/workflow/status/yourusername/cancerpatients/Code%20Quality)](https://github.com/yourusername/cancerpatients/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview
This project predicts the survival of lung cancer patients using a machine learning model. It leverages clinical and demographic data to provide accurate survival predictions, assisting healthcare professionals in decision-making.

## Features
- 🏥 **Medical Data Processing**: Handles complex clinical datasets with missing values
- 🤖 **Machine Learning**: Random Forest model with balanced class handling
- 📊 **Data Visualization**: Built-in plotting and analysis capabilities
- 🧪 **Comprehensive Testing**: Unit and integration tests with coverage reporting
- 🐳 **Docker Support**: Containerized deployment for consistency
- 🔄 **CI/CD Pipeline**: Automated testing and code quality checks
- 📚 **Documentation**: Complete API documentation and examples

## Dataset
- Place your dataset as `data/dataset_med.csv`.
- The dataset should include columns like: `age`, `gender`, `country`, `diagnosis_date`, `cancer_stage`, `family_history`, `smoking_status`, `bmi`, `cholesterol_level`, `hypertension`, `asthma`, `cirrhosis`, `other_cancer`, `treatment_type`, `end_treatment_date`, and `survived`.

## Quick Start

### Using Docker (Recommended)
```bash
# Build and run with Docker
docker build -t cancer-prediction .
docker run -v $(pwd)/data:/app/data cancer-prediction

# Or use docker-compose
docker-compose up
```

### Using Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

## Installation

### Standard Installation
```bash
git clone https://github.com/yourusername/cancerpatients.git
cd cancerpatients
pip install -r requirements.txt
```

### Development Installation
```bash
git clone https://github.com/yourusername/cancerpatients.git
cd cancerpatients
pip install -e .
pip install -r requirements.txt
```

### Using Makefile
```bash
make install    # Install dependencies
make test       # Run tests
make lint       # Run linting
make format     # Format code
make help       # Show all available commands
```

## Usage

### Basic Usage
```python
from src.preprocess import load_and_preprocess
from src.model import train_and_save_model

# Load and preprocess data
df = load_and_preprocess('data/dataset_med.csv')

# Train model
model = train_and_save_model(X_train, y_train)
```

### Command Line
```bash
# Run main pipeline
python main.py

# Run example script
python examples/basic_usage.py

# Using the installed command
cancer-prediction
```

**Example Output:**
```
2024-01-XX 10:30:15,123 - INFO - Loading data from data/dataset_med.csv
2024-01-XX 10:30:15,456 - INFO - Preprocessing complete.
2024-01-XX 10:30:16,789 - INFO - Training Random Forest model...
2024-01-XX 10:30:17,012 - INFO - Model saved to model.joblib
2024-01-XX 10:30:17,234 - INFO - Accuracy: 0.8542
```

## Development

### Running Tests
```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
python -m pytest tests/test_integration.py -v
```

### Code Quality
```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Run all quality checks
make all
```

### Docker Development
```bash
# Build image
make docker-build

# Run container
make docker-run

# Development with Jupyter
docker-compose --profile dev up
```

## Project Structure
```
cancerpatients/
├── src/                    # Source code
│   ├── preprocess.py      # Data preprocessing
│   └── model.py           # Model definition and training
├── tests/                 # Test suite
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_integration.py
├── examples/              # Usage examples
│   └── basic_usage.py
├── docs/                  # Documentation
├── data/                  # Dataset directory
├── main.py               # Main application
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── pyproject.toml       # Modern Python packaging
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker services
├── Makefile             # Development commands
├── .github/             # GitHub Actions workflows
├── README.md            # Project documentation
├── CHANGELOG.md         # Version history
└── LICENSE              # MIT License
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, please contact [Your Name](mailto:your.email@example.com).

---

*This project is maintained with care and professionalism. If you use it in your research or work, please consider citing or starring the repository.*