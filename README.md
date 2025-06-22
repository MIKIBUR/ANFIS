# ANFIS Implementation

A custom implementation of Adaptive Neuro-Fuzzy Inference System (ANFIS) in Python.

## Overview

ANFIS (Adaptive Neuro-Fuzzy Inference System) is a hybrid artificial intelligence system that combines the benefits of neural networks and fuzzy logic. This implementation provides a complete ANFIS system capable of learning fuzzy rules from training data using a hybrid learning algorithm that combines least squares estimation (LSE) with backpropagation.

## Features

- **Complete ANFIS architecture** with all 5 layers implemented
- **Hybrid learning algorithm** combining LSE and backpropagation
- **Gaussian membership functions** with automatic parameter optimization
- **Interactive CLI interface** for testing trained models
- **Comprehensive visualization** of membership functions and training progress
- **Flexible parameterization** for different datasets and use cases

## Project Structure

```
anfis/
├── anfis_core.py       # Core ANFIS implementation
├── fuzzy_logic.py      # Fuzzy logic operations
├── data_loader.py      # Data handling and control functions
└── data.csv           # Sample dataset
```

## Requirements

The project requires the following Python packages:
- `numpy` - For numerical computations and matrix operations
- `pandas` - For data loading and manipulation
- `matplotlib` - For plotting membership functions and training progress
- `scikit-learn` - For data preprocessing and evaluation metrics

Install them using:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage Example

The project includes a `main.py` script that provides a complete ANFIS training and evaluation pipeline. Run it from the command line with various parameters:

### Basic Usage
```bash
python main.py
```

### Command Line Arguments

The script accepts the following arguments:

- `--rows` (int, default=200): Number of rows to load from the dataset
- `--epochs` (int, default=5): Number of training epochs
- `--decision` (str, default="h1"): Target variable selection
  - `h1`: loan_amount
  - `h2`: monthly_payment  
  - `h3`: years_to_pay
- `--rate` (float, default=0.01): Learning rate factor for backpropagation
- `--gamma` (int, default=300): Initial gamma parameter for LSE covariance matrix

### Example Commands

```bash
# Run with default parameters (200 rows, 5 epochs, loan_amount target)
python main.py

# Train for more epochs with larger dataset
python main.py --rows 500 --epochs 20

# Predict monthly payment instead of loan amount
python main.py --decision h2 --epochs 10

# Experiment with different learning parameters
python main.py --rate 0.05 --gamma 500 --epochs 15

# Full training run with years_to_pay prediction
python main.py --decision h3 --rows 1000 --epochs 25 --rate 0.02
```

### Input Features

The script uses the following input features for all prediction tasks:
- `person_age`: Age of the loan applicant
- `person_income`: Annual income
- `person_emp_exp`: Employment experience in years
- `cb_person_cred_hist_length`: Credit history length
- `loan_int_rate`: Loan interest rate

### Output

The script will:
1. Load and preprocess the specified amount of data
2. Generate initial Gaussian membership functions
3. Train the ANFIS model for the specified number of epochs
4. Display training progress and final evaluation metrics
5. Show visualizations of membership functions and training errors
6. Launch an interactive CLI for manual testing (optional)

### Dataset Requirements

Ensure your `data.csv` file contains all the required columns mentioned above. The script will automatically handle data splitting and normalization.
