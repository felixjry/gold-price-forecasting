# Gold Price Forecasting Using Machine Learning

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project Overview

Machine Learning project for predicting gold prices using historical data and various regression models. This project was developed as part of a Financial Engineering specialization, focusing on time series forecasting and model comparison.

**Author:** Félix Jouary
**Specialization:** Financial Engineering
**Dataset:** [Kaggle Gold Price Dataset](https://www.kaggle.com/datasets/rizkykiky/gold-price-dataset)

## Key Results

**Best Model: Lasso Regression**
- **RMSE:** $13.76
- **R² Score:** 0.9977 (99.77% variance explained)
- **MAPE:** 0.62%
- **Training Data:** 11,539 daily observations (80/20 split)

## Project Structure

```
gold-price-forecasting/
│
├── data/
│   ├── raw/                    # Original gold price data (CSV)
│   └── processed/              # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models.ipynb
│   └── 05_model_comparison.ipynb
│
├── models/                      # Saved model files (.pkl)
│
├── reports/
│   ├── figures/                 # Visualizations (PNG)
│   └── ML_Report_FelixJouary.tex
│
├── src/                         # Source code modules
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
│
├── requirements.txt
└── README.md
```

## Models Implemented

We implemented and compared **10 machine learning models**:

### Baseline Models
- Linear Regression (RMSE: $13.77, R²: 0.9977)
- Ridge Regression (RMSE: $13.77, R²: 0.9977)
- **Lasso Regression** (RMSE: $13.76, R²: 0.9977) ✅ **Best Model**
- Decision Tree (RMSE: $46.18, R²: 0.9745)
- Random Forest (RMSE: $50.76, R²: 0.9691)
- Gradient Boosting (RMSE: $60.87, R²: 0.9556)

### Advanced Models
- Support Vector Regression (RMSE: $120.22, R²: 0.8268)
- AdaBoost (RMSE: $53.12, R²: 0.9662)
- Extra Trees (RMSE: $64.08, R²: 0.9508)
- Bagging Regressor (RMSE: $51.39, R²: 0.9684)

## Feature Engineering

From the original price data, we engineered **28 features**:

1. **Temporal Features**: Year, Month, Day, DayOfWeek, Quarter, WeekOfYear
2. **Lag Features**: Price lags at 1, 2, 3, 5, 7, 14, 21, and 30 days
3. **Moving Averages**: 7-day, 14-day, 30-day, 60-day, 90-day MA
4. **Returns**: 1-day, 5-day, and 30-day percentage returns
5. **Volatility Indicators**: 7-day and 30-day rolling standard deviations
6. **Technical Indicators**: Price vs MA ratios, Rolling Min/Max

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gold-price-forecasting.git
cd gold-price-forecasting

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Notebooks

The analysis is divided into 5 Jupyter notebooks:

```bash
# 1. Explore the dataset
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Preprocess and engineer features
jupyter notebook notebooks/02_preprocessing.ipynb

# 3. Train baseline models
jupyter notebook notebooks/03_baseline_models.ipynb

# 4. Train advanced models
jupyter notebook notebooks/04_advanced_models.ipynb

# 5. Compare all models
jupyter notebook notebooks/05_model_comparison.ipynb
```

## Methodology

1. **Data Exploration**: Analyze gold price trends, volatility, and statistical properties
2. **Preprocessing**: Handle missing values, create temporal features
3. **Feature Engineering**: Generate lag features, moving averages, and technical indicators
4. **Model Training**: Train 10 different ML models with hyperparameter tuning (GridSearchCV)
5. **Evaluation**: Compare models using RMSE, MAE, MAPE, and R² metrics
6. **Overfitting Analysis**: Assess generalization using train/test performance ratios

## Key Findings

### Why Did Linear Models Win?

Surprisingly, simple regularized linear models outperformed complex ensemble methods:

- **Effective Feature Engineering**: The 28 engineered features captured the underlying patterns well
- **Overfitting in Tree Models**: Tree-based models showed severe overfitting (overfit ratios >10)
- **Linear Predictability**: Gold prices exhibit strong linear relationships with recent lags and moving averages

### Overfitting Analysis

| Model | Train RMSE | Test RMSE | Overfit Ratio |
|-------|-----------|-----------|---------------|
| Lasso Regression | $8.77 | $13.76 | 1.57 ✅ |
| Ridge Regression | $8.77 | $13.77 | 1.57 ✅ |
| Decision Tree | $3.05 | $46.18 | 15.14 ⚠️ |
| Random Forest | $3.99 | $50.76 | 12.73 ⚠️ |

An overfit ratio close to 1.0 indicates good generalization. Linear models achieved the best balance.

## Business Context

This project addresses gold price forecasting from a **financial engineering perspective**:

- **Portfolio Diversification**: Gold's negative correlation with equities makes it a key hedging asset
- **Inflation Protection**: Gold serves as a hedge against currency devaluation
- **Derivatives Pricing**: Accurate forecasts support options pricing and algorithmic trading
- **Risk Management**: Predictions enable VaR and Expected Shortfall calculations

## Technical Challenges Solved

1. **XGBoost/LightGBM OpenMP Conflicts on macOS**: Replaced with Extra Trees and Bagging
2. **SVR Computational Complexity**: Used subset sampling for hyperparameter search
3. **Time Series Cross-Validation**: Implemented TimeSeriesSplit to respect temporal ordering
4. **Overfitting Prevention**: Applied L1/L2 regularization and tree depth constraints

## Visualizations

The project includes comprehensive visualizations in `reports/figures/`:

- Model performance comparison
- Feature importance analysis
- Residual analysis
- Overfitting diagnostics
- Top model predictions vs actual prices

## Requirements

Main dependencies:
- Python 3.8+
- pandas 2.0+
- numpy 1.24+
- scikit-learn 1.3+
- matplotlib 3.7+
- seaborn 0.12+

See `requirements.txt` for complete list.

## Results Summary

| Metric | Best Model (Lasso) |
|--------|-------------------|
| RMSE | $13.76 |
| MAE | $9.33 |
| MAPE | 0.62% |
| R² Score | 0.9977 |

## Future Work

- Incorporate macroeconomic indicators (USD strength, inflation rates, interest rates)
- Implement regime-switching models for changing market conditions
- Explore deep learning approaches (LSTM, Transformer) with proper computational resources
- Extend to other precious metals (silver, platinum, palladium)

## Report

A comprehensive LaTeX report is available in `reports/ML_Report_FelixJouary.tex`, including:
- Business case and financial context
- Detailed methodology
- Mathematical model descriptions
- Results analysis and interpretation
- Obstacles encountered and solutions

## License

MIT License

## Contact

Félix Jouary - Financial Engineering Specialization

## Acknowledgments

- Dataset: [Kaggle Gold Price Dataset](https://www.kaggle.com/datasets/rizkykiky/gold-price-dataset)
- Course: Machine Learning Project 2025

