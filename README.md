# Gold Price Forecasting Using Historical Data

## Project Overview
Machine Learning project for predicting gold prices using historical data and various regression models.

**Author:** Félix Jouary
**Specialization:** Financial Engineering
**Dataset:** [Kaggle Gold Price Dataset](https://www.kaggle.com/datasets/rizkykiky/gold-price-dataset)

## Project Structure

```
gold-price-forecasting/
│
├── data/
│   ├── raw/                    # Original, immutable data
│   └── processed/              # Cleaned and transformed data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_advanced_models.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/                   # Data loading and processing scripts
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/               # Feature engineering scripts
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                 # Model training and prediction
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/          # Visualization scripts
│       ├── __init__.py
│       └── visualize.py
│
├── reports/
│   ├── figures/                # Generated graphics and figures
│   └── latex/                  # LaTeX report files
│
├── tests/                      # Unit tests
│
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Business Case
[To be completed - Define the problem, objectives, and link to financial engineering]

## Methodology
1. Data Exploration & Analysis
2. Data Preprocessing
3. Feature Engineering
4. Baseline Models Implementation
5. Advanced Models & Hyperparameter Tuning
6. Ensemble Methods
7. Model Comparison & Evaluation

## Models Implemented
- [ ] Linear Regression (Baseline)
- [ ] Ridge/Lasso Regression
- [ ] Decision Tree Regressor
- [ ] Random Forest Regressor
- [ ] Gradient Boosting (XGBoost/LightGBM)
- [ ] Support Vector Regression
- [ ] LSTM Neural Network

## Results
[To be completed after experimentation]

## References
[Scientific papers and external sources]
