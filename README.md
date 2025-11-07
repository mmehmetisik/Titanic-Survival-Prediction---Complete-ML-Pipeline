# Titanic Survival Prediction - Complete ML Pipeline

A comprehensive machine learning project implementing end-to-end workflow for Kaggle's Titanic competition. This repository contains a detailed 34-step approach covering exploratory data analysis, feature engineering, model optimization, and deployment.

## Project Overview

This project demonstrates a complete machine learning pipeline with advanced techniques including SHAP analysis, ablation testing, and hyperparameter optimization using both GridSearchCV and Optuna.

**Final Performance:**
- Cross-Validation Accuracy: 84.5%
- ROC-AUC Score: 0.89
- Features Used: 29 (from original 12)

## Table of Contents

### Data Preparation & Analysis (Sections 1-14)

1. **Importing Required Libraries** - Setting up the project environment
2. **Display Settings Configuration** - Configuring pandas and matplotlib settings
3. **Loading Datasets** - Loading train, test, and submission data
4. **Exploratory Data Analysis (EDA)** - Initial data exploration and statistics
5. **Identifying Variable Types** - Categorizing numerical and categorical variables
6. **Categorical Variable Analysis** - Analyzing categorical features
7. **Numerical Variable Analysis** - Analyzing numerical features
8. **Target vs Categorical Variables** - Relationship between target and categorical features
9. **Target vs Numerical Variables** - Relationship between target and numerical features
10. **Correlation Analysis (Raw Data)** - Initial correlation matrix analysis
11. **Missing Value Analysis** - Identifying and handling missing data
12. **Outlier Detection** - Detecting outliers using IQR method
13. **Logarithmic Transformation** - Skewness analysis and log transformations
14. **Rare Category Encoding** - Handling rare categories in categorical variables

### Feature Engineering & Encoding (Sections 15-19)

15. **Initial Encoding** - First pass at encoding categorical variables
16. **Initial Standardization** - Standardizing numerical features
17. **Base Model Training** - Training baseline models for comparison
18. **Feature Extraction** - Creating new features:
    - Family-related features (FamilySize, IsAlone, FamilyType)
    - Title extraction from names
    - Age groups and categories
    - Fare categories and per-person fare
    - Combination features (WomenChildrenFirst, HighStatus, LowStatus)
    - Name-based features
19. **Encoding New Features** - Encoding newly created features

### Advanced Analysis (Sections 20-27)

20. **Standardization (New Features)** - Standardizing with train/test split
21. **Model Training with New Features** - Training models with engineered features
22. **Base vs Advanced Model Comparison** - Comparing model performance
23. **Feature Importance Analysis (Random Forest)** - Built-in feature importance
24. **SHAP Analysis** - Model interpretability using SHAP values
25. **Correlation Analysis (New Features)** - Updated correlation analysis
26. **Removing Highly Correlated Features** - Hybrid approach (manual + automatic)
27. **Feature Selection** - Selecting top features based on importance

### Model Optimization & Deployment (Sections 28-34)

28. **Ablation Testing** - Testing true feature importance by removal
29. **Cross-Validation Strategy Comparison** - Comparing different CV strategies:
    - Standard K-Fold
    - Stratified K-Fold (5-fold)
    - Stratified K-Fold (10-fold)
    - Repeated Stratified K-Fold
30. **Hyperparameter Optimization** - Optimizing models using:
    - GridSearchCV (exhaustive search)
    - Optuna (Bayesian optimization)
31. **Final Model Evaluation** - Comprehensive evaluation of best model
32. **Base vs Final Model Comparison** - Complete performance comparison
33. **Test Set Predictions** - Making predictions on test data
34. **Kaggle Submission** - Creating submission file

## Key Features

### Advanced Techniques Used

- **Feature Engineering**: Created 18+ new features from original data
- **SHAP Analysis**: Model interpretability and feature impact analysis
- **Ablation Testing**: Validated feature importance through systematic removal
- **Hybrid Feature Selection**: Combined importance-based and correlation-based methods
- **Multiple Optimization Methods**: Compared GridSearchCV vs Optuna
- **CV Strategy Analysis**: Evaluated 4 different cross-validation approaches

### Technologies & Libraries
```python
# Core Libraries
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

# Advanced Libraries
- SHAP (model interpretability)
- Optuna (hyperparameter optimization)
- XGBoost, LightGBM
```

## Project Structure
```
titanic-ml-pipeline/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── notebooks/
│   └── titanic_complete_pipeline.ipynb
│
├── models/
│   └── final_model.pkl
│
├── submissions/
│   └── titanic_submission.csv
│
└── README.md
```

## Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/titanic-ml-pipeline.git

# Install required packages
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/titanic_complete_pipeline.ipynb
```

## Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
lightgbm>=3.3.0
optuna>=3.0.0
shap>=0.41.0
```

## Results Summary

| Model | Method | CV Accuracy | Features |
|-------|--------|-------------|----------|
| Base Model | Random Forest | 82.0% | 16 |
| Final Model | Random Forest (Optuna) | 84.5% | 29 |

**Improvement: +2.5% accuracy**

## Key Insights

1. **Feature Engineering Impact**: Creating family-related and title-based features significantly improved model performance
2. **Feature Selection**: Reducing from 73 to 29 features improved both performance and training speed
3. **CV Strategy**: Stratified K-Fold provided most reliable results for this imbalanced dataset
4. **Optimization**: Optuna achieved similar results to GridSearchCV in 60% less time
5. **Model Interpretability**: SHAP analysis revealed that title (Mr./Mrs./Miss) and fare were the strongest predictors

## Lessons Learned

- Ablation testing revealed 3 features that actually hurt model performance
- Cross-validation strategy matters - Standard K-Fold gave misleading results
- Feature engineering provides more value than hyperparameter tuning
- SHAP analysis and feature importance showed consistent results, validating our approach

## Future Improvements

- [ ] Ensemble methods (stacking multiple models)
- [ ] Deep learning approach with embeddings
- [ ] Additional feature engineering based on domain knowledge
- [ ] Automated feature engineering using tools like Featuretools
- [ ] MLOps pipeline for automated retraining

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the Titanic dataset
- The scikit-learn and SHAP communities for excellent documentation
- Optuna team for the powerful optimization framework

## Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Kaggle: [@yourusername](https://www.kaggle.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

## References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

⭐ If you found this project helpful, please consider giving it a star!
