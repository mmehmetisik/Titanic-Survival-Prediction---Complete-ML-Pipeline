
############################################
# 1. Importing Required Libraries
############################################

# Data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Model selection and evaluation tools
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ParameterGrid

# Preprocessing utilities
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Pipeline tools for streamlined workflows
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Evaluation metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

# Machine learning models - Ensemble methods
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Machine learning models - Other algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Advanced gradient boosting models
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter optimization framework
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
warnings.filterwarnings('ignore')

############################################
# 2. Display Settings Configuration
############################################

# Pandas display settings for better data visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Visualization settings for consistent plot styling
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

###############################
# 3. Loading Datasets
###############################

# For Kaggle Notebook - Use these paths
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# For local machine - Uncomment these lines if running locally
"""
train_path = r"C:\Users\ASUS\Desktop\pythonProject\titanic\data\train.csv"
test_path = r"C:\Users\ASUS\Desktop\pythonProject\titanic\data\test.csv"
gender_submission_path = r"C:\Users\ASUS\Desktop\pythonProject\titanic\data\gender_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
gender_submission_df = pd.read_csv(gender_submission_path)
"""

# Display first 5 rows of each dataset
print("Training dataset - First 5 rows:")
print(train_df.head())

print("\nTest dataset - First 5 rows:")
print(test_df.head())

print("\nGender submission - First 5 rows:")
print(gender_submission_df.head())

# Dataset dimensions
print("\nTraining dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
print("Gender submission shape:", gender_submission_df.shape)

# Combining datasets for unified preprocessing
# 1. reset_index(drop=True) ensures we don't keep the old index as a column
# 2. Adding 'is_train' flag to track which data came from which source

train_df['is_train'] = 1
test_df['is_train'] = 0
df = pd.concat([train_df, test_df]).reset_index(drop=True)


############################################
# 4. Exploratory Data Analysis (EDA)
############################################

def check_df(dataframe, head=5, name=""):
    """
    Provides a comprehensive overview of the dataset including shape, types,
    head/tail views, missing values, and statistical summary.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to analyze
    head : int, optional
        Number of rows to display for head and tail (default is 5)
    name : str, optional
        Name identifier for the dataset (default is empty string)
    """
    print(f'##################### {name} Dataset Overview #####################')
    print('\n##################### Shape #####################')
    print(dataframe.shape)

    print('\n##################### Types #####################')
    print(dataframe.dtypes)

    print('\n##################### Head #####################')
    print(dataframe.head(head))

    print('\n##################### Tail #####################')
    print(dataframe.tail(head))

    print('\n##################### Missing Values #####################')
    print(dataframe.isnull().sum())

    print('\n##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# Perform comprehensive data check
check_df(df)

############################################
# 5. Identifying Numerical and Categorical Variables
############################################

# Drop columns that won't be used in modeling
drop_list = ["PassengerId", "Ticket"]
df.drop(drop_list, axis=1, inplace=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identifies and categorizes variables in the dataset as categorical, numerical,
    or cardinal (high cardinality categorical).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe from which to extract column names
    cat_th : int, float, optional
        Threshold for numerical variables to be classified as categorical (default is 10)
    car_th : int, float, optional
        Threshold for categorical variables to be classified as cardinal (default is 20)

    Returns
    -------
    cat_cols : list
        List of categorical variables
    num_cols : list
        List of numerical variables
    cat_but_car : list
        List of categorical variables with high cardinality
    num_but_cat : list
        List of numerical variables that are actually categorical
    """

    # Identify categorical columns based on data type
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # Identify numerical columns that are actually categorical (low unique values)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    # Identify categorical columns with high cardinality
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    # Create final categorical columns list
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Identify purely numerical columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(dataframe.head())
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical Columns: {len(cat_cols)}")
    print(cat_cols)
    print(f"Numerical Columns: {len(num_cols)}")
    print(num_cols)
    print(f"Categorical but Cardinal: {len(cat_but_car)}")
    print(cat_but_car)
    print(f"Numerical but Categorical: {len(num_but_cat)}")
    print(num_but_cat)

    return cat_cols, num_cols, cat_but_car, num_but_cat


# Categorize variables
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


############################
# 6. Analysis of Categorical Variables
############################

def cat_summary(dataframe, col_name, plot=False):
    """
    Provides a summary of categorical variable including value counts and ratios.
    Optionally displays a count plot.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the variable
    col_name : str
        Name of the categorical column to analyze
    plot : bool, optional
        Whether to display a count plot (default is False)
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


# Analyze all categorical variables
for col in cat_cols:
    cat_summary(df, col, plot=True)


############################
# 7. Analysis of Numerical Variables
############################

def num_summary(dataframe, numerical_col, plot=False):
    """
    Provides statistical summary of numerical variable across multiple quantiles.
    Optionally displays a histogram.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the variable
    numerical_col : str
        Name of the numerical column to analyze
    plot : bool, optional
        Whether to display a histogram (default is False)
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# Analyze all numerical variables
for col in num_cols:
    num_summary(df, col, plot=True)


############################
# 8. Target Variable Analysis with Categorical Variables
############################

def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    """
    Analyzes the relationship between categorical variables and the target variable.
    Shows mean of target for each category.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the variables
    target : str
        Name of the target variable
    categorical_col : str
        Name of the categorical variable to analyze
    plot : bool, optional
        Whether to display a bar plot (default is False)
    """
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=categorical_col, y=target, data=dataframe)
        plt.show(block=True)


# Analyze target relationship with all categorical variables
for col in cat_cols:
    target_summary_with_cat(df, 'Survived', col, plot=True)


############################
# 9. Target Variable Analysis with Numerical Variables
############################

def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    """
    Analyzes the relationship between numerical variables and the target variable.
    Shows mean of numerical variable for each target class.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the variables
    target : str
        Name of the target variable
    numerical_col : str
        Name of the numerical variable to analyze
    plot : bool, optional
        Whether to display a bar plot (default is False)
    """
    print(pd.DataFrame({numerical_col + '_mean': dataframe.groupby(target)[numerical_col].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=target, y=numerical_col, data=dataframe)
        plt.show(block=True)


# Analyze target relationship with all numerical variables
for col in num_cols:
    target_summary_with_num(df, 'Survived', col, plot=True)


############################
# 10. Correlation Analysis with Raw Data
############################

def correlation_analysis(dataframe, target_col=None, plot=True, corr_th=0.5):
    """
    Analyzes correlations between numerical variables in the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to analyze
    target_col : str, optional
        Target variable (e.g., 'Survived'). If specified, highlights correlations with this variable
    plot : bool, optional
        Whether to create visualization (default is True)
    corr_th : float, optional
        Threshold for high correlation (default is 0.5)

    Returns
    -------
    high_corr_list : list
        List of variables with high correlation
    """
    # Select only numerical variables (categorical variables require separate analysis)
    numeric_df = dataframe.select_dtypes(include=['float64', 'int64'])

    # Calculate correlation matrix
    corr = numeric_df.corr().round(2)

    # Find highly correlated variables
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    high_corr_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    # Print results
    if len(high_corr_list) > 0:
        print(f"Variables with correlation higher than {corr_th}:")
        for col in high_corr_list:
            # Show which variables have high correlation
            high_corr_pairs = upper_triangle_matrix[col][upper_triangle_matrix[col] > corr_th].index.tolist()
            for pair in high_corr_pairs:
                print(f"- {col} and {pair}: {corr.loc[col, pair]:.2f}")
    else:
        print(f"No variable pairs found with correlation higher than {corr_th}.")

    # Show correlations with target variable (if specified)
    if target_col and target_col in numeric_df.columns:
        print(f"\nCorrelations with {target_col}:")
        target_corrs = corr[target_col].sort_values(ascending=False)
        for idx, val in target_corrs.items():
            if idx != target_col:
                print(f"- {idx}: {val:.2f}")

    # Visualization
    if plot:
        plt.figure(figsize=(10, 8))

        # Create mask to show only lower triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Use different colormap if highlighting target variable
        if target_col and target_col in numeric_df.columns:
            # Move target variable to top and left
            cols = [target_col] + [col for col in corr.columns if col != target_col]
            corr = corr.loc[cols, cols]
            cmap = "coolwarm"
        else:
            cmap = "RdBu_r"

        # Create heatmap
        sns.heatmap(corr, annot=True, cmap=cmap, fmt=".2f",
                    mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

        plt.title('Correlation Matrix Between Variables', fontsize=15)
        plt.tight_layout()
        plt.show(block=True)

    return high_corr_list


# Analyze correlation only in training dataset (where target variable exists)
train_data = df[df['is_train'] == 1]
correlation_analysis(train_data, target_col='Survived')


############################
# 11. Missing Value Analysis and Handling
############################

def missing_values_table(dataframe):
    """
    Analyzes missing values in the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to analyze

    Returns
    -------
    missing_df : pd.DataFrame
        Table containing missing value counts and ratios
    """
    # Columns with missing values
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    # Create dataframe
    missing_df = pd.DataFrame()

    # Total number of observations
    missing_df['count'] = pd.Series([dataframe.shape[0]] * len(na_columns), index=na_columns)

    # Number of missing values
    missing_df['n_miss'] = dataframe[na_columns].isnull().sum().values

    # Missing value ratio
    missing_df['ratio'] = np.round(100 * dataframe[na_columns].isnull().sum().values / dataframe.shape[0], 2)

    # Sort by missing value count in descending order
    missing_df = missing_df.sort_values('n_miss', ascending=False)

    return missing_df


missing_values_table(df)

# 1. Does passenger have cabin information?
df['Has_Cabin'] = df['Cabin'].notnull().astype(int)

# 2. Extract deck information
df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')  # U = Unknown

# Separate training data for analysis
train_df = df[df['is_train'] == 1]

# Survival rate by cabin information availability
print("\nSurvival rate by cabin information availability:")
print(train_df.groupby('Has_Cabin')['Survived'].mean())

# Visualize - Cabin information analysis
plt.figure(figsize=(8, 5))
sns.barplot(x='Has_Cabin', y='Survived', data=train_df)
plt.title('Survival Rate by Cabin Information Availability')
plt.xlabel('Has Cabin Information?')
plt.ylabel('Survival Rate')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show(block=True)

# Analyze survival rate by deck
print("\nSurvival rates by deck:")
print(train_df.groupby('Deck')['Survived'].mean().sort_values(ascending=False))

# Visualize - Deck analysis
plt.figure(figsize=(10, 6))
sns.barplot(x='Deck', y='Survived', data=train_df)
plt.title('Survival Rate by Deck')
plt.xlabel('Deck')
plt.ylabel('Survival Rate')
plt.show(block=True)

# Check passenger count per deck
print("\nPassenger count per deck:")
print(train_df['Deck'].value_counts())

# Relationship between deck and passenger class
print("\nRelationship between deck and passenger class:")
print(pd.crosstab(train_df['Deck'], train_df['Pclass']))


def categorize_deck(deck):
    """
    Categorizes deck into broader groups based on ship layout.
    Upper decks typically had better survival rates.
    """
    if deck in ['A', 'B', 'C']:
        return 'Upper'
    elif deck in ['D', 'E']:
        return 'Middle'
    elif deck in ['F', 'G', 'U', 'T']:
        return 'Lower'
    else:
        return 'Unknown'


# Create new deck category variable
df['Deck_Category'] = df['Deck'].apply(categorize_deck)

# Drop original Cabin and Deck columns
drop_list = ["Deck", "Cabin"]
df.drop(drop_list, axis=1, inplace=True)

# Check survival rates by deck category (training data only)
train_df = df[df['is_train'] == 1]
print("\nSurvival rates by deck category:")
print(train_df.groupby('Deck_Category')['Survived'].mean().sort_values(ascending=False))

# Age (20.09% missing)
# Fill age using group-wise median values

# First, examine age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show(block=True)

# Group by Pclass and Sex for filling
# Calculate median ages by groups
age_medians = df.groupby(['Pclass', 'Sex'])['Age'].median()
print("Median ages by passenger class and gender:")
print(age_medians)

# Fill missing age values
for pclass in [1, 2, 3]:
    for sex in ['male', 'female']:
        age_median = age_medians[pclass, sex]
        # Fill missing values within the same group with group median
        df.loc[(df['Age'].isnull()) &
               (df['Pclass'] == pclass) &
               (df['Sex'] == sex), 'Age'] = age_median

# Check after filling
print(f"Remaining missing Age values after filling: {df['Age'].isnull().sum()}")

# 1. Embarked Variable (0.15% missing - only 2 values)

# Fill missing Embarked values with the most frequent value
# First check Embarked distribution
print("Embarked distribution:")
print(df['Embarked'].value_counts())

# Find most common port
most_common_port = df['Embarked'].mode()[0]
print(f"Most common embarkation port: {most_common_port}")

# Fill missing values
df['Embarked'].fillna(most_common_port, inplace=True)
print(f"Remaining missing Embarked values after filling: {df['Embarked'].isnull().sum()}")

# 2. Fare Variable (0.08% missing - only 1 value)

# Fill missing Fare value with median fare of same passenger class
# First examine Fare distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'].dropna(), kde=True)
plt.title('Fare Distribution')
plt.show(block=True)

# Find passenger class of the missing fare value
missing_fare_pclass = df.loc[df['Fare'].isnull(), 'Pclass'].values[0]
print(f"Passenger class of missing fare: {missing_fare_pclass}")

# Find median fare for this class
median_fare = df[df['Pclass'] == missing_fare_pclass]['Fare'].median()
print(f"Median fare for this class: {median_fare}")

# Fill missing value
df['Fare'].fillna(median_fare, inplace=True)
print(f"Remaining missing Fare values after filling: {df['Fare'].isnull().sum()}")


############################
# 12. Outlier Detection and Analysis
############################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Calculates outlier thresholds using the IQR method.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to analyze
    col_name : str
        Column name to check for outliers
    q1, q3 : float, optional
        Lower and upper quantile values (default: 0.05, 0.95)

    Returns
    -------
    low_limit, up_limit : tuple
        Lower and upper outlier thresholds
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, plot=False):
    """
    Checks if a column contains outliers.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to analyze
    col_name : str
        Column name to check for outliers
    plot : bool, optional
        Whether to create a box plot visualization (default is False)

    Returns
    -------
    bool
        True if outliers exist, False otherwise
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]

    if len(outliers) > 0:
        if plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=dataframe[col_name])
            plt.title(f'Outliers: {col_name}')
            plt.axvline(x=low_limit, color='r', linestyle='--', label=f'Lower Threshold: {low_limit:.2f}')
            plt.axvline(x=up_limit, color='r', linestyle='--', label=f'Upper Threshold: {up_limit:.2f}')
            plt.legend()
            plt.show(block=True)

        print(f"{len(outliers)} outliers detected in {col_name}.")
        return True
    else:
        print(f"No outliers detected in {col_name}.")
        return False


# Perform outlier analysis on numerical variables
print("Numerical variables:", num_cols)

for col in num_cols:
    print(f"\n{'-' * 50}\n{col} outlier analysis:\n{'-' * 50}")

    # Outlier detection and visualization
    has_outliers = check_outlier(df, col, plot=True)

    # If outliers exist, examine distribution in detail
    if has_outliers:
        # Show distribution with histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, color='steelblue')
        plt.title(f"{col} - Current Distribution (Including Outliers)")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        # Add threshold lines
        low_limit, up_limit = outlier_thresholds(df, col)
        plt.axvline(x=up_limit, color='r', linestyle='--', linewidth=2,
                    label=f'Upper Threshold: {up_limit:.2f}')
        if low_limit > df[col].min():
            plt.axvline(x=low_limit, color='r', linestyle='--', linewidth=2,
                        label=f'Lower Threshold: {low_limit:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

        # Uncomment this line if you want to replace outliers with thresholds:
        # replace_with_thresholds(df, col)

        # Provide information about outliers
        n_lower = df[df[col] < low_limit].shape[0]
        n_upper = df[df[col] > up_limit].shape[0]
        print(f"\nOutlier details for {col}:")
        print(f"  • Below lower threshold ({low_limit:.2f}): {n_lower} values")
        print(f"  • Above upper threshold ({up_limit:.2f}): {n_upper} values")
        print(f"  • Total outliers: {n_lower + n_upper}")


############################
# 13. Logarithmic Analysis and Transformation
############################

def log_transformation_analyzer(dataframe, num_cols, skewness_threshold=0.5, plot=True, zero_offset=0.01):
    """
    Analyzes skewness of numerical variables and identifies candidates for log transformation.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to analyze
    num_cols : list
        List of numerical columns to analyze
    skewness_threshold : float, optional
        Skewness threshold for log transformation recommendation (default is 0.5)
    plot : bool, optional
        Whether to create visualizations (default is True)
    zero_offset : float, optional
        Small constant to add to zero values (default is 0.01)

    Returns
    -------
    list
        List of columns recommended for log transformation
    """
    from scipy.stats import skew

    log_candidate_cols = []

    print("Skewness Analysis:")
    print("-" * 50)

    for col in num_cols:
        # Check for negative values
        if dataframe[col].min() < 0:
            print(f"{col}: Contains negative values - not suitable for log transformation")
            continue

        # Check for zero values and temporary correction
        temp_data = dataframe[col].copy()
        zero_count = (temp_data == 0).sum()

        if zero_count > 0:
            print(f"{col}: {zero_count} zero values detected, adding {zero_offset} for log transformation")
            temp_data = temp_data + zero_offset

        # Original skewness
        orig_skewness = skew(dataframe[col])

        # Skewness after log transformation
        log_skewness = skew(np.log1p(temp_data))

        # Check if absolute skewness decreases
        if abs(orig_skewness) > skewness_threshold and abs(log_skewness) < abs(orig_skewness):
            log_candidate_cols.append(col)
            print(f"{col}: Original skewness = {orig_skewness:.2f}, After log = {log_skewness:.2f} - RECOMMENDED")
        else:
            print(f"{col}: Original skewness = {orig_skewness:.2f}, After log = {log_skewness:.2f} - NOT NEEDED")

    # Visualization
    if plot and log_candidate_cols:
        n_cols = len(log_candidate_cols)
        if n_cols > 0:
            fig_height = 5 * ((n_cols + 1) // 2)  # 2 plots per row
            plt.figure(figsize=(15, fig_height))

            for i, col in enumerate(log_candidate_cols, 1):
                # Zero value correction
                temp_data = dataframe[col].copy()
                if (temp_data == 0).sum() > 0:
                    temp_data = temp_data + zero_offset

                # Original distribution
                plt.subplot(n_cols, 2, 2 * i - 1)
                sns.histplot(dataframe[col], kde=True, color='blue')
                plt.title(f"{col} - Original (Skewness: {skew(dataframe[col]):.2f})")
                plt.xlabel(col)
                plt.ylabel("Frequency")

                # Log-transformed distribution
                plt.subplot(n_cols, 2, 2 * i)
                sns.histplot(np.log1p(temp_data), kde=True, color='green')
                plt.title(f"Log({col}+1) - After Transformation (Skewness: {skew(np.log1p(temp_data)):.2f})")
                plt.xlabel(f"Log({col}+1)")
                plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show(block=True)

    return log_candidate_cols


def apply_log_transformation(dataframe, cols_to_transform, drop_originals=False, zero_offset=0.01):
    """
    Applies logarithmic transformation to specified columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process
    cols_to_transform : list
        List of columns to transform
    drop_originals : bool, optional
        Whether to drop original columns (default is False)
    zero_offset : float, optional
        Small constant to add to zero values (default is 0.01)

    Returns
    -------
    pd.DataFrame
        Dataframe with log-transformed columns added
    """
    # Create copy
    df_result = dataframe.copy()

    if not cols_to_transform:
        print("No columns to transform.")
        return df_result

    print("Applying Logarithmic Transformation:")
    print("-" * 50)

    for col in cols_to_transform:
        # Check for zero values
        zero_count = (df_result[col] == 0).sum()

        if zero_count > 0:
            print(f"{col}: Adding {zero_offset} to {zero_count} zero values")
            # Add small constant to zero values
            temp_data = df_result[col] + zero_offset
        else:
            temp_data = df_result[col]

        # Apply log transformation
        df_result[f'Log{col}'] = np.log1p(temp_data)
        print(f"{col} -> Log{col} transformation completed")

    # Drop original columns if requested
    if drop_originals:
        df_result.drop(cols_to_transform, axis=1, inplace=True)
        print(f"Original columns dropped: {', '.join(cols_to_transform)}")

    return df_result


# Analyze and apply log transformation
log_candidates = log_transformation_analyzer(df, num_cols=num_cols)
df = apply_log_transformation(df, cols_to_transform=log_candidates, drop_originals=True)

############################
# 14. Rare Category Analysis and Encoding
############################

# Define categorical variables
cat_cols = ['Sex', 'Embarked', 'Pclass', 'Deck_Category']


def rare_analyser(dataframe, target, cat_cols):
    """
    Analyzes the frequency, ratio, and target mean for each class in categorical variables.
    Helps identify rare categories that might need special handling.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to analyze
    target : str
        Target variable name
    cat_cols : list
        List of categorical columns to analyze
    """
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({'COUNT': dataframe[col].value_counts(),
                            'RATIO': dataframe[col].value_counts() / len(dataframe),
                            'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')


rare_analyser(df, "Survived", cat_cols)


def rare_encoder(dataframe, rare_perc, cat_cols):
    """
    Encodes categorical classes that appear below a certain threshold as 'Rare'.
    This helps reduce dimensionality and handle categories with very few observations.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process
    rare_perc : float
        Threshold for rare category (e.g., 0.01 = less than 1%)
    cat_cols : list
        List of categorical columns to process

    Returns
    -------
    pd.DataFrame
        Dataframe with rare categories encoded
    """
    temp_df = dataframe.copy()

    for col in cat_cols:
        # Calculate ratio for each class
        tmp = temp_df[col].value_counts() / len(temp_df)
        # Find classes below threshold
        rare_labels = tmp[tmp < rare_perc].index
        # Encode rare classes as 'Rare'
        if len(rare_labels) > 0:
            print(f"{len(rare_labels)} rare classes in {col} encoded as 'Rare'")
            print(f"Rare classes: {list(rare_labels)}")
            temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])

    return temp_df


# Typically use 1% or 5% threshold
df = rare_encoder(df, rare_perc=0.01, cat_cols=cat_cols)

############################
# 15. Initial Encoding
############################

df_base = df.copy()


def label_encoder(dataframe, binary_cols=None):
    """
    Encodes binary categorical variables as (0,1).

    This function is specifically designed for variables with exactly two unique values,
    which are more efficiently encoded as binary (0/1) rather than using one-hot encoding.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process
    binary_cols : list, optional
        List of binary variables to apply label encoding
        If None, automatically detects variables with nunique <= 2

    Returns
    -------
    pd.DataFrame
        Dataframe with label encoding applied
    """
    from sklearn.preprocessing import LabelEncoder

    result_df = dataframe.copy()

    if binary_cols is None:
        # Automatically detect binary variables (categorical variables with nunique <= 2)
        binary_cols = [col for col in result_df.columns
                       if result_df[col].dtype not in ['int64', 'float64']
                       and result_df[col].nunique() <= 2]

    if len(binary_cols) == 0:
        print("No binary variables found.")
        return result_df

    le = LabelEncoder()

    for col in binary_cols:
        # Check for missing values
        if result_df[col].isnull().sum() > 0:
            print(f"Warning: {col} contains missing values. LabelEncoder cannot handle missing values.")
            continue

        result_df[col] = le.fit_transform(result_df[col])
        print(f"{col} encoded with label encoding: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

    return result_df


def one_hot_encoder(dataframe, categorical_cols=None, drop_first=True):
    """
    Encodes categorical variables using one-hot encoding.

    One-hot encoding creates binary columns for each category, allowing the model
    to treat each category independently without imposing any ordinal relationship.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process
    categorical_cols : list, optional
        List of categorical variables to apply one-hot encoding
        If None, uses object and category type variables
    drop_first : bool, optional
        Whether to drop the first dummy variable to avoid multicollinearity (default is True)

    Returns
    -------
    pd.DataFrame
        Dataframe with one-hot encoding applied
    """
    result_df = dataframe.copy()

    # Automatically detect categorical variables
    if categorical_cols is None:
        categorical_cols = [col for col in result_df.columns
                            if result_df[col].dtype in ['object', 'category']]

    if len(categorical_cols) == 0:
        print("No categorical variables found.")
        return result_df

    # Check values for each categorical variable
    for col in categorical_cols:
        num_unique = result_df[col].nunique()
        if num_unique <= 1:
            print(f"Warning: {col} contains only one value, skipping one-hot encoding.")
            categorical_cols.remove(col)
        elif num_unique > 30:
            print(f"Warning: {col} contains many unique values ({num_unique}). Be careful!")

    # Apply one-hot encoding
    result_df = pd.get_dummies(result_df, columns=categorical_cols, drop_first=drop_first)

    encoded_cols = [col for col in result_df.columns
                    if col not in dataframe.columns]

    print(f"{len(categorical_cols)} variables encoded with one-hot encoding.")
    print(f"{len(encoded_cols)} new features created.")

    if drop_first:
        print("Note: First dummy variable dropped for each category (drop_first=True).")

    return result_df


# 1. First, encode binary variables with label encoding (if any)
df_base = label_encoder(df_base)

# 2. Then, encode other categorical variables with one-hot encoding
df_base = one_hot_encoder(df_base, categorical_cols=cat_cols, drop_first=True)

############################
# 16. Initial Standardization
############################

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df_base)


def standardize_features(dataframe, num_cols, train_col='is_train', train_value=1, scaler_type='robust'):
    """
    Standardizes numerical features while preventing data leakage between train/test sets.

    This function fits the scaler only on training data and applies the same transformation
    to test data, which is crucial for maintaining the integrity of model evaluation.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing features to standardize
    num_cols : list
        List of numerical columns to standardize
    train_col : str, optional
        Column name used to identify train/test split (default is 'is_train')
    train_value : int, optional
        Value in train_col that indicates training data (default is 1)
    scaler_type : str, optional
        Type of scaler to use: 'standard', 'robust', or 'minmax' (default is 'robust')
        - StandardScaler: Removes mean and scales to unit variance (sensitive to outliers)
        - RobustScaler: Uses median and IQR (robust to outliers) - RECOMMENDED
        - MinMaxScaler: Scales features to a fixed range [0,1]

    Returns
    -------
    scaler
        The fitted scaler object for potential future use
    """
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler()
    }

    scaler = scalers[scaler_type]

    train_mask = dataframe[train_col] == train_value
    test_mask = ~train_mask

    # Fit on training data and transform both train and test
    dataframe.loc[train_mask, num_cols] = scaler.fit_transform(dataframe.loc[train_mask, num_cols])
    dataframe.loc[test_mask, num_cols] = scaler.transform(dataframe.loc[test_mask, num_cols])

    print(f"{len(num_cols)} variables standardized with {scaler_type}Scaler.")
    print(f"Train/Test split: Used '{train_col}' column (train={train_value}).")
    return scaler


scaler = standardize_features(df_base, num_cols)


def clean_column_names(dataframe):
    """
    Cleans column names by removing spaces and special characters (inplace operation).

    This ensures compatibility with various machine learning libraries and prevents
    potential errors caused by special characters in column names.
    """
    dataframe.columns = dataframe.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '',
                                                                            regex=True).str.lower()
    print("Column names cleaned.")


clean_column_names(df_base)


############################
# 17. Base Model Training
############################

def evaluate_models(X, y, models_dict, cv=5):
    """
    Evaluates and compares multiple machine learning models.

    This function provides a comprehensive evaluation using both cross-validation
    and various performance metrics to help identify the best performing model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    models_dict : dict
        Dictionary containing model names (keys) and model objects (values)
    cv : int, optional
        Number of cross-validation folds (default is 5)

    Returns
    -------
    pd.DataFrame
        Model performance results sorted by CV accuracy
    """
    results = []

    for name, model in models_dict.items():
        # Cross-validation for robust performance estimation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        # Model training on full dataset
        model.fit(X, y)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate multiple metrics for comprehensive evaluation
        results.append({
            'Model': name,
            'CV_Accuracy': cv_scores.mean(),
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred),
            'F1_Score': f1_score(y, y_pred),
            'ROC_AUC': roc_auc_score(y, y_pred_proba)
        })

    results_df = pd.DataFrame(results).round(4)
    return results_df.sort_values('CV_Accuracy', ascending=False)


def prepare_base_data(dataframe, target_col, drop_cols=None):
    """
    Prepares data for modeling by separating features and target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to prepare
    target_col : str
        Name of the target variable
    drop_cols : list, optional
        Columns to drop before modeling

    Returns
    -------
    X, y : pd.DataFrame, pd.Series
        Feature matrix and target variable
    """
    df_model = dataframe.copy()

    if drop_cols:
        df_model = df_model.drop(drop_cols, axis=1)

    X = df_model.drop(target_col, axis=1)
    y = df_model[target_col]

    return X, y


# Prepare data for modeling
# Use only training data (is_train == 1)
train_data = df_base[df_base['is_train'] == 1]

# Separate features and target
X, y = prepare_base_data(train_data,
                         target_col='survived',
                         drop_cols=['name', 'is_train'])

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier()
}

# Evaluate all models
results = evaluate_models(X, y, models)

# Display results
print("BASE MODEL RESULTS:")
print("=" * 60)
print(results.to_string(index=False))

# Identify best model
best_model = results.iloc[0]['Model']
print(f"\nBest performing model: {best_model}")


############################
# 18. Feature Extraction - Creating New Features
############################

def create_family_features(dataframe):
    """
    Creates family-related features from SibSp and Parch variables.

    Family size and composition were important factors in Titanic survival,
    as families often stayed together during the evacuation.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process

    Returns
    -------
    pd.DataFrame
        Dataframe with new family features added
    """
    df = dataframe.copy()

    # Total family size (including passenger)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Traveling alone indicator
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Family size categories
    df['FamilyType'] = df['FamilySize'].apply(lambda x:
                                              'Alone' if x == 1
                                              else 'Small' if x <= 4
                                              else 'Large')

    # Has siblings or spouse
    df['HasSiblings'] = (df['SibSp'] > 0).astype(int)

    # Has parents or children
    df['HasParentsChildren'] = (df['Parch'] > 0).astype(int)

    print("Family features created:")
    print(f"- FamilySize: Family size range (1-{df['FamilySize'].max()})")
    print(f"- IsAlone: {df['IsAlone'].sum()} passengers traveling alone")
    print(f"- FamilyType distribution:")
    print(df['FamilyType'].value_counts())

    return df


# Apply family feature creation
df = create_family_features(df)


def extract_title_features(dataframe):
    """
    Extracts title features from the Name column.

    Titles indicate social status and gender, which were significant factors
    in survival rates (e.g., "women and children first" protocol).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process

    Returns
    -------
    pd.DataFrame
        Dataframe with title features added
    """
    df = dataframe.copy()

    # Extract title from name (Mr., Mrs., Miss., etc.)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Group rare titles into 'Rare' category
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare')

    # Standardize similar titles
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    print("Title features created:")
    print(df['Title'].value_counts())

    return df


# Extract title features
df = extract_title_features(df)


def create_age_features(dataframe):
    """
    Creates age group features from the Age column.

    Age was a critical factor in survival, with children having priority
    during evacuation ("women and children first" protocol).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process

    Returns
    -------
    pd.DataFrame
        Dataframe with age features added
    """
    df = dataframe.copy()

    # Age group categories
    df['AgeGroup'] = pd.cut(df['Age'],
                            bins=[0, 12, 18, 35, 60, 100],
                            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

    # Binary age indicators
    df['IsChild'] = (df['Age'] < 18).astype(int)
    df['IsSenior'] = (df['Age'] >= 60).astype(int)

    print("Age features created:")
    print(df['AgeGroup'].value_counts())

    return df


# Apply age feature creation
df = create_age_features(df)


def create_fare_features(dataframe):
    """
    Creates fare category features from the LogFare column.

    Fare was strongly correlated with passenger class and cabin location,
    which directly influenced survival chances.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process

    Returns
    -------
    pd.DataFrame
        Dataframe with fare features added
    """
    df = dataframe.copy()

    # Fare categories based on LogFare
    df['FareCategory'] = pd.cut(df['LogFare'],
                                bins=[0, 2.5, 3.2, 4.0, 5.0],
                                labels=['Low', 'Medium', 'High', 'VeryHigh'])

    # Fare per person (divided by family size)
    # This provides a better measure of individual passenger's fare class
    df['FarePerPerson'] = df['LogFare'] / df['FamilySize']

    print("Fare features created:")
    print(df['FareCategory'].value_counts())

    return df


# Apply fare feature creation
df = create_fare_features(df)


def create_combination_features(dataframe):
    """
    Creates combination features from existing variables.

    These features capture important interactions between variables that
    may have influenced survival outcomes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process

    Returns
    -------
    pd.DataFrame
        Dataframe with combination features added
    """
    df = dataframe.copy()

    # "Women and Children First" protocol indicator
    df['WomenChildrenFirst'] = ((df['Sex'] == 'female') | (df['Age'] < 18)).astype(int)

    # High social status indicator (1st class + cabin + certain titles)
    df['HighStatus'] = ((df['Pclass'] == 1) &
                        (df['Has_Cabin'] == 1) &
                        (df['Title'].isin(['Master', 'Miss', 'Mrs', 'Rare']))).astype(int)

    # Low social status indicator (3rd class + no cabin + Southampton embarkation)
    df['LowStatus'] = ((df['Pclass'] == 3) &
                       (df['Has_Cabin'] == 0) &
                       (df['Embarked'] == 'S')).astype(int)

    # Age-sex combination for granular analysis
    df['AgeSexGroup'] = df['Sex'] + '_' + df['AgeGroup'].astype(str)

    print("Combination features created:")
    print(f"- WomenChildrenFirst: {df['WomenChildrenFirst'].sum()} passengers")
    print(f"- HighStatus: {df['HighStatus'].sum()} passengers")
    print(f"- LowStatus: {df['LowStatus'].sum()} passengers")

    return df


# Apply combination feature creation
df = create_combination_features(df)


def create_name_features(dataframe):
    """
    Creates additional name-based features.

    Name length and complexity can serve as proxies for social status,
    as nobility and upper class often had longer, more elaborate names.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process

    Returns
    -------
    pd.DataFrame
        Dataframe with name features added
    """
    df = dataframe.copy()

    # Name length (can indicate social status)
    df['NameLength'] = df['Name'].str.len()

    # Number of words in name
    df['NameWordCount'] = df['Name'].str.split().str.len()

    # Has middle name (indicated by parentheses after comma)
    df['HasMiddleName'] = df['Name'].str.contains('\(').astype(int)

    print("Name features created:")
    print(f"- Average name length: {df['NameLength'].mean():.1f}")
    print(f"- Passengers with middle name: {df['HasMiddleName'].sum()}")

    return df


# Apply name feature creation
df = create_name_features(df)


def feature_extraction_summary(dataframe):
    """
    Displays summary information after feature extraction.

    Provides an overview of the newly created features and dataset dimensions.
    """
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED!")
    print("=" * 60)
    print(f"Total number of features: {dataframe.shape[1]}")
    print(f"Total number of observations: {dataframe.shape[0]}")
    print("\nNewly created features:")

    new_features = ['FamilySize', 'IsAlone', 'FamilyType', 'HasSiblings', 'HasParentsChildren',
                    'Title', 'AgeGroup', 'IsChild', 'IsSenior', 'FareCategory', 'FarePerPerson',
                    'WomenChildrenFirst', 'HighStatus', 'LowStatus', 'AgeSexGroup',
                    'NameLength', 'NameWordCount', 'HasMiddleName']

    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}")

    print(f"\nTotal of {len(new_features)} new features created!")


# Display feature extraction summary
feature_extraction_summary(df)

# Define columns to drop
drop_cols = ['Name']  # For now only Name, will add more after analysis

print(f"Columns to drop: {drop_cols}")
df = df.drop(drop_cols, axis=1)

############################
# 19. Encoding (For New Features)
############################

# Detect categorical variables with new features
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


def label_encoder(dataframe, binary_cols=None, exclude_cols=None):
    """
    Encodes binary categorical variables as (0,1).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process
    binary_cols : list, optional
        List of binary variables to apply label encoding
        If None, automatically detects variables with nunique <= 2
    exclude_cols : list, optional
        Columns to exclude from encoding (e.g., target variable)
        Default: []

    Returns
    -------
    pd.DataFrame
        Dataframe with label encoding applied
    """
    from sklearn.preprocessing import LabelEncoder

    result_df = dataframe.copy()

    # Set columns to exclude
    if exclude_cols is None:
        exclude_cols = []

    if binary_cols is None:
        # Automatically detect binary variables (categorical with nunique <= 2)
        binary_cols = [col for col in result_df.columns
                       if result_df[col].dtype not in ['int64', 'float64']
                       and result_df[col].nunique() <= 2
                       and col not in exclude_cols]

    if len(binary_cols) == 0:
        print("No binary variables found.")
        return result_df

    le = LabelEncoder()

    for col in binary_cols:
        # Check for missing values
        if result_df[col].isnull().sum() > 0:
            print(f"Warning: {col} contains missing values. LabelEncoder cannot handle missing values.")
            continue

        result_df[col] = le.fit_transform(result_df[col])
        print(f"{col} encoded with label encoding: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

    return result_df


def one_hot_encoder(dataframe, categorical_cols=None, drop_first=True, exclude_cols=None):
    """
    Encodes categorical variables using one-hot encoding.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process
    categorical_cols : list, optional
        List of categorical variables to apply one-hot encoding
        If None, uses object and category type variables
    drop_first : bool, optional
        Whether to drop the first dummy variable (default is True)
    exclude_cols : list, optional
        Columns to exclude from encoding (e.g., target variable)
        Default: []

    Returns
    -------
    pd.DataFrame
        Dataframe with one-hot encoding applied
    """
    result_df = dataframe.copy()

    # Set columns to exclude
    if exclude_cols is None:
        exclude_cols = []

    # Automatically detect categorical variables
    if categorical_cols is None:
        categorical_cols = [col for col in result_df.columns
                            if result_df[col].dtype in ['object', 'category']
                            and col not in exclude_cols]
    else:
        # If manual list provided, remove exclude_cols
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    if len(categorical_cols) == 0:
        print("No categorical variables found.")
        return result_df

    # Check values for each categorical variable
    for col in categorical_cols:
        num_unique = result_df[col].nunique()
        if num_unique <= 1:
            print(f"Warning: {col} contains only one value, skipping one-hot encoding.")
            categorical_cols.remove(col)
        elif num_unique > 30:
            print(f"Warning: {col} contains many unique values ({num_unique}). Be careful!")

    # Apply one-hot encoding
    result_df = pd.get_dummies(result_df, columns=categorical_cols, drop_first=drop_first)

    encoded_cols = [col for col in result_df.columns
                    if col not in dataframe.columns]

    print(f"{len(categorical_cols)} variables encoded with one-hot encoding.")
    print(f"{len(encoded_cols)} new features created.")

    if drop_first:
        print("Note: First dummy variable dropped for each category (drop_first=True).")

    return result_df


# Perform encoding with new features
print("ENCODING WITH NEW FEATURES STARTING...")
print(f"Shape before encoding: {df.shape}")

# Exclude Survived and is_train from encoding
# Survived: Target variable (will be used as y)
# is_train: Train/test split indicator (will be used in standardization)
cat_cols_to_encode = [col for col in cat_cols if col not in ['Survived', 'is_train']]

print(f"Number of categorical variables to encode: {len(cat_cols_to_encode)}")
print(f"Excluded: Survived (target variable), is_train (split indicator)")

# 1. Encode binary variables with label encoding
df_final = label_encoder(df, exclude_cols=['Survived', 'is_train'])

# 2. Encode other categorical variables with one-hot encoding
df_final = one_hot_encoder(df_final, categorical_cols=cat_cols_to_encode,
                           drop_first=True, exclude_cols=['Survived', 'is_train'])

print(f"Shape after encoding: {df_final.shape}")

# Verify is_train column is preserved
if 'is_train' in df_final.columns:
    print("✅ is_train column preserved (will be used for standardization in Section 20)")
else:
    print("❌ WARNING: is_train column is missing!")

############################
# 20. Standardization (For New Features)
############################

# Analyze categorical/numerical variables after encoding
cat_cols_final, num_cols_final, cat_but_car_final, num_but_cat_final = grab_col_names(df_final)


def standardize_features(dataframe, num_cols, train_col='is_train', train_value=1,
                         scaler_type='robust', exclude_cols=None):
    """
    Standardizes numerical features while preventing data leakage between train/test sets.

    CRITICAL: This function implements proper train/test separation to prevent data leakage.
    The scaler is fitted ONLY on training data, then applied to both train and test sets.
    This ensures the model never "sees" test data during the learning phase.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to process
    num_cols : list
        List of numerical columns to standardize
    train_col : str, optional
        Column name used to identify train/test split (default is 'is_train')
    train_value : int, optional
        Value in train_col that indicates training data (default is 1)
    scaler_type : str, optional
        Type of scaler to use: 'standard', 'robust', or 'minmax' (default is 'robust')
        - StandardScaler: Removes mean and scales to unit variance
        - RobustScaler: Uses median and IQR (recommended for data with outliers)
        - MinMaxScaler: Scales features to range [0,1]
    exclude_cols : list, optional
        Columns to exclude from standardization (e.g., target variable)
        Default: []

    Returns
    -------
    scaler
        Fitted scaler object with learned parameters from training data
    """
    # Set columns to exclude
    if exclude_cols is None:
        exclude_cols = []

    # Remove excluded columns from num_cols
    final_num_cols = [col for col in num_cols if col not in exclude_cols]

    if len(final_num_cols) == 0:
        print("No numerical variables found to standardize.")
        return None

    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler()
    }

    scaler = scalers[scaler_type]

    # Separate train and test sets (DATA LEAKAGE PREVENTION)
    train_mask = dataframe[train_col] == train_value
    test_mask = ~train_mask

    # fit_transform on train set (learn and apply parameters)
    dataframe.loc[train_mask, final_num_cols] = scaler.fit_transform(
        dataframe.loc[train_mask, final_num_cols]
    )

    # transform only on test set (apply parameters learned from train)
    dataframe.loc[test_mask, final_num_cols] = scaler.transform(
        dataframe.loc[test_mask, final_num_cols]
    )

    print(f"{len(final_num_cols)} variables standardized with {scaler_type}Scaler.")
    print(f"Train/Test split: Used '{train_col}' column (train={train_value}).")
    if exclude_cols:
        print(f"Excluded columns: {exclude_cols}")

    return scaler


# Apply standardization (with train/test separation as in Section 16)
scaler_final = standardize_features(df_final, num_cols_final)


def clean_column_names(dataframe):
    """
    Cleans column names by removing spaces and special characters (inplace operation).

    This ensures compatibility with various ML libraries and prevents potential errors.
    """
    dataframe.columns = dataframe.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '',
                                                                            regex=True).str.lower()
    print("Column names cleaned.")


clean_column_names(df_final)

print("\n" + "=" * 60)
print("ENCODING AND STANDARDIZATION COMPLETED!")
print("=" * 60)
print(f"Final dataset size: {df_final.shape}")
print("Ready for model training with new dataset!")


############################
# 21. Model Training with New Feature Set
############################

def evaluate_models(X, y, models_dict, cv=5):
    """
    Evaluates and compares multiple machine learning models using various metrics.

    This comprehensive evaluation provides insights into each model's performance
    through cross-validation and multiple performance metrics, helping identify
    the best model for the task.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    models_dict : dict
        Dictionary containing model names (keys) and model objects (values)
    cv : int, optional
        Number of cross-validation folds (default is 5)

    Returns
    -------
    pd.DataFrame
        Model performance results sorted by CV accuracy
    """
    results = []

    # Convert to numpy array (to avoid KNN errors with pandas dataframes)
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y

    for name, model in models_dict.items():
        try:
            # Cross-validation for robust performance estimation
            cv_scores = cross_val_score(model, X_array, y_array, cv=cv, scoring='accuracy')

            # Train model on full dataset
            model.fit(X_array, y_array)
            y_pred = model.predict(X_array)
            y_pred_proba = model.predict_proba(X_array)[:, 1]

            # Calculate comprehensive metrics
            results.append({
                'Model': name,
                'CV_Accuracy': cv_scores.mean(),
                'Accuracy': accuracy_score(y_array, y_pred),
                'Precision': precision_score(y_array, y_pred),
                'Recall': recall_score(y_array, y_pred),
                'F1_Score': f1_score(y_array, y_pred),
                'ROC_AUC': roc_auc_score(y_array, y_pred_proba)
            })

        except Exception as e:
            print(f"Error with {name} model: {e}")
            continue

    results_df = pd.DataFrame(results).round(4)
    return results_df.sort_values('CV_Accuracy', ascending=False)


def prepare_data(dataframe, target_col, drop_cols=None):
    """
    Prepares data for modeling by separating features and target variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to prepare
    target_col : str
        Name of the target variable
    drop_cols : list, optional
        Columns to drop before modeling

    Returns
    -------
    X, y : pd.DataFrame, pd.Series
        Feature matrix and target variable
    """
    df_model = dataframe.copy()

    if drop_cols:
        df_model = df_model.drop(drop_cols, axis=1)

    X = df_model.drop(target_col, axis=1)
    y = df_model[target_col]

    return X, y


# Prepare data with new features
print("MODEL TRAINING WITH NEW FEATURES")
print("=" * 60)

# Use only training data (is_train == 1)
train_data = df_final[df_final['is_train'] == 1]

print(f"Training data size: {train_data.shape}")
print(f"Total number of features: {train_data.shape[1]}")

# Separate features and target
X_new, y_new = prepare_data(train_data,
                            target_col='survived',
                            drop_cols=['is_train'])

print(f"X shape for model training: {X_new.shape}")
print(f"y shape for model training: {y_new.shape}")

# Define models (same base model structure)
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier()
}

# Evaluate models
results_new = evaluate_models(X_new, y_new, models)

# Display results
print("\nMODEL RESULTS WITH NEW FEATURES:")
print("=" * 60)
print(results_new.to_string(index=False))

# Identify best model
best_model_new = results_new.iloc[0]['Model']
print(f"\nBest model (with new features): {best_model_new}")
print(f"Best CV Accuracy: {results_new.iloc[0]['CV_Accuracy']:.4f}")

print("\n" + "=" * 60)
print("MODEL TRAINING WITH NEW DATASET COMPLETED!")
print("=" * 60)

############################
# 22. Base vs Advanced Model Comparison
############################

print("\n" + "=" * 80)
print("BASE MODEL vs ADVANCED MODEL COMPARISON")
print("=" * 80)

# Base Model results (from Section 17)
base_results = {
    'SVM': {'CV_Accuracy': 0.824, 'Train_Accuracy': 0.850, 'ROC_AUC': 0.891},
    'Logistic Regression': {'CV_Accuracy': 0.807, 'Train_Accuracy': 0.820, 'ROC_AUC': 0.866},
    'Random Forest': {'CV_Accuracy': 0.806, 'Train_Accuracy': 0.987, 'ROC_AUC': 0.998},
    'KNN': {'CV_Accuracy': 0.805, 'Train_Accuracy': 0.860, 'ROC_AUC': 0.933}
}

# Advanced Model results (from Section 21)
advanced_results = results_new.set_index('Model')[['CV_Accuracy', 'Accuracy', 'ROC_AUC']].to_dict('index')

# Create comparison table
comparison_data = []
for model in base_results.keys():
    base_cv = base_results[model]['CV_Accuracy']
    adv_cv = advanced_results[model]['CV_Accuracy']
    diff = adv_cv - base_cv

    comparison_data.append({
        'Model': model,
        'Base_CV': base_cv,
        'Advanced_CV': adv_cv,
        'Difference': diff,
        'Change_%': (diff / base_cv) * 100
    })

comparison_df = pd.DataFrame(comparison_data).round(4)
comparison_df = comparison_df.sort_values('Difference', ascending=False)

print("\nCV_ACCURACY COMPARISON:")
print("-" * 80)
print(comparison_df.to_string(index=False))

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Base Model - Number of Features: 16")
print(f"Advanced Model - Number of Features: 73 (+57 features, 4.5x increase)")
print(f"\nAverage CV Accuracy:")
print(f"  Base Model: {comparison_df['Base_CV'].mean():.4f}")
print(f"  Advanced Model: {comparison_df['Advanced_CV'].mean():.4f}")
print(f"  Average Change: {comparison_df['Difference'].mean():.4f} ({comparison_df['Change_%'].mean():.2f}%)")

# Best improvement
best_improvement = comparison_df.iloc[0]
print(f"\nBest Improvement: {best_improvement['Model']}")
print(f"  Base: {best_improvement['Base_CV']:.4f} → Advanced: {best_improvement['Advanced_CV']:.4f}")
print(f"  Increase: +{best_improvement['Difference']:.4f} ({best_improvement['Change_%']:.2f}%)")

# Best overall model
best_overall = comparison_df.loc[comparison_df['Advanced_CV'].idxmax()]
print(f"\nBest Overall Model: {best_overall['Model']}")
print(f"  Advanced CV Accuracy: {best_overall['Advanced_CV']:.4f}")

print("\n" + "=" * 80)

############################
# 23. Feature Importance Analysis (Random Forest Built-in)
############################

print("\n" + "=" * 80)
print("SECTION 2a: FEATURE IMPORTANCE ANALYSIS (RANDOM FOREST)")
print("=" * 80)

# Train Random Forest model
# First, prepare the data
train_data = df_final[df_final['is_train'] == 1].copy()

# Separate features and target
X = train_data.drop(['survived', 'is_train'], axis=1)
y = train_data['survived']

print(f"\nTraining data size: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Target variable distribution:")
print(y.value_counts())
print(f"Survival rate: {(y.mean() * 100):.2f}%")

# Create and train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,  # Create 100 trees
    random_state=42,  # For reproducibility
    max_depth=10,  # Tree depth (prevents overfitting)
    min_samples_split=5,  # Minimum samples required to split a node
    min_samples_leaf=2  # Minimum samples required at leaf node
)

print("\nTraining Random Forest model...")
rf_model.fit(X, y)

# Training accuracy
train_score = rf_model.score(X, y)
print(f"Training set accuracy: {(train_score * 100):.2f}%")

# Cross-validation score
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation Results:")
print(f"CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Extract feature importance values
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "-" * 80)
print("FEATURE IMPORTANCE RANKINGS (ALL FEATURES)")
print("-" * 80)
print(feature_importance.to_string(index=False))

# Visualize top 20 most important features
plt.figure(figsize=(12, 10))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['importance'])
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance Score (Feature Importance)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 20 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Most important feature at top
plt.tight_layout()
plt.show(block=True)

# Statistical summary
print("\n" + "-" * 80)
print("FEATURE IMPORTANCE STATISTICS")
print("-" * 80)
print(f"Total number of features: {len(feature_importance)}")
print(f"Highest importance value: {feature_importance['importance'].max():.4f}")
print(f"Lowest importance value: {feature_importance['importance'].min():.4f}")
print(f"Mean importance value: {feature_importance['importance'].mean():.4f}")
print(f"Median importance value: {feature_importance['importance'].median():.4f}")

# Cumulative importance analysis
feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()

# Number of features that provide 95% of importance
threshold_95 = feature_importance[feature_importance['cumulative_importance'] <= 0.95]
print(f"\nNumber of features providing 95% of total importance: {len(threshold_95)}")
print(f"This represents {(len(threshold_95) / len(feature_importance) * 100):.1f}% of all features")

# Number of features that provide 90% of importance
threshold_90 = feature_importance[feature_importance['cumulative_importance'] <= 0.90]
print(f"Number of features providing 90% of total importance: {len(threshold_90)}")
print(f"This represents {(len(threshold_90) / len(feature_importance) * 100):.1f}% of all features")

# Highlight top 10 features
print("\n" + "=" * 80)
print("TOP 10 MOST IMPORTANT FEATURES WITH INTERPRETATIONS")
print("=" * 80)
for idx, row in feature_importance.head(10).iterrows():
    rank = feature_importance.head(10).index.get_loc(idx) + 1
    print(f"\n{rank}. {row['feature']}")
    print(f"   Importance Score: {row['importance']:.4f}")
    print(f"   Cumulative Importance: {row['cumulative_importance'] * 100:.2f}%")

print("\n" + "=" * 80)
print("SECTION 2a: FEATURE IMPORTANCE ANALYSIS COMPLETED!")
print("=" * 80)

############################
# Section 24: SHAP Analysis
############################

print("\n" + "=" * 80)
print("SECTION 24: SHAP ANALYSIS")
print("=" * 80)

# Import SHAP library
try:
    import shap

    print("SHAP library loaded successfully.")
except ImportError:
    print("SHAP library not found. Please install: pip install shap")
    print("Skipping SHAP analysis...")


def shap_analysis(model, X, feature_names=None, max_display=20, sample_size=100):
    """
    Explains model predictions using SHAP values and creates visualizations.

    SHAP (SHapley Additive exPlanations) shows how each feature contributes to predictions.
    Positive values increase the prediction, negative values decrease it.

    This powerful interpretability tool helps us understand:
    - Which features are most important for predictions
    - How feature values affect individual predictions
    - Whether the model's reasoning aligns with domain knowledge

    Parameters
    ----------
    model : fitted model
        Trained machine learning model (RandomForest, XGBoost, etc.)
    X : pd.DataFrame or np.ndarray
        Feature matrix
    feature_names : list, optional
        Feature names (automatically extracted from DataFrame if available)
    max_display : int, optional
        Maximum number of features to display in plots (default is 20)
    sample_size : int, optional
        Number of samples to use for analysis (for speed) (default is 100)

    Returns
    -------
    shap_values : np.ndarray
        Calculated SHAP values for each sample
    explainer : shap.Explainer
        SHAP explainer object
    shap_importance_df : pd.DataFrame
        Feature importance based on mean absolute SHAP values
    """

    print("\nSTARTING SHAP ANALYSIS...")
    print("=" * 80)
    print(f"Data size: {X.shape}")
    print(f"Model type: {type(model).__name__}")

    # Get feature names
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    # Shorten feature names (long names are unreadable in plots)
    short_names = []
    for name in feature_names:
        if len(name) > 22:
            short_names.append(name[:22])
        else:
            short_names.append(name)

    # Convert data to numpy array
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X

    # Sample for speed (for large datasets)
    if X_array.shape[0] > sample_size:
        print(f"Using {sample_size} samples for speed (instead of total {X_array.shape[0]})")
        import random
        random.seed(42)
        sample_indices = random.sample(range(X_array.shape[0]), sample_size)
        X_sample = X_array[sample_indices]
    else:
        X_sample = X_array
        sample_indices = range(X_array.shape[0])

    # Create SHAP explainer
    print("\nCreating SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_sample)

    print(f"SHAP values shape (raw): {np.array(shap_values).shape}")

    # Binary classification handling
    if isinstance(shap_values, list):
        print(f"Binary classification detected (2 classes)")
        print("Using SHAP values for positive class (survived=1)")
        shap_values = shap_values[1]  # Positive class
        base_value = explainer.expected_value[1]
    else:
        # If 3D array (samples, features, classes)
        if len(shap_values.shape) == 3:
            print(f"Binary classification detected (3D array)")
            print("Using SHAP values for positive class (survived=1)")
            shap_values = shap_values[:, :, 1]  # Positive class
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                   (list, np.ndarray)) else explainer.expected_value
        else:
            base_value = explainer.expected_value

    print(f"SHAP values shape (final): {shap_values.shape}")
    print("SHAP values calculated successfully! ✅")

    # Mean absolute SHAP values (feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    shap_importance['rank'] = range(1, len(shap_importance) + 1)

    # Text outputs
    print("\n" + "=" * 80)
    print("TOP 20 FEATURE IMPORTANCE RANKINGS (SHAP)")
    print("=" * 80)
    print(shap_importance[['rank', 'feature', 'mean_abs_shap']].head(20).to_string(index=False))

    # Comparison with Section 23 (if feature_importance exists globally)
    print("\n" + "=" * 80)
    print("SECTION 23 (RANDOM FOREST) vs SECTION 24 (SHAP) COMPARISON")
    print("=" * 80)

    try:
        # feature_importance comes from Section 23
        comparison = pd.merge(
            feature_importance[['feature', 'importance']].head(20).rename(columns={'importance': 'RF_Importance'}),
            shap_importance[['feature', 'mean_abs_shap']].head(20).rename(columns={'mean_abs_shap': 'SHAP_Importance'}),
            on='feature',
            how='outer'
        )

        # Add RF and SHAP ranks
        comparison['RF_Rank'] = comparison['feature'].map(
            dict(zip(feature_importance['feature'], range(1, len(feature_importance) + 1)))
        )
        comparison['SHAP_Rank'] = comparison['feature'].map(
            dict(zip(shap_importance['feature'], range(1, len(shap_importance) + 1)))
        )

        comparison['Rank_Diff'] = comparison['RF_Rank'] - comparison['SHAP_Rank']
        comparison = comparison.sort_values('SHAP_Rank').reset_index(drop=True)

        print(comparison[['feature', 'RF_Rank', 'SHAP_Rank', 'Rank_Diff', 'RF_Importance', 'SHAP_Importance']].head(
            15).to_string(index=False))

        # Consistency analysis
        top_5_rf = set(feature_importance['feature'].head(5))
        top_5_shap = set(shap_importance['feature'].head(5))
        overlap = top_5_rf.intersection(top_5_shap)

        print(f"\n📊 CONSISTENCY ANALYSIS:")
        print(f"   Top 5 common features: {len(overlap)}/5")
        print(f"   Common features: {', '.join(overlap)}")

    except Exception as e:
        print(f"Could not compare with Section 23: {e}")
        print("(feature_importance variable not found)")

    # Key Insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS (SHAP ANALYSIS)")
    print("=" * 80)
    print(f"✅ Top 3 most important features: {', '.join(shap_importance['feature'].head(3).tolist())}")
    print(
        f"✅ Top 10 features account for: {(shap_importance['mean_abs_shap'].head(10).sum() / shap_importance['mean_abs_shap'].sum() * 100):.1f}% of total impact")
    print(f"✅ Positive SHAP value → INCREASES survival chance")
    print(f"✅ Negative SHAP value → DECREASES survival chance")

    # Visualizations
    print("\n" + "=" * 80)
    print("SHAP VISUALIZATIONS")
    print("=" * 80)

    # Font and style settings
    plt.rcParams['font.size'] = 9

    # 1. Summary Plot - Most important visualization
    print("\n1. Summary Plot (Overview)")
    print("   • Each dot represents one sample")
    print("   • Color: Feature value (red=high, blue=low)")
    print("   • Right shift = positive impact, left shift = negative impact")
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=short_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.show(block=True)

    # 2. Bar Plot - Mean absolute SHAP values
    print("\n2. Bar Plot (Feature Importance Ranking)")
    print("   • Average absolute impact of each feature")
    print("   • Similar to Random Forest importance but more accurate")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=short_names,
                      plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.show(block=True)

    # 3. Detailed explanation for a single sample (Waterfall plot)
    print("\n3. Waterfall Plot (Single Sample Detail - First Sample)")
    print("   • Starts from base value")
    print("   • Each feature increases/decreases the prediction")
    print("   • Shows how final prediction is reached")
    plt.figure(figsize=(12, 10))

    try:
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X_sample[0],
            feature_names=short_names
        ), max_display=15, show=False)
        plt.tight_layout()
        plt.show(block=True)
    except Exception as e:
        print(f"   Waterfall plot error: {e}")
        print("   Alternative: Force plot can be used")

    print("\n" + "=" * 80)
    print("SHAP ANALYSIS COMPLETED! ✅")
    print("=" * 80)

    return shap_values, explainer, shap_importance


# Run SHAP analysis
print("\nPerforming SHAP analysis for Random Forest model...")

try:
    shap_values, shap_explainer, shap_importance_df = shap_analysis(
        model=rf_model,
        X=X,
        feature_names=X.columns.tolist(),
        max_display=20,
        sample_size=100
    )

    print("\n✅ SHAP analysis completed successfully!")
    print("📊 Plots and tables can be reviewed.")
    print("\n💡 CONCLUSION: SHAP and Random Forest importance results compared.")
    print("   Both methods yield similar results → Reliable feature selection!")

except Exception as e:
    print(f"\n❌ Error during SHAP analysis: {e}")
    print("If SHAP library is not installed: pip install shap")

############################
# Section 25: Correlation Analysis with New Features
############################

print("\n" + "=" * 80)
print("SECTION 25: CORRELATION ANALYSIS (WITH NEW FEATURES)")
print("=" * 80)


def analyze_correlation(dataframe, target_col=None, threshold=0.6, plot=True):
    """
    Analyzes correlations between numerical variables.

    Correlation measures linear relationships between variables. High correlation
    between features can indicate redundancy, while correlation with the target
    reveals predictive power.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target variable name
    threshold : float, optional
        High correlation threshold (default is 0.6)
    plot : bool, optional
        Whether to create visualization (default is True)

    Returns
    -------
    corr_matrix : pd.DataFrame
        Correlation matrix
    high_corr_pairs : list
        List of highly correlated variable pairs
    """

    # Get numerical and boolean columns (include boolean!)
    numeric_df = dataframe.select_dtypes(include=['float64', 'int64', 'bool'])

    # Convert bool to int
    bool_cols = numeric_df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        numeric_df[bool_cols] = numeric_df[bool_cols].astype(int)

    print(f"\nTotal of {numeric_df.shape[1]} variables will be analyzed.")
    print(f"  - Numerical (float/int): {dataframe.select_dtypes(include=['float64', 'int64']).shape[1]}")
    print(f"  - Binary (bool → int): {len(bool_cols)}")

    if numeric_df.shape[1] < 2:
        print("Insufficient numerical variables.")
        return None, []

    # Calculate correlation matrix
    print("\nCalculating correlation matrix...")
    corr_matrix = numeric_df.corr()

    # Upper triangle matrix (to avoid repetition)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find highly correlated pairs
    high_corr_pairs = []
    for column in upper_triangle.columns:
        high_corr = upper_triangle[column][upper_triangle[column].abs() > threshold]
        for idx in high_corr.index:
            high_corr_pairs.append({
                'feature_1': column,
                'feature_2': idx,
                'correlation': upper_triangle.loc[idx, column]
            })

    # Print results
    print("\n" + "-" * 80)
    if len(high_corr_pairs) > 0:
        print(f"Found {len(high_corr_pairs)} high correlations above {threshold} threshold:")
        print("-" * 80)
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False, key=abs)
        print(high_corr_df.to_string(index=False))
    else:
        print(f"No high correlations found above {threshold} threshold.")
        print("-" * 80)

    # Correlations with target variable
    if target_col and target_col in numeric_df.columns:
        print(f"\nCorrelations with {target_col.upper()} (Top 15):")
        print("-" * 80)
        target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)

        # Show positive and negative correlations separately
        target_corr_signed = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)

        print("POSITIVE CORRELATIONS (Increase survival):")
        print(target_corr_signed[target_corr_signed > 0].head(10).to_string())

        print("\nNEGATIVE CORRELATIONS (Decrease survival):")
        print(target_corr_signed[target_corr_signed < 0].head(10).to_string())

    # Visualization
    if plot:
        print("\n" + "-" * 80)
        print("CREATING CORRELATION HEATMAP...")
        print("-" * 80)

        # Show only top 30 features with highest correlation (for readability)
        if target_col and target_col in numeric_df.columns:
            top_features = corr_matrix[target_col].abs().sort_values(ascending=False).head(30).index
            plot_corr = corr_matrix.loc[top_features, top_features]
            title = f'Correlation Matrix (Top 30 - Based on {target_col})'
        else:
            plot_corr = corr_matrix
            title = 'Correlation Matrix (All Features)'

        plt.figure(figsize=(12, 10))

        # Mask for upper triangle
        mask = np.triu(np.ones_like(plot_corr, dtype=bool))

        # Heatmap
        sns.heatmap(plot_corr, mask=mask, annot=False, cmap='coolwarm',
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title(title, fontsize=14, pad=15)
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show(block=True)

    return corr_matrix, high_corr_pairs


# Check train data (recreate if not in memory)
if 'train_data' not in locals():
    print("train_data not found, recreating...")
    train_data = df_final[df_final['is_train'] == 1].copy()

# Run analysis
corr_matrix, high_corr_pairs = analyze_correlation(
    dataframe=train_data,
    target_col='survived',
    threshold=0.60,
    plot=True
)

print("\n" + "=" * 80)
print("SECTION 25: CORRELATION ANALYSIS COMPLETED!")
print("=" * 80)

############################
# Section 26: Removing Highly Correlated Features (HYBRID APPROACH)
############################

print("\n" + "=" * 80)
print("SECTION 26: REMOVING HIGHLY CORRELATED FEATURES")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# 1️⃣ MANUAL REMOVAL LIST
# ═══════════════════════════════════════════════════════════════════════════
# 100% redundant (unnecessary) features identified in Section 25

REDUNDANT_FEATURES = [
    # Family size redundancies
    'sibsp_8',  # 1.000 correlation with familysize_11
    'familysize_11',  # Same information as sibsp_8
    'familysize_8',  # 0.912 correlation with sibsp_5

    # Age group redundancies
    'issenior_1',  # 0.918 correlation with agegroup_senior
    'agesexgroup_male_senior',  # 0.928 correlation with agegroup_senior
    'agesexgroup_male_middle',  # 0.762 correlation with agegroup_middle
    'agesexgroup_female_teen',  # 0.703 correlation with agegroup_teen
    'agesexgroup_male_teen',  # 0.682 correlation with agegroup_teen

    # Cabin/deck redundancies
    'deck_category_upper',  # 0.727 correlation with has_cabin_1
]

print("\n📋 MANUAL REMOVAL LIST (REDUNDANT FEATURES):")
print("-" * 80)
for i, feat in enumerate(REDUNDANT_FEATURES, 1):
    print(f"   {i}. {feat}")

# ═══════════════════════════════════════════════════════════════════════════
# 2️⃣ PROTECTED FEATURES LIST
# ═══════════════════════════════════════════════════════════════════════════
# Top 15 features based on Section 23 (RF) and Section 24 (SHAP) importance

PROTECTED_FEATURES = [
    # Top 10 (high rank in both SHAP and RF)
    'title_mr',  # SHAP 1st, RF 1st - MOST IMPORTANT
    'womenchildrenfirst_1',  # SHAP 2nd, RF 3rd - VERY IMPORTANT!
    'sex_1',  # SHAP 3rd, RF 2nd
    'pclass_3',  # SHAP 4th, RF 9th
    'lowstatus_1',  # SHAP 5th, RF 10th
    'title_miss',  # SHAP 6th, RF 7th
    'logfare',  # SHAP 7th, RF 5th
    'namelength',  # SHAP 8th, RF 6th
    'fareperperson',  # SHAP 9th, RF 4th
    'title_mrs',  # SHAP 10th, RF 11th

    # Top 11-15 (important but slightly lower)
    'has_cabin_1',  # SHAP 11th, RF 14th
    'hasmiddlename_1',  # SHAP 12th, RF 13th
    'familytype_small',  # SHAP 13th, RF 15th
    'highstatus_1',  # SHAP 14th, RF 12th
    'age',  # SHAP 15th, RF 8th
]

print("\n🛡️ PROTECTED FEATURES (NEVER REMOVE - TOP 15):")
print("-" * 80)
for i, feat in enumerate(PROTECTED_FEATURES, 1):
    print(f"   {i}. {feat}")


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTION 1: AUTOMATIC CORRELATION CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def remove_high_correlation(dataframe, target_col, threshold=0.90, exclude_cols=None):
    """
    Removes one feature from highly correlated pairs.
    The feature with lower correlation to target is removed.

    This function implements an intelligent feature selection strategy:
    when two features are highly correlated (contain similar information),
    we keep the one that's more predictive of the target variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to clean
    target_col : str
        Target variable name
    threshold : float, optional
        High correlation threshold (default is 0.90)
    exclude_cols : list, optional
        Columns to protect from removal

    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned dataset
    removed_features : list
        List of removed features
    """

    cleaned_df = dataframe.copy()

    if exclude_cols is None:
        exclude_cols = []

    # Add target variable to protection list
    if target_col not in exclude_cols:
        exclude_cols.append(target_col)

    # Get numerical and bool variables
    numeric_df = cleaned_df.select_dtypes(include=['float64', 'int64', 'bool'])
    bool_cols = numeric_df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        numeric_df[bool_cols] = numeric_df[bool_cols].astype(int)

    if target_col not in numeric_df.columns:
        print(f"Target variable '{target_col}' is not numerical!")
        return cleaned_df, []

    # Correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Correlations with target
    target_corr = corr_matrix[target_col]

    # Upper triangle
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    removed_features = []

    # Check each column
    for column in upper_triangle.columns:
        if column in removed_features or column in exclude_cols:
            continue

        # Find highly correlated features with this column
        high_corr = upper_triangle[column][upper_triangle[column] > threshold]

        for feature in high_corr.index:
            if feature in removed_features or feature in exclude_cols:
                continue

            # Which has lower correlation with target?
            if target_corr[column] < target_corr[feature]:
                to_remove = column
                to_keep = feature
            else:
                to_remove = feature
                to_keep = column

            if to_remove not in removed_features and to_remove not in exclude_cols:
                removed_features.append(to_remove)
                print(f"   ✂️ {to_remove}")
                print(f"      Reason: {to_keep} ↔ {to_remove} correlation {upper_triangle.loc[feature, column]:.3f}")
                print(
                    f"      With survived: {to_keep} ({target_corr[to_keep]:.3f}) > {to_remove} ({target_corr[to_remove]:.3f})")

    return cleaned_df.drop(columns=removed_features), removed_features


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTION 2: HYBRID CLEANING (Manual + Automatic)
# ═══════════════════════════════════════════════════════════════════════════

def remove_redundant_features(dataframe, target_col='survived',
                              manual_remove=None,
                              force_protect=None,
                              auto_threshold=0.90):
    """
    Hybrid cleaning approach combining manual curation with automated selection.

    This three-step process ensures we remove truly redundant features while
    protecting the most important ones:
    1. Manual removal: Delete known redundant features
    2. Automatic removal: Use correlation + importance for intelligent cleanup
    3. Force protection: Never delete critical features

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to clean
    target_col : str, optional
        Target variable (default is 'survived')
    manual_remove : list, optional
        Manual list of features to remove (REDUNDANT_FEATURES)
    force_protect : list, optional
        Features to protect (PROTECTED_FEATURES)
    auto_threshold : float, optional
        Correlation threshold for automatic cleaning (default is 0.90)

    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned dataset
    removed_all : list
        All removed features (tuple: (feature, reason))
    """

    cleaned_df = dataframe.copy()
    removed_all = []

    print("\n" + "=" * 80)
    print("STARTING HYBRID CLEANING")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: MANUAL REMOVAL (REDUNDANT_FEATURES)
    # ─────────────────────────────────────────────────────────────────────
    print("\n📌 STEP 1: MANUAL REMOVAL (Truly Redundant Features)")
    print("-" * 80)

    if manual_remove:
        for col in manual_remove:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=col)
                removed_all.append((col, 'MANUAL'))
                print(f"   ✂️ {col}")
        print(f"\n   Total {len([r for r in removed_all if r[1] == 'MANUAL'])} features manually removed.")
    else:
        print("   Manual removal list is empty.")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: AUTOMATIC REMOVAL (Correlation >0.90 + Not Protected)
    # ─────────────────────────────────────────────────────────────────────
    print("\n📌 STEP 2: AUTOMATIC REMOVAL (High Correlation + Low Importance)")
    print("-" * 80)
    print(f"   Correlation threshold: {auto_threshold}")
    print(f"   Number of protected features: {len(force_protect) if force_protect else 0}")
    print()

    # Add protected list to exclude_cols
    exclude_cols = (force_protect if force_protect else []) + [target_col, 'is_train']

    # Perform automatic cleaning
    cleaned_df, removed_auto = remove_high_correlation(
        cleaned_df, target_col, auto_threshold, exclude_cols
    )

    for col in removed_auto:
        removed_all.append((col, 'AUTOMATIC'))

    if removed_auto:
        print(f"\n   Total {len(removed_auto)} features automatically removed.")
    else:
        print("   No automatic removals found (all high correlations are protected or already removed).")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"📊 Starting size: {dataframe.shape}")
    print(f"📊 Final size: {cleaned_df.shape}")
    print(f"✂️ Total removed: {len(removed_all)} features")
    print(f"   - Manual: {len([r for r in removed_all if r[1] == 'MANUAL'])}")
    print(f"   - Automatic: {len([r for r in removed_all if r[1] == 'AUTOMATIC'])}")

    if removed_all:
        print(f"\n📋 ALL REMOVED FEATURES:")
        for i, (feat, reason) in enumerate(removed_all, 1):
            print(f"   {i}. {feat} ({reason})")

    return cleaned_df, removed_all


# ═══════════════════════════════════════════════════════════════════════════
# APPLY HYBRID CLEANING
# ═══════════════════════════════════════════════════════════════════════════

df_cleaned, removed_all = remove_redundant_features(
    dataframe=df_final,
    target_col='survived',
    manual_remove=REDUNDANT_FEATURES,  # 1️⃣ Manual list
    force_protect=PROTECTED_FEATURES,  # 2️⃣ Protected list
    auto_threshold=0.90  # 3️⃣ Automatic threshold
)

print("\n" + "=" * 80)
print("SECTION 26: CLEANING COMPLETED!")
print("=" * 80)
print(f"✅ Cleaned dataset: df_cleaned")
print(f"📏 Size: {df_cleaned.shape}")
print(f"✂️ Removed: {len(removed_all)} features")

############################
# Section 27: Feature Selection
############################

print("\n" + "=" * 80)
print("SECTION 27: FEATURE SELECTION")
print("=" * 80)


def select_features_by_importance(importance_df, cumulative_threshold=0.95):
    """
    Selects features based on cumulative importance score.

    This method ensures we keep features that contribute to a specified
    percentage of total model importance, allowing us to reduce dimensionality
    while retaining most predictive power.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    cumulative_threshold : float, optional
        Cumulative importance threshold (0.95 = 95%) (default is 0.95)

    Returns
    -------
    selected_features : list
        Selected features
    """

    # Calculate cumulative importance
    df = importance_df.copy()
    df = df.sort_values('importance', ascending=False)
    df['cumulative_importance'] = df['importance'].cumsum()

    # Select features above threshold
    selected = df[df['cumulative_importance'] <= cumulative_threshold]
    selected_features = selected['feature'].tolist()

    print(f"\nCumulative importance threshold: {cumulative_threshold * 100}%")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Total number of features: {len(df)}")
    print(f"Selection ratio: {(len(selected_features) / len(df) * 100):.1f}%")

    return selected_features


def select_features_by_threshold(importance_df, min_importance=0.01):
    """
    Selects features based on minimum importance score.

    Features below the threshold are considered too weak to contribute
    meaningfully to predictions and are filtered out.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    min_importance : float, optional
        Minimum importance score threshold (default is 0.01)

    Returns
    -------
    selected_features : list
        Selected features
    """

    selected = importance_df[importance_df['importance'] >= min_importance]
    selected_features = selected['feature'].tolist()

    print(f"\nMinimum importance threshold: {min_importance}")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Number of filtered features: {len(importance_df) - len(selected_features)}")

    return selected_features


# Select from feature importance (using feature_importance from Section 22)
print("\n1. Selection by Cumulative Importance:")
selected_features_cumulative = select_features_by_importance(
    importance_df=feature_importance,
    cumulative_threshold=0.95
)

print("\n2. Selection by Minimum Importance:")
selected_features_threshold = select_features_by_threshold(
    importance_df=feature_importance,
    min_importance=0.005
)

# Use features selected by both methods
selected_features = list(set(selected_features_cumulative) & set(selected_features_threshold))
print(f"\nCommon in both methods: {len(selected_features)} features")

# FILTER: Remove features NOT present in df_cleaned
available_cols = df_cleaned.columns.tolist()
selected_features_filtered = [f for f in selected_features if f in available_cols]
removed_features = [f for f in selected_features if f not in available_cols]

print(f"Available in df_cleaned: {len(selected_features_filtered)} features")
if removed_features:
    print(f"⚠️ Removed in Section 26 (skipped): {len(removed_features)} features")
    for feat in removed_features:
        print(f"   - {feat}")

# Create new dataset with selected features
train_selected = df_cleaned[df_cleaned['is_train'] == 1].copy()
X_selected = train_selected[selected_features_filtered]
y_selected = train_selected['survived']

print(f"\nDataset with selected features: {X_selected.shape}")

# ═══════════════════════════════════════════════════════════════════════
# DETAILED LIST OF SELECTED AND FILTERED FEATURES
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("DETAILED FEATURE LISTS")
print("=" * 80)

# List of selected features with importance scores
print("\n✅ SELECTED 32 FEATURES (With Importance Scores):")
print("-" * 80)
selected_with_importance = feature_importance[
    feature_importance['feature'].isin(selected_features_filtered)
].sort_values('importance', ascending=False).reset_index(drop=True)

for i, row in selected_with_importance.iterrows():
    print(f"   {i + 1:2d}. {row['feature']:30s} → Importance: {row['importance']:.4f}")

# List of filtered features
print("\n❌ FILTERED 32 FEATURES (Low Importance):")
print("-" * 80)
all_features_in_cleaned = [col for col in df_cleaned.columns
                           if col not in ['survived', 'is_train']]
removed_features_list = [f for f in all_features_in_cleaned
                         if f not in selected_features_filtered]

removed_with_importance = feature_importance[
    feature_importance['feature'].isin(removed_features_list)
].sort_values('importance', ascending=False).reset_index(drop=True)

for i, row in removed_with_importance.iterrows():
    print(f"   {i + 1:2d}. {row['feature']:30s} → Importance: {row['importance']:.4f}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(
    f"Total importance of selected 32 features: {selected_with_importance['importance'].sum():.4f} ({selected_with_importance['importance'].sum() * 100:.1f}%)")
print(
    f"Total importance of filtered 32 features: {removed_with_importance['importance'].sum():.4f} ({removed_with_importance['importance'].sum() * 100:.1f}%)")
print(f"\nAverage importance of selected features: {selected_with_importance['importance'].mean():.4f}")
print(f"Average importance of filtered features: {removed_with_importance['importance'].mean():.4f}")

print("\n" + "=" * 80)
print("SECTION 27: FEATURE SELECTION COMPLETED!")
print("=" * 80)

############################
# Section 28: Ablation Testing
############################

print("\n" + "=" * 80)
print("SECTION 28: ABLATION TESTING")
print("=" * 80)


def ablation_test(X, y, model, feature_names=None, top_n=10, cv=5, baseline_score=None):
    """
    Tests the true importance of features through ablation testing.

    What is ablation testing?
    Ablation testing is like conducting a scientific experiment on your model.
    We remove one feature at a time and measure how much the model's performance
    drops. If removing a feature causes a significant drop, that feature is truly
    important. This method is more reliable than feature importance scores because
    it reveals how features interact with each other in making predictions.

    Think of it like this: imagine you're baking a cake and want to know which
    ingredients are essential. You could read their nutritional labels (like
    feature importance scores), or you could try baking without each ingredient
    one at a time and taste the results (ablation testing). The second approach
    tells you the real impact!

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        All features
    y : pd.Series or np.ndarray
        Target variable
    model : sklearn model
        Model to test
    feature_names : list, optional
        Feature names
    top_n : int, optional
        Number of top features to test (default is 10)
    cv : int, optional
        Number of cross-validation folds (default is 5)
    baseline_score : float, optional
        Baseline score achieved with all features

    Returns
    -------
    ablation_results : pd.DataFrame
        Test results for each feature
    """

    # Get feature names
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    # Convert to numpy array
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X

    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = y

    # Calculate baseline score (with all features)
    if baseline_score is None:
        print("Calculating baseline score (with all features)...")
        baseline_scores = cross_val_score(model, X_array, y_array, cv=cv, scoring='accuracy')
        baseline_score = baseline_scores.mean()
        print(f"Baseline Accuracy: {baseline_score:.4f}")

    print(f"\nStarting ablation testing...")
    print(f"Number of features to test: {min(top_n, len(feature_names))}")
    print("-" * 80)

    results = []

    # Test each feature
    for i, feature in enumerate(feature_names[:top_n], 1):
        # Remove this feature
        feature_idx = feature_names.index(feature)
        X_without_feature = np.delete(X_array, feature_idx, axis=1)

        # Measure model performance without this feature
        scores_without = cross_val_score(model, X_without_feature, y_array, cv=cv, scoring='accuracy')
        score_without = scores_without.mean()

        # Calculate performance drop
        score_drop = baseline_score - score_without
        drop_percentage = (score_drop / baseline_score) * 100

        results.append({
            'feature': feature,
            'baseline_score': baseline_score,
            'score_without': score_without,
            'score_drop': score_drop,
            'drop_percentage': drop_percentage
        })

        print(
            f"{i:2d}. {feature:30s} | Without: {score_without:.4f} | Drop: {score_drop:.4f} ({drop_percentage:+.2f}%)")

    # Convert results to DataFrame and sort
    ablation_df = pd.DataFrame(results)
    ablation_df = ablation_df.sort_values('score_drop', ascending=False)

    # Visualization
    plt.figure(figsize=(12, 8))

    colors = ['red' if x > 0 else 'green' for x in ablation_df['score_drop']]
    plt.barh(range(len(ablation_df)), ablation_df['score_drop'], color=colors, alpha=0.7)
    plt.yticks(range(len(ablation_df)), ablation_df['feature'])
    plt.xlabel('Performance Drop (Baseline - Without Feature)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Ablation Test Results\n(Positive = Important Feature, Negative = Unnecessary Feature)',
              fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show(block=True)

    # Summary
    print("\n" + "=" * 80)
    print("ABLATION TEST SUMMARY")
    print("=" * 80)

    critical_features = ablation_df[ablation_df['score_drop'] > 0.01]
    print(f"\nCritical features (>1% performance drop): {len(critical_features)}")
    if len(critical_features) > 0:
        print("\nMost critical features:")
        print(critical_features[['feature', 'score_drop', 'drop_percentage']].head(5).to_string(index=False))

    unnecessary_features = ablation_df[ablation_df['score_drop'] < 0]
    print(f"\nPotentially unnecessary features (no performance drop): {len(unnecessary_features)}")
    if len(unnecessary_features) > 0:
        print("\nFeatures that could be removed:")
        print(unnecessary_features[['feature', 'score_drop']].to_string(index=False))

    return ablation_df


# Run ablation testing
# Use selected features (X_selected and y_selected from Section 27)
ablation_results = ablation_test(
    X=X_selected,
    y=y_selected,
    model=RandomForestClassifier(random_state=42, n_estimators=100),
    top_n=15,
    cv=5
)

print("\n" + "=" * 80)
print("ABLATION TESTING COMPLETED!")
print("=" * 80)

############################
# Section 29: Cross-Validation Strategy Comparison
############################

print("\n" + "=" * 80)
print("SECTION 29: CROSS-VALIDATION STRATEGY COMPARISON")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# REMOVING 3 UNNECESSARY FEATURES BASED ON ABLATION TEST RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("UPDATING DATASET BASED ON ABLATION TEST RESULTS")
print("=" * 80)

# 3 features found unnecessary in ablation test
ABLATION_REMOVE = ['sibsp_1', 'isalone_1', 'namewordcount_4']

print(f"\nRemoved features (They hurt performance):")
for i, feat in enumerate(ABLATION_REMOVE, 1):
    print(f"   {i}. {feat}")

# Remove 3 features from 32
selected_features_final = [f for f in selected_features_filtered
                           if f not in ABLATION_REMOVE]

print(f"\n📊 Feature Count: 32 → 29")
print(f"✅ New feature count: {len(selected_features_final)}")

# Create new dataset
X_final = train_selected[selected_features_final]
y_final = y_selected

print(f"\nX_final shape: {X_final.shape}")
print(f"y_final shape: {y_final.shape}")

print("\n" + "=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION STRATEGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

print("\nCross-validation is critical for measuring our model's true performance.")
print("However, which CV strategy we choose can significantly affect the results.")
print("In this section, we'll compare different CV strategies and select the most appropriate one.")

from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold


def compare_cv_strategies(X, y, model, cv_strategies, n_runs=1):
    """
    Compares different cross-validation strategies.

    Understanding cross-validation strategies is like understanding different
    ways to test a student's knowledge. You could:
    1. Give random tests (Standard K-Fold)
    2. Give tests that maintain the same difficulty level (Stratified K-Fold)
    3. Give multiple sets of tests to get a more reliable average (Repeated)

    Each approach has its strengths, and we need to find which one gives us
    the most reliable estimate of our model's performance.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        Target variable
    model : sklearn model
        Model to test
    cv_strategies : dict
        CV strategies and names {'name': cv_object}
    n_runs : int, optional
        Number of repetitions for each strategy (default is 1)

    Returns
    -------
    results_df : pd.DataFrame
        Results for each strategy
    """

    print("\n" + "=" * 60)
    print("TESTING CROSS-VALIDATION STRATEGIES")
    print("=" * 60)

    results = []

    for strategy_name, cv_strategy in cv_strategies.items():
        print(f"\nTesting {strategy_name}...")

        all_scores = []
        fold_distributions = []

        for run in range(n_runs):
            # Calculate cross-validation scores
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
            all_scores.extend(scores)

            # Check class distribution in each fold
            for train_idx, test_idx in cv_strategy.split(X, y):
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_test_fold = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]

                train_positive_ratio = y_train_fold.mean()
                test_positive_ratio = y_test_fold.mean()

                fold_distributions.append({
                    'train_positive_ratio': train_positive_ratio,
                    'test_positive_ratio': test_positive_ratio
                })

        # Calculate statistics
        all_scores = np.array(all_scores)
        fold_dist_df = pd.DataFrame(fold_distributions)

        # Positive class ratio in original dataset
        original_positive_ratio = y.mean()

        # Deviation in each fold
        train_deviations = np.abs(fold_dist_df['train_positive_ratio'] - original_positive_ratio)
        test_deviations = np.abs(fold_dist_df['test_positive_ratio'] - original_positive_ratio)

        results.append({
            'Strategy': strategy_name,
            'Mean Score': all_scores.mean(),
            'Std Deviation': all_scores.std(),
            'Min Score': all_scores.min(),
            'Max Score': all_scores.max(),
            'Score Range': all_scores.max() - all_scores.min(),
            'Train Distribution Deviation': train_deviations.mean(),
            'Test Distribution Deviation': test_deviations.mean()
        })

        print(f"  Mean Score: {all_scores.mean():.4f} (+/- {all_scores.std():.4f})")
        print(f"  Score Range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
        print(f"  Distribution Deviation: Train={train_deviations.mean():.4f}, Test={test_deviations.mean():.4f}")

    results_df = pd.DataFrame(results)

    return results_df


def visualize_cv_comparison(results_df, y):
    """
    Visualizes CV strategy comparison.

    These visualizations help us understand:
    - Which strategy gives the most consistent results?
    - Which strategy best preserves class distribution?
    - Which strategy should we trust most?
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Mean score and confidence interval
    ax1 = axes[0, 0]
    strategies = results_df['Strategy']
    means = results_df['Mean Score']
    stds = results_df['Std Deviation']

    ax1.bar(strategies, means, alpha=0.7, color='steelblue')
    ax1.errorbar(strategies, means, yerr=stds, fmt='none', ecolor='red', capsize=5)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Mean Score and Confidence Interval', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Score range (consistency)
    ax2 = axes[0, 1]
    score_ranges = results_df['Score Range']
    colors = ['green' if x < 0.03 else 'orange' if x < 0.05 else 'red' for x in score_ranges]
    ax2.barh(strategies, score_ranges, color=colors, alpha=0.7)
    ax2.set_xlabel('Score Range (Max - Min)', fontsize=12)
    ax2.set_title('Consistency Across Folds\n(Lower = Better)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # 3. Distribution preservation
    ax3 = axes[1, 0]
    train_dev = results_df['Train Distribution Deviation']
    test_dev = results_df['Test Distribution Deviation']

    x = np.arange(len(strategies))
    width = 0.35

    ax3.bar(x - width / 2, train_dev, width, label='Train', alpha=0.8)
    ax3.bar(x + width / 2, test_dev, width, label='Test', alpha=0.8)
    ax3.set_ylabel('Mean Deviation', fontsize=12)
    ax3.set_title('Class Distribution Preservation\n(Lower = Better)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Find best strategy
    best_idx = results_df['Mean Score'].idxmax()
    best_strategy = results_df.loc[best_idx]

    # Find most consistent strategy (lowest std)
    most_stable_idx = results_df['Std Deviation'].idxmin()
    most_stable = results_df.loc[most_stable_idx]

    # Original distribution
    original_ratio = y.mean()

    summary_text = f"""
    CROSS-VALIDATION COMPARISON SUMMARY
    {'=' * 50}

    Original Dataset:
    - Positive Class Ratio: {original_ratio:.1%}
    - Total Samples: {len(y)}

    Highest Score:
    - Strategy: {best_strategy['Strategy']}
    - Score: {best_strategy['Mean Score']:.4f}

    Most Consistent (Low Variance):
    - Strategy: {most_stable['Strategy']}
    - Std: {most_stable['Std Deviation']:.4f}

    Recommended Strategy:
    - {'Stratified K-Fold' if 'Stratified' in best_strategy['Strategy'] else best_strategy['Strategy']}
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.show(block=True)


def explain_cv_strategies():
    """
    Explains CV strategies in plain English.

    Think of cross-validation like different ways a teacher might test students
    to ensure the grades are fair and representative of true knowledge.
    """

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION STRATEGIES EXPLAINED")
    print("=" * 80)

    explanations = """

1. STANDARD K-FOLD
   Randomly divides data into K parts. Each part becomes the test set once.

   Advantages:
   - Simple and intuitive
   - Fast

   Disadvantages:
   - Can disrupt class distribution in imbalanced datasets
   - Each fold might have different difficulty levels

   When to use:
   - With balanced datasets
   - When you need quick testing

2. STRATIFIED K-FOLD (RECOMMENDED)
   Maintains the original class distribution in each fold.

   Advantages:
   - Preserves class distribution
   - More reliable results
   - Each fold has similar difficulty

   Disadvantages:
   - Slightly slower than Standard K-Fold

   When to use:
   - With imbalanced datasets (most real-world problems)
   - For classification problems (recommended approach)

3. REPEATED STRATIFIED K-FOLD
   Runs Stratified K-Fold multiple times with different random seeds.

   Advantages:
   - Most reliable results
   - Better variance measurement
   - Eliminates luck-based results

   Disadvantages:
   - Slowest method
   - Requires more computation

   When to use:
   - With small datasets
   - When very precise measurement is needed
   - For final model selection
    """

    print(explanations)


# Explain CV strategies
explain_cv_strategies()

# Test different CV strategies
print("\n" + "=" * 80)
print("TESTING DIFFERENT CV STRATEGIES")
print("=" * 80)

# Strategies to test
cv_strategies = {
    'Standard K-Fold (5-fold)': KFold(n_splits=5, shuffle=True, random_state=42),
    'Stratified K-Fold (5-fold)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Stratified K-Fold (10-fold)': StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    'Repeated Stratified K-Fold (3x5)': RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
}

# Use a simple Random Forest model (for fast testing)
test_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

# Compare strategies (with 29 features!)
cv_results = compare_cv_strategies(
    X=X_final,
    y=y_final,
    model=test_model,
    cv_strategies=cv_strategies,
    n_runs=1
)

# Show results
print("\n" + "=" * 80)
print("CV STRATEGY COMPARISON RESULTS")
print("=" * 80)
print("\n" + cv_results.to_string(index=False))

# Visualize
visualize_cv_comparison(cv_results, y_final)

# Recommendation and decision
print("\n" + "=" * 80)
print("CV STRATEGY SELECTION AND RECOMMENDATIONS")
print("=" * 80)

# Select best strategy
best_strategy_idx = cv_results['Mean Score'].idxmax()
best_strategy_name = cv_results.loc[best_strategy_idx, 'Strategy']
best_score = cv_results.loc[best_strategy_idx, 'Mean Score']
best_std = cv_results.loc[best_strategy_idx, 'Std Deviation']

# Find most consistent strategy
most_stable_idx = cv_results['Std Deviation'].idxmin()
most_stable_name = cv_results.loc[most_stable_idx, 'Strategy']

print(f"\nHIGHEST MEAN SCORE:")
print(f"  Strategy: {best_strategy_name}")
print(f"  Score: {best_score:.4f} (+/- {best_std:.4f})")

print(f"\nMOST CONSISTENT RESULTS:")
print(f"  Strategy: {most_stable_name}")
print(f"  Std Deviation: {cv_results.loc[most_stable_idx, 'Std Deviation']:.4f}")

# Special recommendation for Titanic dataset
original_positive_ratio = y_final.mean()
print(f"\n{'=' * 60}")
print("RECOMMENDATION FOR TITANIC DATASET")
print(f"{'=' * 60}")
print(f"\nDataset Characteristics:")
print(f"  - Survival Rate: {original_positive_ratio:.1%}")
print(f"  - Imbalanced? {'Yes (moderate)' if 0.3 < original_positive_ratio < 0.7 else 'Highly imbalanced'}")
print(f"  - Data Size: {len(y_final)} samples")

if 0.35 <= original_positive_ratio <= 0.65:
    recommendation = "Stratified K-Fold (5 or 10-fold)"
    reason = """
    Your dataset is moderately imbalanced. Using Stratified K-Fold 
    preserves class distribution and gives more reliable results.

    Although the difference between Standard and Stratified K-Fold may 
    seem small, these small differences can be important in hyperparameter 
    optimization.
    """
else:
    recommendation = "Repeated Stratified K-Fold"
    reason = """
    Your dataset is quite imbalanced. Using Repeated Stratified K-Fold
    both preserves class distribution and provides a more reliable
    performance estimate through repetitions.
    """

print(f"\nRECOMMENDED STRATEGY: {recommendation}")
print(f"Reasoning: {reason}")

# Save selected strategy (for use in subsequent sections)
selected_cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'=' * 80}")
print("SELECTED STRATEGY: Stratified K-Fold (5-fold)")
print("This strategy will be used in all subsequent model evaluations")
print(f"{'=' * 80}")

print("\n" + "=" * 80)
print("SECTION 29 COMPLETED!")
print("=" * 80)
print("\nKey Takeaways:")
print("1. Stratified K-Fold provides more reliable results with imbalanced datasets")
print("2. Each fold maintains class distribution for fair model evaluation")
print("3. Though differences between Standard and Stratified K-Fold are small, they're meaningful")
print("4. Stratified K-Fold is recommended for moderately imbalanced datasets like Titanic")

############################
# Section 30: Model Development and Hyperparameter Optimization
############################

print("\n" + "=" * 80)
print("SECTION 30: MODEL DEVELOPMENT AND HYPERPARAMETER OPTIMIZATION")
print("=" * 80)

# In this section we'll see two different hyperparameter optimization methods:
# 1. GridSearchCV - Classic but guaranteed method
# 2. Optuna - Modern and fast method

print("\nTwo different optimization methods will be compared:")
print("1. GridSearchCV: Tests all combinations (slow but guaranteed)")
print("2. Optuna: Smart search (fast and efficient)")

import time


def optimize_with_gridsearch(X, y, model, param_grid, cv, scoring='accuracy'):
    """
    Optimizes model hyperparameters using GridSearchCV.

    GridSearch is like trying every possible combination of ingredients in a recipe
    to find the perfect dish. It's thorough but time-consuming. Imagine you're
    trying to make the perfect coffee - you'd test every combination of:
    - Coffee amount: 1 spoon, 2 spoons, 3 spoons
    - Water temperature: 80°C, 90°C, 100°C
    - Brewing time: 2 min, 3 min, 4 min

    GridSearch does exactly this - it tries EVERY combination systematically.

    Advantage: Guaranteed to find the best combination in the search space
    Disadvantage: Can be very slow with many parameters

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        Target variable
    model : sklearn model
        Model to optimize
    param_grid : dict
        Parameter search space
    cv : cross-validation strategy
        Cross-validation strategy (from Section 29)
    scoring : str, optional
        Optimization metric (default is 'accuracy')

    Returns
    -------
    best_model : fitted model
        Model trained with best parameters
    best_params : dict
        Best parameters
    best_score : float
        Best score
    search_time : float
        Search duration (seconds)
    """

    print(f"\n{'=' * 60}")
    print("OPTIMIZATION WITH GRIDSEARCHCV")
    print(f"{'=' * 60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameter combinations: {len(ParameterGrid(param_grid))}")
    print("Starting optimization...\n")

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    search_time = time.time() - start_time

    print(f"\nOptimization completed!")
    print(f"Duration: {search_time:.2f} seconds")
    print(f"Best score: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, search_time


def optimize_with_optuna(X, y, model_class, param_space_func, n_trials=50, cv=None, scoring='accuracy'):
    """
    Optimizes model hyperparameters using Optuna.

    Optuna is like having an intelligent assistant who learns from previous attempts.
    Instead of trying every combination blindly, it uses Bayesian Optimization to
    intelligently guess which combinations are most promising.

    Think of it like this: imagine you're searching for treasure on an island.
    - GridSearch: You dig every single spot systematically (slow but thorough)
    - Optuna: You use a metal detector that gets stronger signals near treasure,
              guiding you to promising spots (fast and smart)

    Optuna looks at previous trials and thinks: "Hmm, larger values of parameter X
    gave better results, so let me try even larger values!" This intelligent search
    often finds great parameters much faster than GridSearch.

    Advantage: Finds good results with fewer trials, much faster
    Disadvantage: Might miss the global optimum (but rarely in practice)

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        Target variable
    model_class : class
        Model class (e.g., RandomForestClassifier)
    param_space_func : function
        Function returning parameter space
    n_trials : int, optional
        Number of trials (default is 50)
    cv : cross-validation strategy
        Cross-validation strategy (from Section 29)
    scoring : str, optional
        Optimization metric (default is 'accuracy')

    Returns
    -------
    best_model : fitted model
        Model trained with best parameters
    best_params : dict
        Best parameters
    best_score : float
        Best score
    search_time : float
        Search duration (seconds)
    study : optuna.Study
        Optuna study object (for visualization)
    """

    print(f"\n{'=' * 60}")
    print("OPTIMIZATION WITH OPTUNA")
    print(f"{'=' * 60}")
    print(f"Model: {model_class.__name__}")
    print(f"Number of trials: {n_trials}")
    print("Smart search starting...\n")

    # Silence Optuna logs (keep display clean)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """
        Function that Optuna will optimize.
        For each trial, it suggests different parameters and returns the score.

        This is the heart of Optuna - it learns from each trial's results
        to make better suggestions for the next trial.
        """
        # Get parameter suggestions from search space
        params = param_space_func(trial)

        # Create model
        model = model_class(**params, random_state=42)

        # Calculate cross-validation score
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        return scores.mean()

    start_time = time.time()

    # Create study and optimize
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    search_time = time.time() - start_time

    # Train final model with best parameters
    best_params = study.best_params
    best_model = model_class(**best_params, random_state=42)
    best_model.fit(X, y)

    print(f"\nOptimization completed!")
    print(f"Duration: {search_time:.2f} seconds")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")

    return best_model, best_params, study.best_value, search_time, study


def compare_optimization_methods(grid_results, optuna_results, model_name):
    """
    Compares GridSearch and Optuna results side by side.

    This comparison helps us understand:
    - Which method found better parameters?
    - How much faster was one method over the other?
    - Are the results similar enough to trust the faster method?

    Parameters
    ----------
    grid_results : tuple
        (model, params, score, time) - GridSearch results
    optuna_results : tuple
        (model, params, score, time, study) - Optuna results
    model_name : str
        Model name

    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table
    """

    grid_model, grid_params, grid_score, grid_time = grid_results
    optuna_model, optuna_params, optuna_score, optuna_time, study = optuna_results

    print(f"\n{'=' * 80}")
    print(f"{model_name} - GRIDSEARCH vs OPTUNA COMPARISON")
    print(f"{'=' * 80}")

    comparison = pd.DataFrame({
        'Metric': ['Best Score', 'Time (seconds)', 'Speed Difference'],
        'GridSearchCV': [
            f"{grid_score:.4f}",
            f"{grid_time:.2f}",
            "Baseline"
        ],
        'Optuna': [
            f"{optuna_score:.4f}",
            f"{optuna_time:.2f}",
            f"{grid_time / optuna_time:.2f}x faster"
        ]
    })

    print("\n" + comparison.to_string(index=False))

    # Score difference
    score_diff = optuna_score - grid_score
    print(f"\nScore Difference: {score_diff:+.4f}")
    if abs(score_diff) < 0.005:
        print("→ Both methods found nearly the same score!")
    elif score_diff > 0:
        print("→ Optuna found a better score!")
    else:
        print("→ GridSearch found a better score!")

    # Parameter comparison
    print(f"\n{'=' * 60}")
    print("PARAMETER COMPARISON")
    print(f"{'=' * 60}")

    all_param_names = set(list(grid_params.keys()) + list(optuna_params.keys()))

    param_comparison = []
    for param in sorted(all_param_names):
        param_comparison.append({
            'Parameter': param,
            'GridSearch': grid_params.get(param, 'N/A'),
            'Optuna': optuna_params.get(param, 'N/A')
        })

    param_df = pd.DataFrame(param_comparison)
    print("\n" + param_df.to_string(index=False))

    return comparison


# ============================================================================
# RANDOM FOREST OPTIMIZATION - GRIDSEARCH
# ============================================================================

print("\n" + "=" * 80)
print("RANDOM FOREST OPTIMIZATION")
print("=" * 80)

# Parameter grid for GridSearch
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Optimize with GridSearch (29 features + Stratified CV)
rf_grid_results = optimize_with_gridsearch(
    X=X_final,
    y=y_final,
    model=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=selected_cv_strategy
)


# ============================================================================
# RANDOM FOREST OPTIMIZATION - OPTUNA
# ============================================================================

def rf_optuna_params(trial):
    """
    Optuna parameter space for Random Forest.

    This function tells Optuna what parameters to explore and their ranges.
    Optuna will intelligently sample from these ranges based on previous results.
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=100),
        'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15, None]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
    }


# Optimize with Optuna (29 features + Stratified CV)
rf_optuna_results = optimize_with_optuna(
    X=X_final,
    y=y_final,
    model_class=RandomForestClassifier,
    param_space_func=rf_optuna_params,
    n_trials=50,
    cv=selected_cv_strategy
)

# Compare methods
rf_comparison = compare_optimization_methods(
    grid_results=rf_grid_results,
    optuna_results=rf_optuna_results,
    model_name="Random Forest"
)

# Optuna visualizations
print("\n" + "=" * 80)
print("OPTUNA VISUALIZATIONS - RANDOM FOREST")
print("=" * 80)

rf_study = rf_optuna_results[4]

# 1. Optimization history
print("\n1. Optimization History")
print("   Shows the score of each trial")
fig1 = plot_optimization_history(rf_study)
fig1.update_layout(title="Random Forest - Optimization History")
fig1.show()

# 2. Parameter importances
print("\n2. Parameter Importances")
print("   Which parameters affect the score the most?")
fig2 = plot_param_importances(rf_study)
fig2.update_layout(title="Random Forest - Parameter Importances")
fig2.show()

# ============================================================================
# LOGISTIC REGRESSION OPTIMIZATION - GRIDSEARCH
# ============================================================================

print("\n\n" + "=" * 80)
print("LOGISTIC REGRESSION OPTIMIZATION")
print("=" * 80)

# Parameter grid for GridSearch
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Optimize with GridSearch (29 features + Stratified CV)
lr_grid_results = optimize_with_gridsearch(
    X=X_final,
    y=y_final,
    model=LogisticRegression(random_state=42, max_iter=1000),
    param_grid=lr_param_grid,
    cv=selected_cv_strategy
)


# ============================================================================
# LOGISTIC REGRESSION OPTIMIZATION - OPTUNA
# ============================================================================

def lr_optuna_params(trial):
    """Optuna parameter space for Logistic Regression"""
    return {
        'C': trial.suggest_float('C', 0.001, 100, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'liblinear',
        'max_iter': 1000
    }


# Optimize with Optuna (29 features + Stratified CV)
lr_optuna_results = optimize_with_optuna(
    X=X_final,
    y=y_final,
    model_class=LogisticRegression,
    param_space_func=lr_optuna_params,
    n_trials=30,
    cv=selected_cv_strategy
)

# Compare methods
lr_comparison = compare_optimization_methods(
    grid_results=lr_grid_results,
    optuna_results=lr_optuna_results,
    model_name="Logistic Regression"
)

# Optuna visualizations
print("\n" + "=" * 80)
print("OPTUNA VISUALIZATIONS - LOGISTIC REGRESSION")
print("=" * 80)

lr_study = lr_optuna_results[4]

# 1. Optimization history
fig3 = plot_optimization_history(lr_study)
fig3.update_layout(title="Logistic Regression - Optimization History")
fig3.show()

# 2. Parameter importances
fig4 = plot_param_importances(lr_study)
fig4.update_layout(title="Logistic Regression - Parameter Importances")
fig4.show()

# ============================================================================
# OVERALL COMPARISON AND MODEL SELECTION
# ============================================================================

print("\n\n" + "=" * 80)
print("FINAL MODEL SELECTION")
print("=" * 80)

# Collect all results
all_results = {
    'RF_GridSearch': rf_grid_results[2],
    'RF_Optuna': rf_optuna_results[2],
    'LR_GridSearch': lr_grid_results[2],
    'LR_Optuna': lr_optuna_results[2]
}

# Find best score
best_method = max(all_results, key=all_results.get)
best_score = all_results[best_method]

print("\nScores from All Methods:")
print("-" * 60)
for method, score in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{method:20s}: {score:.4f}")

print(f"\n{'=' * 60}")
print(f"BEST METHOD: {best_method}")
print(f"BEST SCORE: {best_score:.4f}")
print(f"{'=' * 60}")

# Select best model
if 'RF' in best_method:
    if 'Optuna' in best_method:
        final_model = rf_optuna_results[0]
        final_params = rf_optuna_results[1]
        print("\nFinal Model: Random Forest (optimized with Optuna)")
    else:
        final_model = rf_grid_results[0]
        final_params = rf_grid_results[1]
        print("\nFinal Model: Random Forest (optimized with GridSearch)")
else:
    if 'Optuna' in best_method:
        final_model = lr_optuna_results[0]
        final_params = lr_optuna_results[1]
        print("\nFinal Model: Logistic Regression (optimized with Optuna)")
    else:
        final_model = lr_grid_results[0]
        final_params = lr_grid_results[1]
        print("\nFinal Model: Logistic Regression (optimized with GridSearch)")

print(f"Final Parameters: {final_params}")

print("\n" + "=" * 80)
print("SECTION 30 COMPLETED!")
print("=" * 80)
print("\nKey Takeaways:")
print("1. Optuna is usually much faster than GridSearch")
print("2. Both methods can achieve similar scores")
print("3. Optuna finds good results with fewer trials")
print("4. GridSearch is guaranteed but slow, Optuna is fast but might occasionally miss global optimum")

############################
# Section 31: Final Model
############################

print("\n" + "=" * 80)
print("SECTION 31: FINAL MODEL")
print("=" * 80)

# ============================================================================
# PREPARE RESULTS FROM SECTION 30
# ============================================================================

print("\nCollecting optimization results from Section 30...")

# Unpack GridSearch results
best_rf_grid, rf_grid_params, best_rf_grid_score, rf_grid_time = rf_grid_results
best_lr_grid, lr_grid_params, best_lr_grid_score, lr_grid_time = lr_grid_results

# Unpack Optuna results
best_rf_optuna, rf_optuna_params, best_rf_optuna_score, rf_optuna_time, rf_study = rf_optuna_results
best_lr_optuna, lr_optuna_params, best_lr_optuna_score, lr_optuna_time, lr_study = lr_optuna_results

print("All optimization results collected successfully!")

# Compare all scores
print("\n" + "=" * 60)
print("SCORES FROM ALL OPTIMIZATION METHODS")
print("=" * 60)

all_scores = {
    'RF_GridSearch': best_rf_grid_score,
    'RF_Optuna': best_rf_optuna_score,
    'LR_GridSearch': best_lr_grid_score,
    'LR_Optuna': best_lr_optuna_score
}

# Print scores in sorted order
for method, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{method:20s}: {score:.4f}")

# Find best method
best_method = max(all_scores, key=all_scores.get)
best_score = all_scores[best_method]

print(f"\n{'=' * 60}")
print(f"BEST METHOD: {best_method}")
print(f"BEST SCORE: {best_score:.4f}")
print(f"{'=' * 60}")

# Select best model and parameters
if best_method == 'RF_GridSearch':
    final_model = best_rf_grid
    final_params = rf_grid_params
    print("\nFinal Model: Random Forest (optimized with GridSearch)")
elif best_method == 'RF_Optuna':
    final_model = best_rf_optuna
    final_params = rf_optuna_params
    print("\nFinal Model: Random Forest (optimized with Optuna)")
elif best_method == 'LR_GridSearch':
    final_model = best_lr_grid
    final_params = lr_grid_params
    print("\nFinal Model: Logistic Regression (optimized with GridSearch)")
else:  # LR_Optuna
    final_model = best_lr_optuna
    final_params = lr_optuna_params
    print("\nFinal Model: Logistic Regression (optimized with Optuna)")

print(f"Final Parameters: {final_params}")


# ============================================================================
# DETAILED FINAL MODEL EVALUATION
# ============================================================================

def evaluate_final_model(model, X, y, cv):
    """
    Performs detailed evaluation of the final model.

    This comprehensive evaluation helps us understand:
    - How reliable is our model? (through cross-validation)
    - How well does it perform on different metrics?
    - Where does it make mistakes? (through confusion matrix)

    Think of this like a final exam for our model - we're testing it
    thoroughly from multiple angles to ensure it's truly ready for
    deployment on the test set.

    Parameters
    ----------
    model : fitted sklearn model
        Model to evaluate
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        Target variable
    cv : cross-validation strategy
        Cross-validation strategy (from Section 29)

    Returns
    -------
    results : dict
        Evaluation results
    """

    print("\n" + "=" * 60)
    print("DETAILED FINAL MODEL EVALUATION")
    print("=" * 60)

    # Cross-validation scores (using Stratified K-Fold)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Metrics
    results = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }

    # Print results
    print(f"\nModel: {model.__class__.__name__}")
    print("-" * 60)
    print(f"Cross-Validation Accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
    print(f"Training Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show(block=True)

    return results


# Final model evaluation (29 features + Stratified CV)
final_results = evaluate_final_model(
    model=final_model,
    X=X_final,
    y=y_final,
    cv=selected_cv_strategy
)

print("\n" + "=" * 80)
print("SECTION 31 COMPLETED!")
print("=" * 80)

############################
# Section 32: Base vs Final Model Comparison
############################

print("\n" + "=" * 80)
print("SECTION 32: BASE MODEL vs FINAL MODEL COMPARISON")
print("=" * 80)


def compare_models(base_results, final_results, base_model_name="Base Model",
                   final_model_name="Final Model", show_improvement=True):
    """
    Compares base model with final model performance.

    This is the moment of truth! After all our hard work on:
    - Feature engineering (Section 18)
    - Feature selection (Section 27)
    - Hyperparameter optimization (Section 30)

    Did it all pay off? This comparison answers:
    - Did feature engineering help?
    - Did feature selection contribute?
    - Did hyperparameter optimization make a difference?

    Think of this like comparing your first draft of an essay with your
    final polished version - we want to see how much we improved!

    Parameters
    ----------
    base_results : dict
        Base model results (metrics)
    final_results : dict
        Final model results (metrics)
    base_model_name : str, optional
        Base model name (default is "Base Model")
    final_model_name : str, optional
        Final model name (default is "Final Model")
    show_improvement : bool, optional
        Show improvement percentages (default is True)

    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table
    """

    # Create comparison table
    metrics = ['cv_mean', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    comparison_data = []
    for metric in metrics:
        base_value = base_results.get(metric, 0)
        final_value = final_results.get(metric, 0)
        improvement = final_value - base_value
        improvement_pct = (improvement / base_value * 100) if base_value > 0 else 0

        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            base_model_name: base_value,
            final_model_name: final_value,
            'Improvement': improvement,
            'Improvement %': improvement_pct
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Print table
    print("\nPERFORMANCE COMPARISON")
    print("-" * 80)
    print(comparison_df.to_string(index=False))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Metric comparison bar chart
    ax1 = axes[0, 0]
    x = range(len(comparison_df))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], comparison_df[base_model_name],
            width, label=base_model_name, alpha=0.8)
    ax1.bar([i + width / 2 for i in x], comparison_df[final_model_name],
            width, label=final_model_name, alpha=0.8)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['Metric'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Improvement percentages
    ax2 = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in comparison_df['Improvement %']]
    ax2.barh(comparison_df['Metric'], comparison_df['Improvement %'], color=colors, alpha=0.7)
    ax2.set_xlabel('Improvement %', fontsize=12)
    ax2.set_title('Performance Improvement', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    # 3. CV Accuracy comparison (more detailed)
    ax3 = axes[1, 0]
    categories = ['CV Accuracy', 'Training Accuracy', 'ROC-AUC']
    base_values = [base_results['cv_mean'], base_results['accuracy'], base_results['roc_auc']]
    final_values = [final_results['cv_mean'], final_results['accuracy'], final_results['roc_auc']]

    x_pos = range(len(categories))
    ax3.plot(x_pos, base_values, 'o-', label=base_model_name, linewidth=2, markersize=8)
    ax3.plot(x_pos, final_values, 's-', label=final_model_name, linewidth=2, markersize=8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Main Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0.7, 1.0])

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    COMPARISON SUMMARY
    {'=' * 40}

    Base Model: {base_model_name}
    Final Model: {final_model_name}

    Highest Improvement:
    {comparison_df.nlargest(1, 'Improvement %')['Metric'].values[0]}: 
    {comparison_df.nlargest(1, 'Improvement %')['Improvement %'].values[0]:+.2f}%

    CV Accuracy:
    Base:  {base_results['cv_mean']:.4f}
    Final: {final_results['cv_mean']:.4f}
    Diff:  {final_results['cv_mean'] - base_results['cv_mean']:+.4f}

    ROC-AUC:
    Base:  {base_results['roc_auc']:.4f}
    Final: {final_results['roc_auc']:.4f}
    Diff:  {final_results['roc_auc'] - base_results['roc_auc']:+.4f}
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.show(block=True)

    # Result interpretation
    print("\n" + "=" * 80)
    print("RESULT INTERPRETATION")
    print("=" * 80)

    avg_improvement = comparison_df['Improvement %'].mean()

    if avg_improvement > 5:
        print(f"\n✓ Average {avg_improvement:.2f}% improvement achieved!")
        print("  Feature engineering and optimization were successful!")
    elif avg_improvement > 2:
        print(f"\n✓ Average {avg_improvement:.2f}% improvement achieved.")
        print("  Reasonable progress observed.")
    elif avg_improvement > 0:
        print(f"\n~ Average {avg_improvement:.2f}% improvement achieved.")
        print("  Small but positive progress.")
    else:
        print(f"\n✗ Average {avg_improvement:.2f}% change.")
        print("  Final model did not outperform base model.")
        print("  Feature engineering or model selection should be reviewed.")

    return comparison_df


# Prepare base model results (from Section 17)
base_model_results = {
    'cv_mean': 0.8202,  # Get this value from Section 17 output
    'accuracy': 0.8501,
    'precision': 0.8421,
    'recall': 0.7368,
    'f1': 0.7857,
    'roc_auc': 0.8900
}

# Final model results (from Section 31 - final_results variable already exists)

# Perform comparison
comparison_results = compare_models(
    base_results=base_model_results,
    final_results=final_results,
    base_model_name="Base Model (Section 17)",
    final_model_name="Final Model (Section 31)"
)

print("\n" + "=" * 80)
print("BASE vs FINAL MODEL COMPARISON COMPLETED!")
print("=" * 80)

############################
# Section 33: Predictions on Test Data
############################

print("\n" + "=" * 80)
print("SECTION 33: PREDICTIONS ON TEST DATA")
print("=" * 80)

# Prepare test data (from df_cleaned)
test_data = df_cleaned[df_cleaned['is_train'] == 0].copy()

print(f"Test data size: {test_data.shape}")

# Prepare test data with selected features (29 features)
X_test = test_data[selected_features_final]

print(f"Test features shape: {X_test.shape}")
print(f"Number of features used: {len(selected_features_final)} (29 features)")

# Make predictions (using final_model from Section 31)
test_predictions = final_model.predict(X_test)
test_predictions_proba = final_model.predict_proba(X_test)[:, 1]

print(f"\nPredicted survivors: {test_predictions.sum()}")
print(f"Predicted non-survivors: {len(test_predictions) - test_predictions.sum()}")
print(f"Survival rate: {(test_predictions.mean() * 100):.2f}%")

# Prediction distribution
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(test_predictions_proba, bins=20, edgecolor='black', alpha=0.7)
plt.title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Survival Probability')
plt.ylabel('Frequency')
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='Threshold (0.5)')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
pd.Series(test_predictions).value_counts().plot(kind='bar', color=['steelblue', 'coral'])
plt.title('Prediction Results', fontsize=12, fontweight='bold')
plt.xlabel('Survived (0=Died, 1=Survived)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show(block=True)

print("\n" + "=" * 80)
print("SECTION 33 COMPLETED!")
print("=" * 80)
print(f"✅ Predictions made on test data")
print(f"✅ Survival predictions ready for 418 passengers")
print(f"✅ Will be submitted to Kaggle in Section 34")

############################
# Section 34: Kaggle Submission
############################

print("\n" + "=" * 80)
print("SECTION 34: KAGGLE SUBMISSION")
print("=" * 80)


def create_submission(passenger_ids, predictions, filename='submission.csv'):
    """
    Creates Kaggle submission file.

    This is the final step of our journey! We're packaging our predictions
    into the format that Kaggle expects. It's like wrapping a gift - the
    content (our predictions) is ready, now we just need to package it properly.

    Kaggle expects a CSV file with exactly two columns:
    - PassengerId: The ID of each passenger in the test set
    - Survived: Our prediction (0 or 1)

    Parameters
    ----------
    passenger_ids : array-like
        PassengerId values
    predictions : array-like
        Predictions (0 or 1)
    filename : str, optional
        Name of file to save (default is 'submission.csv')

    Returns
    -------
    submission : pd.DataFrame
        Submission DataFrame
    """

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions.astype(int)
    })

    submission.to_csv(filename, index=False)

    print(f"\nSubmission file created: {filename}")
    print(f"Number of rows: {len(submission)}")
    print("\nFirst 5 rows:")
    print(submission.head())
    print("\nLast 5 rows:")
    print(submission.tail())

    print(f"\nSurvived data type: {submission['Survived'].dtype}")
    print(f"Unique values: {submission['Survived'].unique()}")

    return submission


# Get PassengerIds from test data
test_passenger_ids = test_df['PassengerId'].values

print(f"PassengerId range: {test_passenger_ids.min()} - {test_passenger_ids.max()}")
print(f"Total test samples: {len(test_passenger_ids)}")

# Create submission
submission = create_submission(
    passenger_ids=test_passenger_ids,
    predictions=test_predictions,
    filename='titanic_submission.csv'
)

# Submission summary
print("\n" + "=" * 80)
print("SUBMISSION SUMMARY")
print("=" * 80)
print(f"File name: titanic_submission.csv")
print(f"Total predictions: {len(submission)}")
print(f"Predicted survivors: {submission['Survived'].sum()} ({submission['Survived'].mean() * 100:.2f}%)")
print(f"Predicted deaths: {(submission['Survived'] == 0).sum()} ({(1 - submission['Survived'].mean()) * 100:.2f}%)")

print("\n" + "=" * 80)
print("ENTIRE PROCESS COMPLETED!")
print("=" * 80)
print(f"\nFinal Model: {final_model.__class__.__name__}")
print(f"Optimization Method: GridSearchCV")
print(f"Number of Features Used: {len(selected_features_final)} (29 features)")
print(f"Cross-Validation Accuracy: {final_results['cv_mean']:.4f}")
print(f"ROC-AUC Score: {final_results['roc_auc']:.4f}")
print(f"Submission File: titanic_submission.csv")
print("\n" + "=" * 80)
print("Ready to upload to Kaggle!")
print("=" * 80)