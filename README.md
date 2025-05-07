# Stats-Assignments

This repository contains code and datasets for various statistics and machine learning assignments and exercises, including a project focused on house price prediction. The analyses cover data exploration, statistical testing, data preprocessing, model training, evaluation, and interpretation.

## Contents

This repository includes the following main components:

1.  **Jupyter Notebooks:**
    * **`Stats_Assignment_1.ipynb`**: Analyzes student performance factors. It performs descriptive statistics and creates visualizations like histograms, box plots, and scatter plots to explore correlations with exam scores.
    * **`Stats_Assignment_2.ipynb`**: Performs Exploratory Data Analysis (EDA) on the Iris dataset, including visualizations such as pair plots.
    * **`Stats_Assignment_3.ipynb`**: Conducts statistical tests, specifically an independent t-test and a one-way ANOVA, on the Iris dataset to compare petal lengths and widths between species.
    * **`ML_Assignment_4.ipynb`**: Analyzes the Music & Mental Health Survey dataset. This notebook includes extensive data preprocessing such as handling missing values using various techniques (KNNImputation, forward fill, and regression-based imputation), outlier removal, and feature scaling with MinMaxScaler. It also covers feature selection using Decision Trees, training and evaluation of Linear Regression models, application of L2 (Ridge) and L1 (Lasso) regularization, Polynomial Regression, and model interpretation using SHAP.
    * **`Exercise_Day_8_Tree_and_KNN_classification.ipynb`**: Continues analysis on the Music & Mental Health Survey dataset. It involves training, evaluating, and comparing Decision Tree, Random Forest, Gradient Boosting, and K-Nearest Neighbors (KNN) classifiers. This notebook also includes a process for finding the optimal 'k' value for the KNN classifier.
    * **`Project_Work.ipynb`**: Focuses on predicting house prices in King County, USA. The work includes EDA, handling outliers using the IQR method, applying log transformation to the price, feature engineering by creating a `price_per_sqft` feature, and training/evaluating Linear Regression, Random Forest, and XGBoost models. Additionally, it implements a weighted ensemble method and analyzes feature importance.

2.  **Data Folder (`Exercise-Day_7-Group-Exercise/`):**
    * `mxmh_survey_results.csv`: Dataset containing survey results on music listening habits and mental health.
    * `Exercise_7.py`: Python script likely performing analysis related to the Music & Mental Health survey.
    * `force_plot_target_*.html`: These are SHAP force plots generated for model explanation purposes.

3.  **Other Datasets (loaded within notebooks):**
    * `StudentPerformanceFactors.csv`
    * Iris dataset
    * `kc_house_data.csv`

## Key Libraries Used

* pandas
* numpy
* scikit-learn (sklearn)
* matplotlib
* seaborn
* xgboost
* shap
* scipy

## Analyses Performed

* **Descriptive Statistics:** Calculation of mean, median, standard deviation, and variance.
* **Exploratory Data Analysis (EDA):** Creation of visualizations including histograms, box plots, scatter plots, pair plots, and correlation heatmaps.
* **Statistical Tests:** Application of independent t-tests and one-way ANOVA.
* **Data Preprocessing:**
    * Missing Value Imputation: Techniques include K-Nearest Neighbors Imputation, forward fill, and imputation using RandomForestRegressor/Classifier.
    * Outlier Handling: Primarily using the Interquartile Range (IQR) method.
    * Feature Scaling: Using StandardScaler and MinMaxScaler.
    * Feature Engineering: Creation of new features from existing ones, such as `price_per_sqft`.
    * Encoding Categorical Features: Using OneHotEncoder (via `pd.get_dummies`) and LabelEncoder.
* **Machine Learning Models:**
    * *Regression:* Development and evaluation of Linear Regression, RandomForestRegressor, XGBoostRegressor, Polynomial Regression, Ridge Regression (L2), and Lasso Regression (L1) models. A weighted ensemble method for regression is also implemented.
    * *Classification:* Development and evaluation of Decision Tree, Random Forest, Gradient Boosting, and K-Nearest Neighbors (KNN) classifiers.
* **Model Evaluation:** Assessment of model performance using metrics like Mean Squared Error (MSE), R-squared, Accuracy, Precision, Recall, F1-score, and utilization of Cross-Validation techniques.
* **Model Interpretation:** Analysis of feature importance, generation of residual plots, and application of SHAP (SHapley Additive exPlanations) value analysis.

## Datasets

* `StudentPerformanceFactors.csv`: Contains data related to student study habits, attendance, and exam scores.
* Iris Dataset: The classic dataset for classification tasks, based on measurements of different iris flower species.
* `mxmh_survey_results.csv`: Includes survey data on music preferences, listening habits, and self-reported mental health metrics.
* `kc_house_data.csv`: Features King County, USA house sales data, used for house price prediction tasks.

## How to Run

The analyses are primarily conducted within the Jupyter notebooks (`.ipynb` files). To run these:
1.  Ensure you have a Python environment with Jupyter Notebook/JupyterLab or Google Colab set up.
2.  Install the required libraries (listed above), typically using pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap scipy
    ```
3.  Open the desired notebook file in your Jupyter environment.
4.  Execute the cells sequentially.
5.  For notebooks requiring external CSV files, ensure the respective CSV files (`StudentPerformanceFactors.csv`, `mxmh_survey_results.csv`, `kc_house_data.csv`) are placed in the correct directory relative to the notebook, or update the file paths within the notebook code. If using Google Colab, you may need to upload the files using the provided code cells.
