# Titanic Passenger Survival Prediction

## Overview

This project aims to build a machine learning model that predicts whether a passenger on the Titanic survived or not. The prediction is based on various features available in the Titanic dataset, such as `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`, and `Embarked`.

## Project Structure

The core of this project is a Jupyter Notebook that walks through the entire machine learning pipeline:

* **`Task-1 Titanic_Survival_Prediction.ipynb`**: This notebook contains all the code for data loading, exploration, preprocessing, model training, evaluation, and prediction.

## Dataset

The dataset used in this project is the classic Titanic dataset, which is a common dataset for introductory machine learning tasks. It contains information about Titanic passengers, including their survival status.

## Key Steps and Findings

The following steps were performed in the analysis and model building:

### 1. Data Loading and Initial Exploration
* The `Titanic-Dataset.csv` file was loaded into a pandas DataFrame.
* Initial inspection revealed the structure and data types of the columns, along with the presence of missing values.

### 2. Data Preprocessing and Cleaning
* **Missing Values Handling**:
    * Missing values in the 'Age' column were imputed with the median age.
    * Missing values in the 'Embarked' column were imputed with the mode (most frequent) embarkation point.
    * The 'Cabin' column was dropped due to a significant number of missing values, which would make it difficult to use effectively without substantial imputation or feature engineering.
* **Categorical Feature Encoding**:
    * Categorical features such as 'Sex' and 'Embarked' were converted into numerical representations using one-hot encoding (`pd.get_dummies`). This created new columns like 'Sex_male', 'Embarked_Q', and 'Embarked_S'.
* The `Name`, `PassengerId`, and `Ticket` columns were not used for model training as they were deemed irrelevant for predicting survival in this context.

### 3. Data Splitting
* The preprocessed dataset was split into training and and testing sets.
* `Survived` was designated as the target variable (`y`), and the remaining relevant features formed the input features (`X`).
* A 75%-25% train-test split was used, with 75% of the data for training and 25% for testing.

### 4. Model Building and Training
* A **Logistic Regression** model was chosen for this classification task.
* The model was trained on the training data (`X_train` and `y_train`) using the selected features: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex_male`, `Embarked_Q`, and `Embarked_S`.

### 5. Model Evaluation
* The trained model's performance was assessed on the unseen testing data.
* The model achieved an **accuracy of approximately 80.7%**.

### 6. Predictions on New Data
* The trained model was used to predict the survival status of a few hypothetical new passengers, demonstrating its predictive capability.


## Future Enhancements

* **Explore other models**: Evaluate the performance of other classification algorithms like Decision Trees, Random Forests, Support Vector Machines, or Gradient Boosting.
* **Feature Engineering**: Create new features from existing ones (e.g., family size from SibSp and Parch, title from Name) to potentially improve model performance.
* **Hyperparameter Tuning**: Optimize the hyperparameters of the chosen model to achieve better performance.
* **Cross-validation**: Implement cross-validation for more robust model evaluation.
* **Detailed Evaluation Metrics**: Include precision, recall, F1-score, and a confusion matrix for a more comprehensive understanding of model performance.

## Author

* **Tushar Surja**
* **Batch**: June 2025 batch B33
* **Domain**: Data Science
