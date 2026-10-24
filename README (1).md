# E-Commerce Customer Intelligence & Churn Prediction

An end-to-end machine learning project focused on understanding e-commerce customer behavior, identifying factors associated with customer churn, and applying multiple machine learning algorithms for prediction and analysis.

**Author:** Jaweria Afzal

---

## 📌 Project Overview

Customer churn is an important business problem for e-commerce companies because retaining existing customers can be more valuable than continuously acquiring new ones.

This project analyzes customer demographic information, engagement behavior, purchasing activity, customer service interactions, and financial characteristics to understand customer behavior and predict whether a customer is likely to churn.

The project demonstrates multiple machine learning techniques, including:

- Linear Regression
- Gradient Descent
- Ridge Regression
- Lasso Regression
- Logistic Regression
- Decision Tree

The goal is not only to build predictive models, but also to understand the data, visualize important patterns, interpret model behavior, and create a complete machine learning workflow.

---

## 🎯 Objectives

1. Understand customer behavior in an e-commerce environment.
2. Clean and preprocess customer data.
3. Perform exploratory data analysis (EDA).
4. Identify relationships between customer characteristics and churn.
5. Engineer meaningful features from existing customer behavior variables.
6. Build regression models for predicting customer lifetime value.
7. Implement Gradient Descent to understand the optimization process.
8. Build classification models for customer churn prediction.
9. Visualize and interpret a Decision Tree.
10. Compare different machine learning algorithms.

---

## 📊 Dataset

The project uses the **Ecommerce Customer Behavior Dataset** from Kaggle.

The dataset contains customer-level information covering demographics, engagement, purchasing behavior, customer service interactions, and customer value.

### Dataset Features

| Feature | Description |
|---|---|
| `Age` | Customer age |
| `Gender` | Customer gender |
| `Country` | Customer country |
| `City` | Customer city |
| `Membership_Years` | Number of years as a customer/member |
| `Login_Frequency` | Customer login frequency |
| `Session_Duration_Avg` | Average session duration |
| `Pages_Per_Session` | Average pages viewed per session |
| `Cart_Abandonment_Rate` | Rate at which carts are abandoned |
| `Wishlist_Items` | Number of wishlist items |
| `Total_Purchases` | Total number of purchases |
| `Average_Order_Value` | Average value of an order |
| `Days_Since_Last_Purchase` | Days since the customer's last purchase |
| `Discount_Usage_Rate` | Rate of discount usage |
| `Returns_Rate` | Customer return rate |
| `Email_Open_Rate` | Email open rate |
| `Customer_Service_Calls` | Number of customer service calls |
| `Product_Reviews_Written` | Number of product reviews written |
| `Social_Media_Engagement_Score` | Social media engagement score |
| `Mobile_App_Usage` | Mobile application usage |
| `Payment_Method_Diversity` | Diversity of payment methods used |
| `Lifetime_Value` | Customer lifetime value |
| `Credit_Balance` | Customer credit balance |
| `Churned` | Target variable indicating customer churn |
| `Signup_Quarter` | Quarter in which the customer signed up |

### Dataset Source

Kaggle: [Ecommerce Customer Behavior Dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset)

> The raw dataset is not included in this repository. Download it from Kaggle if you want to reproduce the analysis.

---

## 🧠 Machine Learning Problems

### 1. Regression

The regression task focuses on predicting:

**Target:** `Lifetime_Value`

Models explored:

- Linear Regression
- Gradient Descent
- Ridge Regression
- Lasso Regression

### 2. Classification

The classification task focuses on predicting:

**Target:** `Churned`

Models explored:

- Logistic Regression
- Decision Tree Classifier

---

## 🔄 Project Workflow

```text
Raw Customer Data
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Data Preprocessing
        ↓
Regression & Classification
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Interpretation
```

---

## 🧹 Data Cleaning

The data-cleaning stage includes:

- Inspecting the dataset structure
- Checking rows and columns
- Reviewing data types
- Checking missing values
- Checking duplicate records
- Handling missing numerical values
- Handling missing categorical values
- Verifying the cleaned dataset

---

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns related to churn.

The analysis includes:

- Churn distribution
- Numerical feature distributions
- Boxplots
- Correlation matrix
- Correlation heatmap
- Feature-to-churn correlation
- Customer behavior scatter plots

Particular attention was given to:

- `Lifetime_Value`
- `Days_Since_Last_Purchase`
- `Total_Purchases`
- `Login_Frequency`
- `Customer_Service_Calls`

---

## 🛠️ Feature Engineering

Additional behavioral features were created to capture relationships between existing variables.

### Purchases Per Year

```python
Purchases_Per_Year = Total_Purchases / (Membership_Years + 1)
```

This provides an approximate measure of purchasing frequency relative to membership duration.

### Pages Per Minute

```python
Pages_Per_Minute = Pages_Per_Session / (Session_Duration_Avg + 1)
```

This provides an additional measure of browsing intensity.

The `+1` helps prevent division-by-zero problems.

---

## ⚙️ Data Preprocessing

The dataset is divided into training and testing sets using an **80/20 split**.

For the churn classification task, stratified splitting is used to preserve the approximate class distribution.

### Numerical Features

Numerical variables are standardized using:

`StandardScaler`

### Categorical Features

Categorical variables are transformed using:

`OneHotEncoder`

with:

```python
handle_unknown="ignore"
```

A `ColumnTransformer` and machine learning `Pipeline` are used to organize preprocessing and model training.

---

# 🤖 Machine Learning Models

## 1. Linear Regression

Linear Regression is used as a baseline regression algorithm for predicting customer lifetime value.

**Target:** `Lifetime_Value`

---

## 2. Gradient Descent

Gradient Descent is used to understand how model parameters can be optimized iteratively by minimizing a loss function.

The project demonstrates the underlying optimization concept rather than treating the model as a black box.

---

## 3. Ridge Regression

Ridge Regression extends Linear Regression by applying **L2 regularization**.

It is useful when:

- Features are correlated.
- Model coefficients may become large.
- We want to control model complexity.

---

## 4. Lasso Regression

Lasso Regression applies **L1 regularization**.

One useful property of Lasso is that it can shrink some coefficients toward zero, which can help with feature selection and interpretability.

---

## 5. Logistic Regression

Logistic Regression is used for the customer churn classification problem.

**Target:** `Churned`

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 6. Decision Tree

A Decision Tree Classifier is used to model customer churn.

Decision Trees are useful for interpretability because their decision-making process can be represented visually.

The project includes a visualization of the trained Decision Tree.

---

# 📏 Model Evaluation

### Regression Metrics

Regression models can be evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification Metrics

Classification models can be evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

For churn prediction, Precision and Recall are considered alongside Accuracy because different prediction errors can have different business consequences.

---

# 📊 Results

Final numerical results should be added directly from the completed notebook rather than estimated.

### Regression Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | — | — | — |
| Gradient Descent | — | — | — |
| Ridge Regression | — | — | — |
| Lasso Regression | — | — | — |

### Classification Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | — | — | — | — |
| Decision Tree | — | — | — | — |

> Replace the dashes with the actual values generated by the notebook.

---

# 📉 Visualizations

The project includes visualizations such as:

- Churn distribution
- Numerical feature distributions
- Boxplots
- Correlation heatmap
- Feature-to-churn correlation
- Customer behavior scatter plots
- Confusion matrix
- Decision Tree visualization

Recommended repository location:

```text
images/
├── churn_distribution.png
├── correlation_heatmap.png
├── confusion_matrix.png
└── decision_tree.png
```

Example:

```markdown
![Correlation Heatmap](images/correlation_heatmap.png)
```

---

# 🔍 Business Questions

This project investigates questions such as:

- What customer characteristics are associated with churn?
- Does purchasing frequency relate to customer retention?
- Does customer engagement differ between churned and active customers?
- Does time since the last purchase provide useful information about churn?
- How do customer service interactions relate to churn?
- Which variables are useful for predicting customer lifetime value?
- Which features are most influential in the Decision Tree?

---

# 💡 Business Value

A churn prediction system can help an e-commerce company:

- Identify customers who may be at risk of leaving.
- Prioritize retention campaigns.
- Personalize offers and promotions.
- Improve customer engagement.
- Identify behavioral patterns associated with churn.
- Allocate customer retention resources more efficiently.

This project connects machine learning techniques with a practical business use case rather than focusing only on model accuracy.

---

# 🧰 Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Machine Learning

- Linear Regression
- Gradient Descent
- Ridge Regression
- Lasso Regression
- Logistic Regression
- Decision Tree

### Development Tools

- Jupyter Notebook
- Git
- GitHub

---

# 📁 Project Structure

```text
ecommerce-customer-intelligence-churn-prediction/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── all_algorithum.ipynb
│
├── data/
│   └── README.md
│
├── reports/
│   └── project_report.pdf
│
├── images/
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── decision_tree.png
│
└── src/
    └── README.md
```

---

# 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-customer-intelligence-churn-prediction.git
```

```bash
cd ecommerce-customer-intelligence-churn-prediction
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scriptsctivate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the dataset from Kaggle and place the CSV file in the appropriate data directory.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/all_algorithum.ipynb
```

Run the notebook cells sequentially.

---

# 🔮 Future Improvements

Possible future extensions include:

- Hyperparameter tuning
- Cross-validation
- Random Forest
- Gradient Boosting
- XGBoost
- ROC-AUC analysis
- Precision-Recall curves
- Feature importance analysis
- Customer segmentation
- Interactive Streamlit dashboard
- Automated churn-risk scoring
- API deployment
- Model monitoring and retraining

---

# ⚠️ Limitations

- The dataset represents historical customer behavior.
- Correlation does not imply causation.
- Model performance may change on real-world data.
- Churn predictions depend on the quality and availability of customer behavior data.
- Additional validation would be required before using such a model in a production business environment.

---

# 📚 Learning Outcomes

This project demonstrates:

- Data loading and inspection
- Data cleaning
- Missing-value handling
- Exploratory Data Analysis
- Data visualization
- Correlation analysis
- Feature engineering
- Train/test splitting
- Standardization
- One-hot encoding
- Regression
- Classification
- Gradient Descent
- Regularization
- Model evaluation
- Confusion matrices
- Decision Tree interpretation
- Machine learning pipelines

---
## Live Demo
[Click here to use the app](https://customer-churn-prediction-hu7ccwfijengqqecrr35gd.streamlit.app/)
# 👩‍💻 Author

**Jaweria Afzal**

This project was developed as part of my machine learning learning journey and portfolio development.

---

# ⭐ Acknowledgements

- Kaggle — Ecommerce Customer Behavior Dataset
- Python open-source ecosystem
- Scikit-learn community
- Jupyter Project

---

## 📌 Project Status

**Status: Completed — Initial ML Pipeline**

The core data analysis, feature engineering, regression/classification experiments, and model evaluation are implemented in the Jupyter Notebook.

Future improvements can focus on advanced models, visualization, customer segmentation, and deployment.
