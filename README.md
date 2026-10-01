# 🛍️ Customer Spending Prediction using Linear Regression

A machine learning project that predicts a customer's **Spending Score (1–100)** using demographic and income-related features from the **Mall Customers Dataset**.

The main goal of this project was not only to build a regression model, but also to practice the complete machine learning workflow: understanding the dataset, selecting features, preparing the data, training a model, making predictions, and evaluating the results.

---

## 📌 Project Overview

In this project, I used **Linear Regression** to predict a customer's Spending Score based on:

* Genre
* Age
* Annual Income (k$)

The target variable is:

**Spending Score (1–100)**

Since the target is a numerical value, this project is treated as a **regression problem**.

---

## 📂 Dataset

The project uses the **Mall Customers Dataset**, which contains information about 200 customers.

The original dataset includes:

| Feature                | Description                             |
| ---------------------- | --------------------------------------- |
| CustomerID             | Unique identifier for each customer     |
| Genre                  | Customer gender                         |
| Age                    | Customer age                            |
| Annual Income (k$)     | Annual income in thousands of dollars   |
| Spending Score (1–100) | Spending score assigned to the customer |

`CustomerID` was removed because it is an identifier and does not provide useful predictive information for this model.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Checked the dataset for missing values.
2. Removed the `CustomerID` column.
3. Converted the categorical `Genre` column into numerical values:

```text
Male → 0
Female → 1
```

4. Selected the input features:

```text
Genre
Age
Annual Income (k$)
```

5. Selected `Spending Score (1–100)` as the target variable.

---

## 🔎 Exploratory Data Analysis

Before training the model, I explored the relationships between the selected features and the target.

### Average Spending Score by Genre

The average Spending Score was calculated for each genre.

The results showed a small difference between the two groups:

* Male: approximately **48.51**
* Female: approximately **51.53**

This difference was treated as an observation from this dataset, not as a general conclusion about gender and spending behavior.

### Age and Spending Score

The correlation between Age and Spending Score was approximately:

```text
-0.327
```

This indicates a weak-to-moderate negative linear relationship in this dataset.

### Annual Income and Spending Score

The correlation between Annual Income and Spending Score was approximately:

```text
0.0099
```

This is extremely close to zero, indicating almost no linear relationship between these two variables in this dataset.

---

## 🤖 Machine Learning Model

### Linear Regression

I used **Linear Regression** because the target variable is numerical and the goal is to predict a numerical value.

The model was trained using:

```python
from sklearn.linear_model import LinearRegression
```

The dataset was divided into training and testing sets using:

```python
train_test_split()
```

with:

```text
Test size = 20%
Random state = 50
```

---

## 📊 Model Evaluation

The model was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R-squared (R²)

### Results

| Metric | Result |
| ------ | -----: |
| MAE    |  20.46 |
| MSE    | 598.52 |
| R²     | 0.0759 |

### Interpretation

#### Mean Absolute Error (MAE)

The MAE was approximately:

```text
20.46
```

This means that, on average, the model's predictions were about **20.46 Spending Score points away from the actual values**.

Since the Spending Score ranges from 1 to 100, this represents a relatively large prediction error.

#### Mean Squared Error (MSE)

The MSE was:

```text
598.52
```

MSE gives more weight to larger prediction errors because the errors are squared.

#### R-squared (R²)

The R² score was:

```text
0.0759
```

This means that the selected features explain only a small portion of the variation in Spending Score for this model.

The result suggests that **Genre, Age, and Annual Income alone are not sufficient to accurately predict Spending Score using this Linear Regression model**.

---

## 💡 Key Findings

This project demonstrated an important machine learning concept:

> A reasonable-looking feature does not necessarily become a useful predictive feature.

For example, Annual Income might seem logically related to customer spending. However, the correlation in this dataset was approximately `0.0099`, showing almost no linear relationship.

The model results also showed that the selected features were not enough to produce strong predictions.

Instead of hiding or changing the results, the actual model performance is reported as part of the project.

---

## 🧠 What I Learned

Through this project, I practiced:

* Loading datasets using Pandas
* Checking for missing values
* Removing unnecessary columns
* Encoding categorical variables
* Selecting features and target variables
* Exploratory Data Analysis (EDA)
* Using `groupby()`
* Using `corr()`
* Understanding correlation
* Choosing between classification and regression
* Splitting data into training and testing sets
* Training a Linear Regression model
* Making predictions
* Evaluating regression models
* Interpreting MAE, MSE, and R²
* Understanding that model performance depends heavily on the selected features

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn

---

## 📁 Project Structure

```text
customer-spending-prediction/
│
├── Mall_Customers.csv
├── customer_spending_prediction.py
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd customer-spending-prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Python script

```bash
python customer_spending_prediction.py
```

---

## 📦 Requirements

```text
pandas
scikit-learn
```

---

## 🚀 Future Improvements

Possible improvements for future versions of this project include:

* Testing additional relevant features
* Exploring different regression algorithms
* Comparing multiple models
* Performing additional feature engineering
* Visualizing predictions versus actual values
* Investigating non-linear relationships
* Performing more detailed exploratory data analysis

---

## 📌 Note

This project is primarily a **learning and practice project** focused on understanding the machine learning workflow.

The goal was not to achieve the highest possible prediction score, but to understand how to move from raw data to a trained model and interpret its performance honestly.
