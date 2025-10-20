# 🚀 AI Engineering Portfolio: Telco Customer Churn Prediction

## Executive Summary

This project demonstrates an end-to-end Machine Learning Engineering workflow for predicting customer churn at a fictional telecommunications company. The goal was to build a robust, interpretable model to identify high-risk customers, allowing the business to target retention campaigns effectively and minimize lost revenue.

---

## 1. Methodology & Engineering Highlights

This solution was built following industry best practices to ensure the model is reliable and deployment-ready.

### A. Core Workflow & Leakage Prevention
The entire workflow, from feature engineering to modeling, was encapsulated in a **`scikit-learn` Pipeline** with a **`ColumnTransformer`**.

* **Data Leakage Mitigation:** Critical features that cause data leakage (e.g., `Churn Score`, `Churn Reason`) were **explicitly dropped** before the $\text{Train/Test}$ split.
* **Robust Preprocessing:** The Pipeline safely handled:
    * **Imputation** of missing data (`Total Charges`).
    * **Standard Scaling** of numerical features.
    * **One-Hot Encoding** of categorical features.

### B. Model & Tuning
* **Model:** **Regularized Logistic Regression** was chosen for its high **interpretability** via feature coefficients.
* **Tuning:** **`GridSearchCV`** was used to find the optimal regularization strength ($\text{C}$) to balance bias and variance. The optimal hyperparameter found was **$\text{C} = 10000.0$**, indicating the model performed best with minimal regularization.

---

## 2. Key Results and Model Performance

The final model was evaluated on the completely unseen $20\%$ test set.

### A. Performance Metrics
The primary evaluation metric was **$\text{ROC AUC}$** (Area Under the Curve), as it accurately measures the model's ability to rank customers by churn risk, which is crucial for retention budgeting.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **ROC AUC Score** | $\mathbf{0.8474}$ | A strong result; the model is highly effective at distinguishing between churners and non-churners. |
| **Accuracy** | $80\%$ | Overall classification correctness. |

### B. ROC Curve Visualization
The curve remains high and far from the random guessing baseline ($\text{AUC}=0.5$), confirming the model's predictive power across all classification thresholds.


![Receiver Operating Characteristic (ROC) Curve](roc_curve_plot.png)



---

## 3. Actionable Business Insights (Coefficient Analysis)

The standardized coefficients of the Logistic Regression model directly translate into business strategy by showing the impact of each feature on churn probability.

### Top Drivers of Churn Probability
![Feature Coefficients for Churn Prediction](bar-chart.png)

The analysis reveals the two most critical levers for retention:

| Feature | Impact | Coefficient ($\beta$) | Actionable Insight |
| :--- | :--- | :--- | :--- |
| **Contract\_Month-to-month** | **Strong Positive** (Risk) | (Value from your output) | High-risk factor. Convert these customers to long-term contracts immediately. |
| **Tenure Months** | **Strong Negative** (Retention) | (Value from your output) | The longer a customer stays, the less likely they are to churn. Focus retention efforts early in the customer lifecycle. |



---

## 4. Setup and Dependencies

To replicate this project, clone the repository and install the necessary libraries:

```bash
# Clone the repository
git clone https://github.com/Gbenga95/telco-churn-prediction-ml

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
