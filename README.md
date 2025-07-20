# GreyNoise Customer Churn Predictor

## Live Application

The interactive churn prediction dashboard is publicly accessible at:

**[https://gn-churn.streamlit.app/](https://gn-churn.streamlit.app/)**

*(Note: The live dashboard uses a model trained on a synthetically generated dataset for demonstration purposes.)*

## Project Overview

This project is a Python-based machine learning application designed to predict customer churn at **GreyNoise Intelligence**. It serves as a modernized, deployable version of an initial analysis I conducted during an internship using R. The primary goal is to provide an interactive tool to assess churn risk in real-time.

To protect proprietary information, the public-facing version of this project, including the live app and the code repository, uses a synthetically generated dataset.

## Features

- **Interactive Prediction Form:** A clean, user-friendly interface to input customer data (Industry, Country, ARR, etc.).
- **Real-Time Churn Score:** Instantly calculates and displays the probability of a customer churning.
- **Risk Assessment:** Provides a clear, color-coded risk level (Low, Moderate, High) based on the prediction.

## Model Performance

The logistic regression model was validated using a **Stratified 5-Fold Cross-Validation** strategy. The metrics below reflect the performance on the original, proprietary dataset.

| Metric          | Score |
| :-------------- | :---- |
| **Mean Accuracy** | 0.79  |
| **Mean ROC AUC** | 0.70  |

An ROC AUC score of 0.70 indicates that the model has a reasonable capability to distinguish between customers who will churn and those who will not.

## Key Predictive Factors

Analysis of the model's coefficients reveals the most significant factors influencing churn predictions. The findings below are presented in qualitative terms to protect insights from the original data.

#### Top Factors Increasing Churn Risk
1.  **Geographic Region:** The model identified that customers from certain geographic regions had a significantly higher propensity to churn.
2.  **Unknown Account History:** When a customer's prior account signup status was unknown, they were flagged as a high churn risk.
3.  **Specific Industry Segments:** Customers within certain specialized industries, such as cybersecurity software, showed a higher tendency to churn.

#### Top Factors Decreasing Churn Risk
1.  **Acquisition Source:** Customers who came to GreyNoise via direct traffic were the least likely to churn, indicating strong brand recognition or intent.
2.  **Existing Account History:** Knowing a customer had a previous free account was a strong signal of loyalty and a significantly lower churn risk.
3.  **Annual Recurring Revenue (ARR):** As a customer's ARR increased, their likelihood of churn decreased substantially, suggesting that higher-paying customers are more invested in the service.

## Tech Stack & Methodology

- **Modeling:** Python 3.11, scikit-learn, pandas, numpy, joblib
- **Web App:** Streamlit
- **Validation:** Stratified K-Fold Cross-Validation
- **Source Code & Data:** The `gn_churn.ipynb` notebook contains the complete code for data processing and modeling. To protect confidential information, the original dataset is not included. The notebook generates a **synthetic dataset** that mimics the structure of the original.

## 📁 File Structure

```
.
├── app.py                  # The main Streamlit application script
├── gn_churn.ipynb          # Jupyter Notebook for model training and evaluation
├── churn_model.joblib      # The exported model (trained on synthetic data)
├── requirements.txt        # Python dependencies for the project
└── README.md               # This file
