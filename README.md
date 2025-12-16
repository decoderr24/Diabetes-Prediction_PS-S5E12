# 🏥 Diabetes Prediction - Kaggle Playground Series 5 Episode 12

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This repository contains my solution for the **Kaggle Playground Series 5, Episode 12** competition. The goal is to predict whether a patient has diabetes based on various health indicators (features) such as BMI, Age, General Health, and Cholesterol levels.

The solution utilizes an **Ensemble Learning** approach, stacking three powerful Gradient Boosting Decision Tree (GBDT) models and optimizing their weights using **Hill Climbing (Nelder-Mead)** to maximize the ROC-AUC score.

**Competition Link:** [Playground Series S5E12 - Multi-Class Prediction of Obesity Risk](https://www.kaggle.com/competitions/playground-series-s5e12)

## 📊 Dataset
The dataset is synthetically generated from a deep learning model trained on the **CDC Diabetes Health Indicators Dataset**. It mimics real-world healthcare data but contains subtle distribution shifts.

* **Target:** `diagnosed_diabetes` (Binary Classification)
* **Evaluation Metric:** ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

## 🛠️ Methodology & Approach

### 1. Data Preprocessing & Cleaning
* **Log Transformation:** Applied `np.log1p` to skewed features like `BMI`, `Physical_Health`, and `Mental_Health` to normalize distributions.
* **Categorical Encoding:** Label Encoding was used for categorical variables to make them compatible with GBDT models.

### 2. Feature Engineering (Medical Domain Knowledge)
To improve model sensitivity, I engineered several features based on medical indices:
* **Blood Pressure Metrics:**
    * *Pulse Pressure:* `Systolic - Diastolic`
    * *MAP (Mean Arterial Pressure):* Weighted average of systolic and diastolic pressure.
* **Cholesterol Ratios:**
    * *Cholesterol Ratio:* `Total Cholesterol / HDL`
    * *Non-HDL:* `Total Cholesterol - HDL`
* **Interaction Features:**
    * `Health_Risk_Index`: Interaction between General Health perception and BMI.
    * `BMI_Categorical`: Binning BMI into standard medical categories (Underweight, Normal, Overweight, Obese).

### 3. Modeling Strategy
I employed a **Stratified K-Fold Cross-Validation (K=5)** to ensure the model generalizes well across different data subsets.
* **LightGBM:** High efficiency on large datasets, tuned for gradient-based one-side sampling.
* **XGBoost:** robust baseline with optimized regularization (`reg_alpha`, `reg_lambda`).
* **CatBoost:** Excellent handling of categorical features without extensive preprocessing.

### 4. Ensemble Optimization (Hill Climbing)
Instead of a simple weighted average, I used the **Nelder-Mead** optimization algorithm (via `scipy.optimize`) to mathematically find the optimal blending weights for the three models based on OOF (Out-of-Fold) predictions.

$$FinalPred = w_1 \cdot P_{LGBM} + w_2 \cdot P_{XGB} + w_3 \cdot P_{CatBoost}$$

## 📂 Project Structure

| File Name | Description |
| :--- | :--- |
| `ps-s5e12-diabetes-prediction.ipynb` | **Main Notebook**. Contains the full pipeline: EDA, advanced feature engineering, and final model training. |
| `diabetic-prediction-ensemble.ipynb` | **Ensemble Logic**. Focuses on the Hill Climbing optimization script and blending strategy implementation. |
| `ps-s5e12-base-model.ipynb` | **Baseline**. Initial exploration, simple modeling, and benchmark score establishment. |

## 🚀 Results

| Model | CV Score (AUC) | Public LB |
| :--- | :--- | :--- |
| Baseline (Single XGB) | 0.703+ | 0.702xxx |
| **Ensemble (Hill Climbing)** | **0.706** | **0.703+** |

## Model Comparasion 
<img width="855" height="509" alt="image" src="https://github.com/user-attachments/assets/e49ffc46-09ec-4454-b15b-3a0d0e8e76cf" />


<img width="1079" height="253" alt="bokeh_plot (3)" src="https://github.com/user-attachments/assets/a91b810a-bb76-4f44-a942-6d50bb1b784e" />

<img width="1079" height="349" alt="bokeh_plot (4)" src="https://github.com/user-attachments/assets/c3fa6260-d17a-4e3f-9e59-7dec8a9b1c3d" />


*> Note: Scores are approximate based on the latest run.*


