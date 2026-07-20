# Road Accident Severity Prediction

## Overview
End-to-end Machine Learning project to predict road accident severity (Fatal, Serious, Slight) using UK government data. Built to demonstrate a complete ML pipeline from EDA to Streamlit deployment.

## Dataset
- Source: [UK Road Accident Data — Kaggle](https://www.kaggle.com/datasets/devansodariya/road-accident-united-kingdom-uk-dataset)
- Total Rows: 1,504,150 | Sample Used: 100,000
- Target Variable: Accident_Severity (1=Fatal, 2=Serious, 3=Slight)

## Project Structure
```
road_accident_project/
├── data/
├── notebooks/
├── models/
├── visualizations/
├── app/
└── requirements.txt
```

## Pipeline
1. Problem Definition
2. Data Collection
3. EDA & Visualization
4. Data Preprocessing
5. Feature Engineering
6. Model Training
7. Model Evaluation
8. Hyperparameter Tuning
9. Deployment

## Model Comparison
| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| Logistic Regression | 50% | 0.60 |
| Decision Tree | 75% | 0.75 |
| Random Forest | 84% | 0.79 |

Random Forest selected as best model.

## Key Findings
- Dataset severely imbalanced — 85% Slight, 13.6% Serious, 1.27% Fatal
- Peak accident hours — 8-9 AM and 4-6 PM (rush hours)
- 68.7% accidents on dry roads — high volume roads, not weather driven
- Fatal accidents correlate with higher speed limits

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn, Imbalanced-learn
- Matplotlib, Seaborn
- Streamlit, Joblib

## How to Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/vaibhavanalytix/Road-Accident-Severity-Prediction.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in data/ folder
# https://www.kaggle.com/datasets/devansodariya/road-accident-united-kingdom-uk-dataset

# 4. Run notebook to generate model files
# Open notebooks/eda.ipynb and Run All cells

# 5. Run the app
streamlit run app/app.py
```
> Dataset and model files excluded due to GitHub size limits.

## Limitations
- Model predicts mostly "Slight" - baseline accuracy of always predicting Slight = 85%
- Model's 84% accuracy is not meaningfully better than this baseline for Fatal/Serious cases
- Fixes attempted:
  - class_weight='balanced' - minimal improvement
  - SMOTE - synthetic samples insufficient, only 1,016 real Fatal records to learn from
- Root cause: Fatal accidents are rare by nature - data limitation, not model limitation