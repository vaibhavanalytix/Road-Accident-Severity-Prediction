# Road Accident Severity Prediction

## Overview
A Machine Learning project to predict the severity of road accidents (Fatal, Serious, Slight) based on time, weather, road type, and light conditions.

## Dataset
- Source: Kaggle — UK Road Accident Data
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

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest (Best Model — 84% Accuracy)

## Key Findings
- Dataset is highly imbalanced — 85% Slight, 13.6% Serious, 1.27% Fatal
- Peak accident hours — 8-9 AM and 4-6 PM
- 68.7% accidents occur on dry roads
- SMOTE and class_weight both tried to handle imbalance

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

# 3. Run notebook to generate model files
# Open notebooks/eda.ipynb and run all cells

# 4. Run the app
streamlit run app/app.py
```
> Note: Dataset and model files not included due to size limits.
> Download dataset from Kaggle: UK Road Accident Data

## Limitations
- Model biased toward Slight due to class imbalance
- Fatal cases only 1.27% - insufficient data to learn patterns