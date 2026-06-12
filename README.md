# IELTS Performance Prediction System

Predicts a student's IELTS band score using assessment records and behavioural learning features.

## Results

| Model | RMSE | MAE | R2 | CV RMSE (5-fold) |
|---|---|---|---|---|
| **Linear Regression** | **0.1582** | **0.1281** | **0.9813** | **0.1526** |
| XGBoost | 0.1636 | 0.1337 | 0.9800 | 0.1629 |
| Random Forest | 0.1689 | 0.1369 | 0.9786 | 0.1689 |

> Best model: Linear Regression (RMSE 0.158, R2 0.981 on held-out test set)

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit

## Dataset

Synthetic dataset of **5,000 student records** with the following features:

| Feature | Description |
|---|---|
| Reading_Score | IELTS Reading component score (1-9) |
| Writing_Score | IELTS Writing component score (1-9) |
| Listening_Score | IELTS Listening component score (1-9) |
| Speaking_Score | IELTS Speaking component score (1-9) |
| Attendance_Percentage | Class attendance (40-100%) |
| Practice_Hours | Weekly self-study hours |
| Vocabulary_Score | Vocabulary assessment score (20-100) |
| Grammar_Score | Grammar assessment score (20-100) |
| Mock_Test_Score | Full mock test band score (1-9) |
| **Final_IELTS_Band** | **Target - actual IELTS band achieved** |

## Project Structure

```
IELTS-Performance-Prediction/
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Cleaned / transformed data
├── notebooks/
│   ├── eda.ipynb          # Exploratory Data Analysis
│   └── model_training.ipynb  # Training + evaluation
├── src/
│   ├── generate_dataset.py   # Synthetic data generator
│   ├── preprocessing.py      # Load, split, scale
│   ├── feature_engineering.py # Composite features
│   ├── train.py              # Model definitions + CV
│   └── evaluate.py           # Metrics + comparison table
├── models/                # Saved model files (.pkl)
├── results/               # Plots and CSV outputs
├── app.py                 # Streamlit prediction app
└── requirements.txt
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python src/generate_dataset.py

# 4. Open notebooks (run in order)
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/model_training.ipynb

# 5. Run the Streamlit app (after training)
streamlit run app.py
```

## Key Findings

- **Linear Regression** achieved the best performance: RMSE 0.158, R2 0.981
- **Reading Score**, **Listening Score**, and **Mock Test Score** are the strongest predictors
- **Attendance, Practice Hours, and Grammar** contribute meaningfully to the final band score
- 5-fold cross-validation confirmed strong generalisation (CV RMSE 0.153) with low variance across folds