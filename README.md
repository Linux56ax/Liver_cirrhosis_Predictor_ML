Liver Cirrhosis Stage Predictor

Predicts the stage of liver cirrhosis based on patient clinical features using a Random Forest classifier.

Features

Predicts 4 stages: Normal, Fatty Liver, Liver Fibrosis, Liver Cirrhosis

Uses clinical features like Age, Sex, Bilirubin, SGOT, Albumin, Protime, etc.

Handles categorical encoding and missing data

Repo Structure
data_set/        # Train/test CSVs
model/           # Saved model & encoders
randomforest.py  # Train Random Forest
model_test.py    # Test predictions
split_data.py    # Train/test split
balance_data.py  # Handle class imbalance
pipeline_model.py# Preprocessing & pipeline

Requirements
pip install pandas numpy scikit-learn joblib matplotlib

Usage

Train model:

python randomforest.py


Test predictions:

python model_test.py

Evaluation

Accuracy, Precision, Recall, F1-score

Confusion matrix visualization
