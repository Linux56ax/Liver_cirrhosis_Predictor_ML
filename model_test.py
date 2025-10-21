
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

#File paths
test_path = r"C:\project\Liver_cirrhosis_Predictor_ML\data_set\liver_test.csv"
model_path = r"C:\project\Liver_cirrhosis_Predictor_ML\model\liver_random_forest.pkl"
encoder_path = r"C:\project\Liver_cirrhosis_Predictor_ML\model\label_encoders.pkl"

#Load test dataset
test_data = pd.read_csv(test_path)

#Split into features (X) and target (y)
X_test = test_data.drop(columns=["Stage"])
y_test = test_data["Stage"]

#Load saved model and encoders
rf = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)

#Apply the same encodings as during training
for col in X_test.select_dtypes(include="object").columns:
    le = label_encoders[col]
    X_test[col] = le.transform(X_test[col])

#Make predictions
y_pred = rf.predict(X_test)

#Evaluate model performance
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
