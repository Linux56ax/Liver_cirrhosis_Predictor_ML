
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

#Load your training dataset
train_path = r"C:\project\Liver_cirrhosis_Predictor_ML\data_set\liver_train.csv"
train_data = pd.read_csv(train_path)

#Separate features (X) and target (y)
X_train = train_data.drop(columns=["Stage"])
y_train = train_data["Stage"]

#Encode categorical columns
label_encoders = {}
for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    label_encoders[col] = le

#Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Model training completed!")

#Save trained model and encoders
joblib.dump(rf, r"C:\project\Liver_cirrhosis_Predictor_ML\model\liver_random_forest.pkl")
joblib.dump(label_encoders, r"C:\project\Liver_cirrhosis_Predictor_ML\model\label_encoders.pkl")
print("Model and encoders saved successfully!")
