# === Liver Cirrhosis ML Pipeline ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#Load dataset
df = pd.read_csv("liver_cirrhosis.csv")

#Remove unhelpful columns
df = df.drop(columns=["N_Days", "Status"])

#Separate features and target
X = df.drop(columns=["Stage"])
y = df["Stage"]

#Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),  # scale numeric features
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)  # encode categorical
    ]
)

#Build pipeline (preprocessing + Random Forest)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

#Train the model
pipeline.fit(X_train, y_train)
print("Model training completed!")

#Evaluate performance
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Save the full pipeline
joblib.dump(pipeline, "liver_pipeline.pkl")
print("\nPipeline saved successfully as 'liver_pipeline.pkl'")
