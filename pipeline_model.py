# === Evaluate Status Prediction Model ===
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os


#Load dataset
df = pd.read_csv("C:\project\Liver_cirrhosis_Predictor_ML\data_set\liver_cirrhosis.csv")

# Remove unhelpful columns
df = df.drop(columns=["N_Days", "Status"])

# Separate features and target
X = df.drop(columns=["Stage"])
y = df["Stage"]

# Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

#  Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),  # scale numeric features
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)  # encode categorical
    ]
)

#  Build pipeline (preprocessing + Random Forest)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

#  Train the model
pipeline.fit(X_train, y_train)
print("Model training completed!")

# Evaluate performance
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Save the full pipeline
joblib.dump(pipeline, "C:\project\Liver_cirrhosis_Predictor_ML\model1\liver_pipeline.pkl")
print("\n💾 Pipeline saved successfully as 'liver_pipeline.pkl'")


# Load the trained pipeline
pipeline = joblib.load("C:\project\Liver_cirrhosis_Predictor_ML\model1\liver_pipeline.pkl")

# Extract Random Forest model from pipeline
rf_model = pipeline.named_steps["classifier"]

# Get the preprocessor (used to find all transformed feature names)
preprocessor = pipeline.named_steps["preprocessor"]

# --- Get feature names after encoding ---
# Numeric feature names
num_features = preprocessor.transformers_[0][2]

# Categorical feature names (expanded after OneHotEncoding)
cat_encoder = preprocessor.transformers_[1][1]
cat_features = preprocessor.transformers_[1][2]
encoded_cat_features = cat_encoder.get_feature_names_out(cat_features)

# Combine all feature names
all_features = np.concatenate([num_features, encoded_cat_features])

# --- Get feature importances ---
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# --- Display top 15 important features ---
print("\nTop 15 Most Important Features:\n")
print(feature_importance_df.head(15))

# --- Plot the top features ---
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"].head(15)[::-1],
         feature_importance_df["Importance"].head(15)[::-1])
plt.title("Top 15 Important Features in Liver Cirrhosis Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


