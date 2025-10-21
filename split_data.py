from sklearn.model_selection import train_test_split
import pandas as pd
#loading the data set 
df = pd.read_csv("C:\project\Liver_cirrhosis_Predictor_ML\data_set\liver_cirrhosis.csv")
# Separate features (X) and target (y)
X = df.drop(columns=["Status"])

y = df["Status"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Save split datasets to CSV (optional)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
#train_data.to_csv("liver_train.csv", index=False)
#test_data.to_csv("liver_test.csv", index=False)
print("Train and test sets saved successfully!")
