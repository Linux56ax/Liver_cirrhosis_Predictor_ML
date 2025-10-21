import pandas as pd

# Load dataset
df = pd.read_csv("C:\project\Liver_cirrhosis_Predictor_ML\data_set\liver_cirrhosis.csv")

# Drop unhelpful columns
df = df.drop(columns=["N_Days", "Status"])

# Check balance of target variable
print("Stage distribution:\n", df["Stage"].value_counts())
print("\nStage distribution (in %):\n", df["Stage"].value_counts(normalize=True) * 100)
