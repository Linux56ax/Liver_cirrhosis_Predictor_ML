
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.pyplot as plt

#File paths
test_path = r"C:\project\Liver_cirrhosis_Predictor_ML\data_set\liver_test.csv"
model_path = r"C:\project\Liver_cirrhosis_Predictor_ML\model\liver_random_forest.pkl"
encoder_path = r"C:\project\Liver_cirrhosis_Predictor_ML\model\label_encoders.pkl"
output_path = r"C:\project\Liver_cirrhosis_Predictor_ML\results\evaluation_results.csv"

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


y_test_codes = y_test.astype('category').cat.codes
y_pred_codes = pd.Series(y_pred).astype('category').cat.codes

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(y_test_codes, y_pred_codes, color='blue', alpha=0.7)
plt.plot([y_test_codes.min(), y_test_codes.max()],
         [y_test_codes.min(), y_test_codes.max()], 'r--', label='Perfect Prediction')
plt.xlabel('y_test (Actual Stage)')
plt.ylabel('y_pred (Predicted Stage)')
plt.title('Actual vs Predicted Stages')
plt.legend()
plt.grid(True)
plt.show()


# Compute metrics
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)



#plotting the result 
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Liver Cirrhosis Stage Prediction")
plt.colorbar()
plt.xlabel("Predicted Stage")
plt.ylabel("Actual Stage")
plt.xticks([0, 1, 2], [1, 2, 3])
plt.yticks([0, 1, 2], [1, 2, 3])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.show()

# --- Plot Accuracy per Stage ---
stage_labels = list(report.keys())[:-3]  # Exclude avg rows
stage_accuracy = [report[label]["f1-score"] for label in stage_labels]

plt.figure(figsize=(6, 4))
plt.bar(stage_labels, stage_accuracy)
plt.title("F1-Score per Stage")
plt.xlabel("Stage")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
for i, v in enumerate(stage_accuracy):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()