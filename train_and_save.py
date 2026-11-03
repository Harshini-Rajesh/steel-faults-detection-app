import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('test_data.csv')

# Define feature columns and target columns
feature_cols = df.columns[:27].tolist()
target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

X = df[feature_cols]
# Convert one-hot encoded target columns into a single multi-class label (0 to 6)
y = df[target_cols].idxmax(axis=1).astype('category').cat.codes

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Critical for SVM, KNN, and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('model', exist_ok=True) # Create 'model' directory if it doesn't exist

# Save the fitted scaler
joblib.dump(scaler, 'model/scaler.pkl')
print("Saved: model/scaler.pkl")

# Initialize and Train the 5 Required Models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "support_vector_machine": SVC(probability=True, random_state=42)
}

for name, model in models.items():
    # Use scaled data for models sensitive to scale, unscaled for tree/NB 
    if name in ["logistic_regression", "knn", "support_vector_machine"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
        
    # Save as a .pkl file
    file_path = f"model/{name}.pkl"
    joblib.dump(model, file_path)
    print(f"Saved: {file_path}")

print("\nAll model .pkl files generated successfully inside the 'model/' directory")