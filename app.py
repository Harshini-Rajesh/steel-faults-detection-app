import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# Page Configuration
st.set_page_config(
    page_title="Steel Faults Classification Dashboard",
    layout="wide"
)

##### TO DO: revisit the layout and customization here
# Custom Styling 
st.markdown("""
    <style>
    .main-title {font-size: 30px; font-weight: 700; color: #1f77b4; margin-bottom: 0px;}
    .sub-title {font-size: 16px; color: #4f4f4f; margin-top: 5px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Industrial Steel Plates Fault Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning Demo App</p>', unsafe_allow_html=True)
st.markdown("---")

# Load Cached Artifacts (Models & Scaler)
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load('model/scaler.pkl')
        models = {
            "Logistic Regression": joblib.load('model/logistic_regression.pkl'),
            "Decision Tree": joblib.load('model/decision_tree.pkl'),
            "KNN": joblib.load('model/knn.pkl'),
            "Naive Bayes": joblib.load('model/naive_bayes.pkl'),
            "Support Vector Machine": joblib.load('model/support_vector_machine.pkl')
        }
        return scaler, models, None
    except Exception as e:
        return None, None, str(e)

scaler, models, err = load_artifacts()

if err:
    st.error(f"Error loading model artifacts: {err}. Ensure the 'model/' folder with all .pkl files exists in your repository.")
    st.stop()

# Sidebar Control Panel
st.sidebar.header("Control Panel")
model_choice = st.sidebar.selectbox("Select Classification Model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Management")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

# Data Loading Logic
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom test CSV loaded successfully!")
else:
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
    else:
        st.error("'test_data.csv' not found in the project root directory. Please upload it via the sidebar or place it in your folder.")
        st.stop()

# Separate Features and Target Columns
feature_cols = test_df.columns[:27].tolist()
target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

X_test = test_df[feature_cols]
y_true_onehot = test_df[target_cols]
y_test = y_true_onehot.idxmax(axis=1).astype('category').cat.codes
class_names = y_true_onehot.columns.tolist()

# Preprocessing & Inference
selected_model = models[model_choice]

if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
    X_test_processed = scaler.transform(X_test)
else:
    X_test_processed = X_test

preds = selected_model.predict(X_test_processed)
try:
    probs = selected_model.predict_proba(X_test_processed)
except Exception:
    probs = np.zeros((len(preds), len(class_names)))

# Compute Metrics for Selected Model
acc = accuracy_score(y_test, preds)
try:
    auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
except Exception:
    auc = 0.0
prec = precision_score(y_test, preds, average='macro', zero_division=0)
rec = recall_score(y_test, preds, average='macro', zero_division=0)
f1 = f1_score(y_test, preds, average='macro', zero_division=0)
mcc = matthews_corrcoef(y_test, preds)

# Display Evaluation Metrics Dashboard
st.subheader(f"Performance Evaluation: {model_choice}")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

st.markdown("---")

# Visualizations and Data Preview Section
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Actual Class", fontsize=10)
    st.pyplot(fig)

with col_right:
    st.subheader("🔍 Sample Predictions Preview")
    preview_df = X_test.head(5).copy()
    preview_df['Actual Fault'] = y_true_onehot.idxmax(axis=1).head(5).values
    preview_df['Predicted Fault'] = pd.Series(preds).map(dict(enumerate(class_names))).head(5).values
    st.dataframe(preview_df[['Actual Fault', 'Predicted Fault']], use_container_width=True)

# Overall Comparison Summary Table (Calculated Dynamically)
st.markdown("---")
st.subheader(" Overall Model Comparison Benchmark")

@st.cache_data
def compute_all_models_metrics():
    summary_results = []
    for name, mdl in models.items():
        if name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_proc = scaler.transform(X_test)
        else:
            X_proc = X_test
            
        p = mdl.predict(X_proc)
        try:
            prob = mdl.predict_proba(X_proc)
        except Exception:
            prob = np.zeros((len(p), len(class_names)))
            
        ac = accuracy_score(y_test, p)
        try:
            au = roc_auc_score(y_test, prob, multi_class='ovr', average='macro')
        except Exception:
            au = 0.0
        pr = precision_score(y_test, p, average='macro', zero_division=0)
        rc = recall_score(y_test, p, average='macro', zero_division=0)
        f1_val = f1_score(y_test, p, average='macro', zero_division=0)
        mc = matthews_corrcoef(y_test, p)
        
        summary_results.append({
            "ML Model Name": name,
            "Accuracy": round(ac, 4),
            "AUC": round(au, 4),
            "Precision": round(pr, 4),
            "Recall": round(rc, 4),
            "F1": round(f1_val, 4),
            "MCC": round(mc, 4)
        })
    return pd.DataFrame(summary_results)

comparison_df = compute_all_models_metrics()
st.table(comparison_df)


######## TO DO: add observation section here
