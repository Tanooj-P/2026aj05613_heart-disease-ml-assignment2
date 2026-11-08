"""
Heart Disease Classification - Streamlit Web Application
BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2

This app loads pre-trained classification models and evaluates them
on uploaded test data without retraining.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Heart Disease ML Classifier",
    page_icon="🫀",
    layout="wide"
)

# ===================== CONSTANTS =====================
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

TARGET_COL = "target"


# ===================== LOAD MODELS =====================
@st.cache_resource
def load_models():
    """Load all saved models and the scaler."""
    loaded_models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                loaded_models[name] = pickle.load(f)

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    return loaded_models, scaler


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all 6 evaluation metrics."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Disease (0)', 'Disease (1)'],
        yticklabels=['No Disease (0)', 'Disease (1)'],
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob, model_name):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ===================== MAIN APP =====================
def main():
    st.title("🫀 Heart Disease Classification Dashboard")
    st.markdown("""
    **BITS Pilani M.Tech (AIML) — Machine Learning Assignment 2**

    This application evaluates 5 pre-trained classification models on uploaded test data.
    Upload a CSV with the required features and target column to see results.
    """)

    st.divider()

    # Load models
    models, scaler = load_models()

    if not models:
        st.error("No trained models found in the model/ directory. Please run train_models.py first.")
        return

    if scaler is None:
        st.error("Scaler not found. Please run train_models.py first.")
        return

    # ===================== SIDEBAR =====================
    st.sidebar.header("⚙️ Configuration")

    # Model selection dropdown
    selected_model_name = st.sidebar.selectbox(
        "Select a Classification Model",
        list(models.keys()),
        index=0
    )

    show_all_models = st.sidebar.checkbox("Compare All Models", value=True)

    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.markdown("""
    **Dataset:** Heart Disease (UCI)
    **Features:** 13
    **Instances:** 920
    **Task:** Binary Classification
    **Target:** Presence of heart disease (0/1)
    """)

    st.sidebar.divider()
    st.sidebar.markdown("### 📋 Feature Description")
    feature_desc = {
        "age": "Age in years",
        "sex": "Sex (1=male, 0=female)",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar > 120 mg/dl",
        "restecg": "Resting ECG results (0-2)",
        "thalach": "Max heart rate achieved",
        "exang": "Exercise induced angina",
        "oldpeak": "ST depression by exercise",
        "slope": "Slope of peak exercise ST",
        "ca": "Number of major vessels (0-3)",
        "thal": "Thalassemia (1=normal, 2=fixed, 3=reversible)"
    }
    for feat, desc in feature_desc.items():
        st.sidebar.markdown(f"- **{feat}**: {desc}")

    # ===================== FILE UPLOAD =====================
    st.header("📁 Upload Test Data (CSV)")

    uploaded_file = st.file_uploader(
        "Upload a CSV file with all 13 features and 'target' column",
        type=["csv"],
        help="The CSV must contain columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target"
    )

    # Default test data option
    use_default = st.checkbox("Use default test data (test_data.csv)", value=True)

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        use_default = False
        st.success(f"Uploaded file: {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
    elif use_default:
        default_path = "test_data.csv"
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
            st.info(f"Using default test data: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            st.warning("Default test_data.csv not found. Please upload a CSV file.")

    if df is None:
        st.warning("Please upload a CSV file or select default test data to proceed.")
        return

    # Validate columns
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        st.error(f"Missing required features: {missing_features}")
        return

    if TARGET_COL not in df.columns:
        st.error(f"Missing target column: '{TARGET_COL}'")
        return

    # Show data preview
    with st.expander("📊 Preview Uploaded Data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        with col2:
            st.markdown(f"**Target distribution:** {dict(df[TARGET_COL].value_counts())}")

    # Prepare data
    X_test = df[FEATURE_NAMES]
    y_test = df[TARGET_COL]
    X_test_scaled = scaler.transform(X_test)

    st.divider()

    # ===================== ALL MODELS COMPARISON =====================
    if show_all_models:
        st.header("📊 All Models — Comparison Table")

        all_results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            all_results[name] = compute_metrics(y_test, y_pred, y_prob)

        results_df = pd.DataFrame(all_results).T
        results_df.index.name = "ML Model Name"

        st.dataframe(
            results_df.style.highlight_max(axis=0, color='lightgreen')
                            .highlight_min(axis=0, color='lightyellow')
                            .format("{:.4f}"),
            use_container_width=True
        )

        # Best model
        best_model = results_df['F1'].idxmax()
        st.success(f"🏆 **Best Model (by F1 Score):** {best_model} with F1 = {results_df.loc[best_model, 'F1']:.4f}")

        # Bar chart comparison
        st.subheader("📈 Metrics Comparison Chart")
        fig, ax = plt.subplots(figsize=(12, 5))
        results_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticklabels(results_df.index, rotation=30, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

    # ===================== SINGLE MODEL DETAILS =====================
    st.header(f"🔍 Model Details: {selected_model_name}")

    model = models[selected_model_name]
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Metrics cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC", f"{metrics['AUC']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")
    col4.metric("Recall", f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics['F1']:.4f}")
    col6.metric("MCC", f"{metrics['MCC']:.4f}")

    # Confusion Matrix and ROC Curve side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_test, y_pred, selected_model_name)
        st.pyplot(fig_cm)
        plt.close()

    with col_right:
        st.subheader("ROC Curve")
        fig_roc = plot_roc_curve(y_test, y_prob, selected_model_name)
        st.pyplot(fig_roc)
        plt.close()

    # Classification Report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    st.divider()

    # ===================== FOOTER =====================
    st.markdown("""
    ---
    **Heart Disease Classification Dashboard** | BITS Pilani M.Tech (AIML) | Machine Learning Assignment 2
    *Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM*
    *Dataset: Heart Disease (UCI) — 920 instances, 13 features*
    """)


if __name__ == "__main__":
    main()
