
# ============================================================
# BREAST CANCER CLASSIFICATION - ML EVALUATION DASHBOARD
# BITS Pilani - Machine Learning Deployment Assignment
# ============================================================

# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# 2. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer ML Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 3. CUSTOM CSS STYLING
# ------------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

/* Main title */
.main-title {
    font-size: 32px;
    font-weight: 750;
    color: #17365D;
    margin-bottom: 5px;
}

.main-subtitle {
    font-size: 15px;
    color: #64748B;
    margin-bottom: 20px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #DCE4EF;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #EAF0F8;
}

/* Section headings */
h2, h3 {
    color: #17365D !important;
}

/* Buttons */
.stButton button {
    background-color: #17365D;
    color: white;
    border-radius: 8px;
}

/* Information boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# 4. APPLICATION HEADER
# ------------------------------------------------------------

st.markdown("""
<div class="main-title">
    🧬 Breast Cancer Classification Dashboard
</div>

<div class="main-subtitle">
    Machine Learning Model Evaluation and Comparative Analysis
    | Wisconsin Diagnostic Dataset
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# 5. MODEL FILE CONFIGURATION
# ------------------------------------------------------------

MODEL_PATH = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}


# 6. LOAD SAVED MODELS
# ------------------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, file_name in model_files.items():

        file_path = os.path.join(MODEL_PATH, file_name)

        loaded_models[model_name] = joblib.load(file_path)

    return loaded_models


# Check model files before loading

missing_files = [
    file_name
    for file_name in model_files.values()
    if not os.path.exists(os.path.join(MODEL_PATH, file_name))
]

if missing_files:

    st.error(f"Missing model files: {missing_files}")
    st.stop()


try:

    models = load_models()

except Exception as error:

    st.error(f"Unable to load trained models: {error}")
    st.stop()


# Obtain the original feature names from the trained model

expected_features = list(
    models["Logistic Regression"].feature_names_in_
)


# 7. SIDEBAR
# ------------------------------------------------------------

st.sidebar.markdown("## 🧬 ML Dashboard")

st.sidebar.markdown("---")

st.sidebar.markdown("### Project Information")

st.sidebar.write("**Dataset**")
st.sidebar.caption("Breast Cancer Wisconsin Diagnostic")

st.sidebar.write("**Dataset Size**")
st.sidebar.caption("569 observations")

st.sidebar.write("**Input Features**")
st.sidebar.caption("30 numerical features")

st.sidebar.write("**Training Records**")
st.sidebar.caption("455")

st.sidebar.write("**Testing Records**")
st.sidebar.caption("114")

st.sidebar.write("**Classification Models**")
st.sidebar.caption("5 trained algorithms")

st.sidebar.markdown("---")

st.sidebar.markdown("### Target Classification")

st.sidebar.markdown(
    """
    <div style="
        background-color:#DCEFE4;
        padding:10px;
        border-radius:8px;
        color:#166534;">
        <b>0 = Benign</b>
    </div>

    <br>

    <div style="
        background-color:#FCE4E4;
        padding:10px;
        border-radius:8px;
        color:#991B1B;">
        <b>1 = Malignant</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "BITS Pilani | Machine Learning Deployment Assignment"
)

st.sidebar.caption(
    "Academic demonstration only. Not intended for medical diagnosis."
)


# 8. MODEL LOADING STATUS
# ------------------------------------------------------------

st.success("All five previously trained models loaded successfully.")


# 9. DATASET UPLOAD
# ------------------------------------------------------------

st.header("📁 Dataset Upload")

st.write(
    "Upload a CSV file containing the 30 input features "
    "and the target column."
)

uploaded_file = st.file_uploader(
    "Choose your test dataset",
    type=["csv"],
    help="CSV must contain 30 input features and a target column."
)


# 10. PROCESS UPLOADED DATA
# ------------------------------------------------------------

if uploaded_file is not None:

    try:

        test_data = pd.read_csv(uploaded_file)

    except Exception as error:

        st.error(f"Unable to read CSV file: {error}")
        st.stop()


    # Validate target column

    if "target" not in test_data.columns:

        st.error("The uploaded CSV must contain a 'target' column.")
        st.stop()


    # Validate feature columns

    uploaded_features = [
        column for column in test_data.columns
        if column != "target"
    ]

    missing_features = [
        feature for feature in expected_features
        if feature not in uploaded_features
    ]

    extra_features = [
        feature for feature in uploaded_features
        if feature not in expected_features
    ]

    if missing_features or extra_features:

        st.error(
            f"Feature mismatch. Missing: {missing_features}. "
            f"Unexpected: {extra_features}."
        )
        st.stop()


    # Arrange features in the same order as training

    X_test = test_data[expected_features]
    y_test = test_data["target"]


    # Validate empty data and missing values

    if test_data.empty:

        st.error("The uploaded dataset is empty.")
        st.stop()

    if X_test.isnull().any().any() or y_test.isnull().any():

        st.error("The uploaded dataset contains missing values.")
        st.stop()


    # Validate target values

    if not y_test.isin([0, 1]).all():

        st.error("Target column must contain only 0 and 1.")
        st.stop()

    if y_test.nunique() < 2:

        st.error("Test dataset must contain both target classes.")
        st.stop()


    # Validate numeric features

    try:

        X_test = X_test.astype(float)
        y_test = y_test.astype(int)

        if not np.isfinite(X_test.to_numpy()).all():

            st.error("Features contain infinite or invalid values.")
            st.stop()

    except (ValueError, TypeError):

        st.error("All input features must be numerical.")
        st.stop()


    st.success(
        f"Dataset validated successfully: "
        f"{len(test_data)} records and "
        f"{len(expected_features)} features."
    )


    # 11. DATASET OVERVIEW
    # --------------------------------------------------------

    st.markdown("---")

    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(test_data))

    col2.metric("Input Features", len(expected_features))

    col3.metric("Target Classes", y_test.nunique())

    with st.expander("Preview Uploaded Dataset"):

        st.dataframe(
            test_data.head(10),
            use_container_width=True
        )

    with st.expander("Target Class Distribution"):

        class_distribution = (
            y_test.map({
                0: "Benign",
                1: "Malignant"
            }).value_counts()
        )

        st.bar_chart(class_distribution)


    # 12. MODEL SELECTION
    # --------------------------------------------------------

    st.markdown("---")

    st.header("⚙️ Model Evaluation")

    selected_model_name = st.selectbox(
        "Select a classification algorithm",
        list(models.keys())
    )

    selected_model = models[selected_model_name]

    st.info(f"Currently selected model: {selected_model_name}")


    # 13. PREDICTIONS
    # --------------------------------------------------------

    try:

        y_pred = selected_model.predict(X_test)

        y_prob = selected_model.predict_proba(X_test)[:, 1]

    except Exception as error:

        st.error(f"Prediction failed: {error}")
        st.stop()


    # 14. CALCULATE METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)


    # 15. DISPLAY PERFORMANCE METRICS
    # --------------------------------------------------------

    st.markdown("---")

    st.header("📈 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col6.metric(
        "MCC",
        f"{mcc:.4f}"
    )


    # 16. CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown("---")

    st.header("🔍 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    ax.set_title(
        f"Confusion Matrix - {selected_model_name}",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Predicted Class")

    ax.set_ylabel("Actual Class")

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Benign", "Malignant"])

    ax.set_yticklabels(["Benign", "Malignant"])

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="black"
            )

    fig.colorbar(image, ax=ax)

    st.pyplot(fig)

    plt.close(fig)


    # 17. CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.markdown("---")

    st.header("📋 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


    # 18. COMPARISON OF ALL FIVE MODELS
    # --------------------------------------------------------

    st.markdown("---")

    st.header("🏆 Comparative Model Analysis")

    comparison_results = []

    for model_name, model in models.items():

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        comparison_results.append({

            "Model": model_name,

            "Accuracy": accuracy_score(
                y_test, predictions
            ),

            "AUC": roc_auc_score(
                y_test, probabilities
            ),

            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),

            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),

            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),

            "MCC": matthews_corrcoef(
                y_test, predictions
            )
        })


    comparison_df = pd.DataFrame(comparison_results)

    st.subheader("Model Performance Comparison Table")

    st.dataframe(
        comparison_df.set_index("Model").round(4),
        use_container_width=True
    )


    # 19. VISUAL MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Performance Metrics Comparison")

    chart_data = comparison_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"]
    ]

    st.bar_chart(
        chart_data,
        y_label="Metric Score",
        x_label="Classification Model"
    )


    # AUC comparison

    st.subheader("AUC Comparison")

    auc_data = comparison_df.set_index("Model")[["AUC"]]

    st.bar_chart(
        auc_data,
        y_label="AUC Score",
        x_label="Classification Model"
    )


    # 20. IDENTIFY BEST MODEL
    # --------------------------------------------------------

    best_model_row = comparison_df.loc[
        comparison_df["MCC"].idxmax()
    ]

    st.markdown("---")

    st.header("⭐ Project Insights")

    st.success(
        f"Best overall model based on MCC: "
        f"{best_model_row['Model']}"
    )

    st.write(
        f"Accuracy: {best_model_row['Accuracy']:.4f}"
    )

    st.write(
        f"Recall: {best_model_row['Recall']:.4f}"
    )

    st.write(
        f"F1 Score: {best_model_row['F1 Score']:.4f}"
    )

    st.write(
        f"MCC: {best_model_row['MCC']:.4f}"
    )

    st.markdown("""
    **Observations:**

    - Five classification algorithms are evaluated using the same test dataset.
    - SVM demonstrated the strongest overall performance in the original evaluation.
    - Logistic Regression achieved the highest AUC in the original evaluation.
    - Gaussian Naive Bayes achieved the highest precision in the original evaluation.
    - Model evaluation is performed using previously saved trained models.
    - No model retraining takes place during application execution.
    """)


    # 21. DOWNLOAD COMPARISON RESULTS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📥 Download Evaluation Results")

    csv_data = comparison_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Model Comparison CSV",
        data=csv_data,
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )


else:

    st.markdown("---")

    st.info(
        "Please upload your test_data.csv file to start "
        "the model evaluation."
    )

    st.markdown("""
    ### About This Application

    This dashboard evaluates five machine learning algorithms
    trained on the Breast Cancer Wisconsin Diagnostic dataset.

    **Algorithms included:**

    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors
    4. Gaussian Naive Bayes
    5. Support Vector Machine

    **Evaluation metrics:**

    Accuracy, AUC, Precision, Recall, F1 Score and MCC.
    """)


# 22. FOOTER
# ------------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748B; font-size:13px;">
        Breast Cancer Classification Dashboard |
        BITS Pilani Machine Learning Assignment
        <br>
        Built using Python, Scikit-learn, Pandas and Streamlit
        <br>
        Academic demonstration only - Not intended for medical diagnosis
    </div>
    """,
    unsafe_allow_html=True
)
