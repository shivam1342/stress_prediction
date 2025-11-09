"""
Streamlit App for Stress Level Prediction
Optimized version: cached data loading, fixed UI flow, evaluation working
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc



# Hide TensorFlow & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Streamlit cache settings
os.environ["STREAMLIT_CACHE_DIR"] = "./.streamlit_cache"
# os.environ["STREAMLIT_PERSIST"] = "true"

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import WESADDataProcessor
from models import StressPredictionModels

# =================== PAGE CONFIG ===================
st.set_page_config(
    page_title="Stress Level Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== STYLING ===================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #42a5f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #42a5f5);
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565c0, #1e88e5);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# =================== CACHED DATA LOADER ===================
# @st.cache_data(show_spinner=True, persist="disk")
@st.cache_data(show_spinner=True)
def load_wesad_data():
    processor = WESADDataProcessor()
    possible_paths = ['data/S3']
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not data_path:
        st.warning("⚠️ WESAD dataset not found — using synthetic data.")
        return processor.create_sample_data(n_samples=1000)

    st.info(f"Loading dataset from: {data_path}")
    X, y = processor.process_subject_data(data_path)
    if X is None or y is None or len(X) == 0:
        st.warning("❌ Failed to extract features — using synthetic data instead.")
        return processor.create_sample_data(n_samples=1000)

    # Clean invalid labels (0,6,7 etc.)
    y = np.array(y)
    y = np.where(y == 2, 1, 0)  # Map to binary stress vs non-stress
    return X, y


# =================== MODEL TRAINING ===================
# @st.cache_resource(show_spinner=True, persist="disk")
@st.cache_resource(show_spinner=True)
def train_all_models(X, y):
    models = StressPredictionModels()
    models.train_models(X, y)
    evaluations = models.evaluate_models()
    return models, evaluations
    



# =================== MAIN PAGE ===================
def main_page():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white;">🧠 Stress Level Prediction System</h1>
        <p style="color: white;">Predict stress levels from physiological signals using ML & DL</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        # Data loading
        st.subheader("📁 Data Management")
        if st.button("Load Data", use_container_width=True):
            with st.spinner("Loading WESAD data..."):
                X, y = load_wesad_data()
                st.session_state.X, st.session_state.y = X, y
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully")

        if st.session_state.get("data_loaded", False):
            st.metric("Samples", len(st.session_state.X))
            st.metric("Features", st.session_state.X.shape[1])
            st.metric("Stress Cases", int(np.sum(st.session_state.y == 1)))
            st.metric("Non-Stress", int(np.sum(st.session_state.y == 0)))

        st.markdown("---")

        # Model training
        st.subheader("🤖 Model Training")
        if st.button("Train Models", use_container_width=True,
                     disabled=not st.session_state.get("data_loaded", False)):
            with st.spinner("Training models..."):
                X, y = st.session_state.X, st.session_state.y
                models, evaluations = train_all_models(X, y)
                st.session_state.models = models
                st.session_state.evaluations = evaluations
                st.session_state.models_trained = True
                st.session_state.results = models.results  # store predictions for ROC/PR
                st.session_state.X_test = models.X_test    # save test data for metrics
                st.session_state.y_test = models.y_test
                st.success("✅ Training complete!")

        if st.session_state.get("models_trained", False):
            st.success("✅ Models trained")
            if st.button("View Evaluation Metrics", use_container_width=True):
                st.session_state.page = "evaluation"
                st.experimental_rerun()

    # ---------- Main Area ----------
    st.header("🔬 About the System")
    st.markdown("""
    This system uses **WESAD physiological signals** to classify stress.
    - **ACC:** Accelerometer (movement)
    - **BVP:** Blood Volume Pulse (heart activity)
    - **EDA:** Electrodermal Activity (sweat response)
    - **TEMP:** Skin Temperature
    """)

    if st.session_state.get("data_loaded", False):
        stress_count = np.sum(st.session_state.y == 1)
        non_stress_count = np.sum(st.session_state.y == 0)
        fig = px.pie(
            values=[stress_count, non_stress_count],
            names=['Stress', 'No Stress'],
            color_discrete_map={'Stress': '#ef5350', 'No Stress': '#66bb6a'},
            title="Label Distribution"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load data to see statistics.")

    if not st.session_state.get("models_trained", False):
        st.markdown("---")
        st.info("""
        ### 🚀 Steps:
        1. Click **Load Data** (sidebar)
        2. Click **Train Models**
        3. Click **View Evaluation Metrics**
        """)


# =================== EVALUATION PAGE ===================
def evaluation_page():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white;">📊 Model Evaluation Metrics</h1>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("models_trained", False):
        st.warning("Please train models first.")
        return

    evaluations = st.session_state.evaluations

    st.header("📈 Model Performance Comparison")

    metrics_df = pd.DataFrame({
        "Model": list(evaluations.keys()),
        "Accuracy": [evaluations[m]['accuracy'] for m in evaluations],
        "Precision": [evaluations[m]['precision'] for m in evaluations],
        "Recall": [evaluations[m]['recall'] for m in evaluations],
        "Specificity": [evaluations[m]['specificity'] for m in evaluations],
        "Balanced Accuracy": [evaluations[m]['balanced_accuracy'] for m in evaluations],
        "F1 Score": [evaluations[m]['f1_score'] for m in evaluations],
        "AUC": [evaluations[m]['auc'] for m in evaluations],
        "MCC": [evaluations[m]['mcc'] for m in evaluations],
        "Cohen’s Kappa": [evaluations[m]['kappa'] for m in evaluations]
    })

    # Highlight best per column
    styled_df = metrics_df.style.highlight_max(axis=0, color='lightgreen')
    st.dataframe(styled_df, use_container_width=True)

    # Save figure for research paper
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(metrics_df.set_index("Model"), annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
    plt.title("Model Evaluation Summary")
    plt.tight_layout()
    plt.savefig("summary_table.png", dpi=300)
    st.success("✅ Summary table image saved as `summary_table.png`")

    # Bar chart for visual comparison
    fig = px.bar(metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                 x='Model', y='Score', color='Metric', barmode='group',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True, height=450)

    # Confusion matrices
    st.header("🔍 Confusion Matrices")
    cols = st.columns(len(evaluations))
    for idx, (model_name, eval_data) in enumerate(evaluations.items()):
        with cols[idx]:
            cm = eval_data["confusion_matrix"]
            fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"),
                            x=['No Stress', 'Stress'], y=['No Stress', 'Stress'],
                            title=model_name, color_continuous_scale='Blues', text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    # ROC & Precision-Recall curves
    st.header("📉 ROC & Precision-Recall Curves")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, eval_data in evaluations.items():
        y_prob = st.session_state.results[model_name].get('y_prob')
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(st.session_state.y_test, y_prob)
            prec, rec, _ = precision_recall_curve(st.session_state.y_test, y_prob)
            ax[0].plot(fpr, tpr, label=f"{model_name} (AUC={eval_data['auc']:.2f})")
            ax[1].plot(rec, prec, label=f"{model_name}")

    ax[0].set_title("ROC Curve")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].legend()

    ax[1].set_title("Precision-Recall Curve")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend()

    st.pyplot(fig)

    if st.button("← Back to Main Page"):
        st.session_state.page = "main"
        st.experimental_rerun()


# =================== APP ENTRY ===================
if 'page' not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "evaluation":
    evaluation_page()
