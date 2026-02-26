"""
Streamlit App for Stress Level Prediction
Optimized version: cached data loading, fixed UI flow, evaluation working
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Hide TensorFlow & oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Streamlit cache settings
os.environ["STREAMLIT_CACHE_DIR"] = "./.streamlit_cache"

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
@st.cache_data(show_spinner=True)
def load_wesad_data():
    processor = WESADDataProcessor()
    try:
        X, y = processor.process_subject_data('data/S2')
        st.success("✅ Data loaded successfully!")
        return X, y, processor
    except Exception as e:
        st.warning(f"⚠️ Could not load real data: {e}")
        st.info("📊 Using synthetic data for demonstration...")
        X, y = processor.create_sample_data(n_samples=1000)
        return X, y, processor

# =================== MODEL TRAINING ===================
@st.cache_resource(show_spinner=True)
def train_all_models(X, y):
    models = StressPredictionModels()
    with st.spinner("🔄 Training models... This may take a minute."):
        results = models.train_models(X, y, test_size=0.2)
    st.success("✅ All models trained successfully!")
    return models, results

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
        st.header("⚙️ Settings")
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

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
        st.info(f"✅ Data loaded: {st.session_state.X.shape[0]} samples, {st.session_state.X.shape[1]} features")
    else:
        if st.button("📥 Load Dataset", use_container_width=True):
            X, y, processor = load_wesad_data()
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.processor = processor
            st.session_state.data_loaded = True
            st.rerun()

    if not st.session_state.get("models_trained", False):
        if st.session_state.get("data_loaded", False):
            if st.button("🚀 Train Models", use_container_width=True):
                models, results = train_all_models(st.session_state.X, st.session_state.y)
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.models_trained = True
                st.session_state.X_train = models.X_train
                st.session_state.X_test = models.X_test
                st.session_state.y_train = models.y_train
                st.session_state.y_test = models.y_test
                evaluations = models.evaluate_models()
                st.session_state.evaluations = evaluations
                st.rerun()

# =================== EVALUATION PAGE ===================
def evaluation_page():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white;">📊 Model Evaluation Metrics</h1>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("models_trained", False):
        st.warning("⚠️ Please train models first on the main page!")
        return

    evaluations = st.session_state.evaluations

    st.header("📈 Model Performance Comparison")

    metrics_df = pd.DataFrame({
        'Model': list(evaluations.keys()),
        'Accuracy': [evaluations[m]['accuracy'] for m in evaluations],
        'Precision': [evaluations[m]['precision'] for m in evaluations],
        'Recall': [evaluations[m]['recall'] for m in evaluations],
        'F1 Score': [evaluations[m]['f1_score'] for m in evaluations]
    })

    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score']), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(metrics_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                     title='Model Performance Metrics', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            fig.add_trace(go.Scatterpolar(
                r=metrics_df[metric],
                theta=metrics_df['Model'],
                fill='toself',
                name=metric
            ))
        fig.update_layout(title='Radar Chart: Model Comparison', polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)

# =================== PREDICTION SIMULATION PAGE ===================
def prediction_simulation_page():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white;">🔮 Stress Level Prediction Simulator</h1>
        <p style="color: white;">Enter physiological signal values to predict stress level</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("models_trained", False):
        st.warning("⚠️ Please train models first on the main page!")
        return

    # Get data ranges from training data
    X_train = st.session_state.X_train
    
    st.header("📋 Input Physiological Signals")
    st.markdown("""
    Adjust the sliders below for different physiological signal values.
    The ranges are based on your training data.
    """)

    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    # ========== ACC (Accelerometer) ==========
    with col1:
        st.subheader("📍 ACC (Accelerometer)")
        acc_min, acc_max = float(X_train[:, 0].min()), float(X_train[:, 0].max())
        
        st.info(f"📊 Range: {acc_min:.2f} to {acc_max:.2f}")
        acc_mean_slider = st.slider(
            "ACC Mean",
            min_value=acc_min,
            max_value=acc_max,
            value=(acc_min + acc_max) / 2,
            step=(acc_max - acc_min) / 100,
            key="acc_mean",
            help=f"Accelerometer mean value (movement activity)"
        )
        acc_std_slider = st.slider(
            "ACC Std Dev",
            min_value=0.0,
            max_value=float(X_train[:, 1].max()),
            value=float(X_train[:, 1].mean()),
            step=float(X_train[:, 1].max()) / 100,
            key="acc_std",
            help=f"Accelerometer standard deviation (movement variability)"
        )

    # ========== BVP (Blood Volume Pulse) ==========
    with col2:
        st.subheader("❤️ BVP (Blood Volume Pulse)")
        bvp_min, bvp_max = float(X_train[:, 20].min()), float(X_train[:, 20].max())
        
        st.info(f"📊 Range: {bvp_min:.2f} to {bvp_max:.2f}")
        bvp_mean_slider = st.slider(
            "BVP Mean",
            min_value=bvp_min,
            max_value=bvp_max,
            value=(bvp_min + bvp_max) / 2,
            step=(bvp_max - bvp_min) / 100,
            key="bvp_mean",
            help=f"Blood Volume Pulse mean (heart rate activity)"
        )
        bvp_std_slider = st.slider(
            "BVP Std Dev",
            min_value=0.0,
            max_value=float(X_train[:, 21].max()),
            value=float(X_train[:, 21].mean()),
            step=float(X_train[:, 21].max()) / 100,
            key="bvp_std",
            help=f"Blood Volume Pulse standard deviation (heart rate variability)"
        )

    col3, col4 = st.columns(2)
    
    # ========== EDA (Electrodermal Activity) ==========
    with col3:
        st.subheader("💧 EDA (Electrodermal Activity)")
        eda_min, eda_max = float(X_train[:, 40].min()), float(X_train[:, 40].max())
        
        st.info(f"📊 Range: {eda_min:.2f} to {eda_max:.2f}")
        eda_mean_slider = st.slider(
            "EDA Mean",
            min_value=eda_min,
            max_value=eda_max,
            value=(eda_min + eda_max) / 2,
            step=(eda_max - eda_min) / 100,
            key="eda_mean",
            help=f"Electrodermal Activity mean (sweat response)"
        )
        eda_std_slider = st.slider(
            "EDA Std Dev",
            min_value=0.0,
            max_value=float(X_train[:, 41].max()),
            value=float(X_train[:, 41].mean()),
            step=float(X_train[:, 41].max()) / 100,
            key="eda_std",
            help=f"Electrodermal Activity standard deviation"
        )

    # ========== TEMP (Temperature) ==========
    with col4:
        st.subheader("🌡️ TEMP (Skin Temperature)")
        temp_min, temp_max = float(X_train[:, 60].min()), float(X_train[:, 60].max())
        
        st.info(f"📊 Range: {temp_min:.2f} to {temp_max:.2f}")
        temp_mean_slider = st.slider(
            "TEMP Mean",
            min_value=temp_min,
            max_value=temp_max,
            value=(temp_min + temp_max) / 2,
            step=(temp_max - temp_min) / 100,
            key="temp_mean",
            help=f"Skin Temperature mean value"
        )
        temp_std_slider = st.slider(
            "TEMP Std Dev",
            min_value=0.0,
            max_value=float(X_train[:, 61].max()),
            value=float(X_train[:, 61].mean()),
            step=float(X_train[:, 61].max()) / 100,
            key="temp_std",
            help=f"Skin Temperature standard deviation"
        )

    # ========== Model Selection ==========
    st.header("🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose a model for prediction:",
        options=['RandomForest', 'SVM', 'NeuralNetwork', 'LogisticRegression', 'GradientBoosting'],
        index=0,
        help="Select which trained model to use for prediction"
    )

    # ========== Predict Button ==========
    if st.button("🚀 Predict Stress Level", use_container_width=True):
        # Create input features (80 features: 20 per signal)
        input_features = np.zeros(80)
        
        # Fill in the mean and std values for each signal type
        input_features[0] = acc_mean_slider
        input_features[1] = acc_std_slider
        input_features[20] = bvp_mean_slider
        input_features[21] = bvp_std_slider
        input_features[40] = eda_mean_slider
        input_features[41] = eda_std_slider
        input_features[60] = temp_mean_slider
        input_features[61] = temp_std_slider
        
        input_features = input_features.reshape(1, -1)

        # Make prediction
        models = st.session_state.models
        prediction, probability = models.predict(input_features, model_name=selected_model)

        # Display Results
        st.success("✅ Prediction Complete!")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            stress_level = "🔴 HIGH STRESS" if prediction[0] == 1 else "🟢 NO STRESS"
            st.markdown(f"<h2 style='text-align: center;'>{stress_level}</h2>", unsafe_allow_html=True)

        with result_col2:
            st.metric(
                label="Stress Probability",
                value=f"{probability[0]*100:.2f}%",
                delta=f"{'High' if probability[0] > 0.7 else 'Moderate' if probability[0] > 0.4 else 'Low'} confidence"
            )

        with result_col3:
            st.metric(
                label="Model Used",
                value=selected_model
            )

        # Visualization
        st.subheader("📊 Prediction Details")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart for stress probability
        colors = ['#FF6B6B', '#4ECDC4']
        sizes = [probability[0], 1 - probability[0]]
        labels = ['Stress', 'No Stress']
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Stress Probability Distribution')
        
        # Bar chart for input signals
        signal_names = ['ACC\nMean', 'BVP\nMean', 'EDA\nMean', 'TEMP\nMean']
        signal_values = [acc_mean_slider, bvp_mean_slider, eda_mean_slider, temp_mean_slider]
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ax2.bar(signal_names, signal_values, color=colors_bar)
        ax2.set_title('Input Signal Values')
        ax2.set_ylabel('Value')
        
        plt.tight_layout()
        st.pyplot(fig)

        # Display input values summary
        st.subheader("📝 Input Summary")
        summary_df = pd.DataFrame({
            'Signal': ['ACC (Accelerometer)', 'BVP (Blood Volume)', 'EDA (Sweat Response)', 'TEMP (Skin Temp)'],
            'Mean': [f"{acc_mean_slider:.2f}", f"{bvp_mean_slider:.2f}", f"{eda_mean_slider:.2f}", f"{temp_mean_slider:.2f}"],
            'Std Dev': [f"{acc_std_slider:.2f}", f"{bvp_std_slider:.2f}", f"{eda_std_slider:.2f}", f"{temp_std_slider:.2f}"]
        })
        st.dataframe(summary_df, use_container_width=True)

# =================== APP ENTRY ===================
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Sidebar Navigation
with st.sidebar:
    st.title("📑 Navigation")
    
    page_options = ["🏠 Main", "📊 Evaluation", "🔮 Predict Stress"]
    page_values = ["main", "evaluation", "prediction"]
    
    current_index = page_values.index(st.session_state.page) if st.session_state.page in page_values else 0
    
    page_selection = st.radio(
        "Go to:",
        options=page_options,
        index=current_index
    )
    
    selected_page = page_values[page_options.index(page_selection)]
    st.session_state.page = selected_page

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "evaluation":
    evaluation_page()
elif st.session_state.page == "prediction":
    prediction_simulation_page()