# Stress Level Prediction Project - Summary

## 📋 Project Overview

This project implements a **Stress Level Prediction System** using physiological signals from the WESAD (Wearable Stress and Affect Detection) dataset. The system uses multiple Machine Learning and Deep Learning models to predict stress levels from wearable sensor data.

## 🎯 Key Features

### 1. Multiple ML Models
- **Random Forest Classifier**: Ensemble learning with decision trees
- **Support Vector Machine (SVM)**: RBF kernel-based classification
- **Neural Network**: Deep learning model with 4 hidden layers
- **Logistic Regression**: Linear classification model
- **Gradient Boosting**: Sequential ensemble learning

### 2. Beautiful Web Interface
- Modern Streamlit-based UI with gradient styling
- Interactive visualizations using Plotly
- Real-time predictions
- Comprehensive evaluation metrics page

### 3. Data Processing
- Handles WESAD dataset structure
- Feature extraction from physiological signals:
  - ACC (Accelerometer)
  - BVP (Blood Volume Pulse)
  - EDA (Electrodermal Activity)
  - TEMP (Temperature)
- Statistical and frequency domain features
- Wavelet transform features

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- Confusion matrices
- Classification reports
- Training history visualization (Neural Network)
- Model comparison charts

## 📊 Dataset

**WESAD Dataset (Subject S2)**
- Source: UCI ML Repository / Kaggle
- Size: ~180 MB (S2 only)
- Signals: ACC, BVP, EDA, TEMP
- Labels: Baseline, Stress, Amusement, Meditation

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Libraries**: scikit-learn, TensorFlow/Keras
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: NumPy, Pandas, SciPy
- **Feature Extraction**: PyWavelets

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data loading and preprocessing
├── models.py              # ML models implementation
├── download_data.py       # Dataset download script
├── setup.py               # Setup script
├── test_models.py         # Test script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── QUICKSTART.md          # Quick start guide
├── PROJECT_SUMMARY.md     # This file
└── data/                  # Dataset directory
    └── S2/                # Subject S2 data
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   python setup.py
   ```

2. **Download dataset** (optional):
   ```bash
   python download_data.py
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 📈 Model Performance

The system trains all models on the same dataset and provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## 🎨 UI Features

### Main Page
- Data loading interface
- Model training controls
- Real-time predictions
- Interactive visualizations

### Evaluation Page
- Performance metrics comparison
- Confusion matrices
- Training history (Neural Network)
- Detailed classification reports
- Radar charts for model comparison

## 🔬 Methodology

1. **Data Preprocessing**:
   - Window-based feature extraction (60-second windows)
   - Statistical features (mean, std, min, max, percentiles)
   - Frequency domain features (FFT, spectral centroid)
   - Wavelet transform features

2. **Model Training**:
   - Train-test split (80-20)
   - StandardScaler for feature normalization
   - Early stopping for Neural Network
   - Cross-validation ready

3. **Evaluation**:
   - Comprehensive metrics for each model
   - Visual comparisons
   - Detailed classification reports

## 📝 Notes

- The app works with synthetic data if WESAD dataset is not available
- Subject S2 is used for prototyping (can be extended to other subjects)
- All models are trained on the same data for fair comparison
- Models can be saved and loaded for future use

## 🎓 Educational Value

This project demonstrates:
- Multi-model ML pipeline
- Feature engineering from time-series data
- Deep learning implementation
- Web app development with Streamlit
- Model evaluation and comparison
- Data visualization

## 🔮 Future Enhancements

- Support for multiple subjects
- Real-time data streaming
- Model ensemble methods
- Hyperparameter tuning interface
- Export predictions to CSV
- Model deployment options

## 📚 References

- WESAD Dataset: https://archive.ics.uci.edu/ml/datasets/WESAD
- Kaggle: https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset

## 👨‍💻 Author

Developed as a comprehensive ML project for stress prediction from physiological signals.

