# Stress Level Prediction from Physiological Signals

A web application for predicting stress levels using physiological signals from the WESAD (Wearable Stress and Affect Detection) dataset.

## Features

- **Multiple ML Models**: Random Forest, SVM, Neural Networks, and more
- **Beautiful UI**: Modern Streamlit interface with interactive visualizations
- **Evaluation Metrics**: Comprehensive performance metrics and visualizations
- **Real-time Prediction**: Upload data and get stress predictions

## Dataset

This project uses the WESAD dataset (Subject S2 only for prototype):
- Available on [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/WESAD%2B%28Wearable%2BStress%2Band%2BAffect%2BDetection%29)
- Also available on [Kaggle](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset)

## Installation

### Quick Setup

1. Run the setup script:
```bash
python setup.py
```

2. Download dataset (optional - app works with synthetic data):

**Option A: Manual Download and Extract**
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset)
2. Download `S2.zip`
3. Place `S2.zip` in the project directory
4. Run extraction script:
```bash
python download_data.py
```
Or manually extract `S2.zip` to `data/S2/` directory (ensure `S2.mat` is at `data/S2/S2.mat`)

**Option B: Using Kaggle API (requires API setup)**
```bash
python download_data.py
```
Note: This requires Kaggle API credentials. For manual download, use Option A.

3. Run the application:
```bash
streamlit run app.py
```

### Manual Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create directories:
```bash
mkdir -p data/S2 saved_models
```

3. Download dataset (see above)

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Start the Streamlit app
2. Navigate to the main page for predictions
3. Click "View Evaluation Metrics" to see model performance
4. Upload physiological data or use sample data for predictions

## Models Included

- Random Forest Classifier
- Support Vector Machine (SVM)
- Neural Network (Deep Learning)
- Logistic Regression
- Gradient Boosting Classifier

