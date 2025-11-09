# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
python setup.py
```

This will install all required packages and create necessary directories.

### Step 2: Download Dataset (Optional)
The app works with synthetic data, but for real predictions, download the WESAD dataset:

```bash
python download_data.py
```

Or manually:
1. Go to https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset
2. Download `S2.zip`
3. Extract to `data/S2/` directory

### Step 3: Run the App
```bash
streamlit run app.py
```

The app will open in your browser automatically.

## 📱 Using the App

### Main Page
1. **Load Data**: Click "Load Data" in the sidebar
   - If WESAD data is available, it will load real data
   - Otherwise, synthetic data will be used

2. **Train Models**: Click "Train Models" to train all ML models
   - This may take a few minutes
   - Five models will be trained: Random Forest, SVM, Neural Network, Logistic Regression, Gradient Boosting

3. **Make Predictions**: 
   - Select a model from the dropdown
   - Choose number of samples
   - Click "Predict" to see results

### Evaluation Page
1. Click "View Evaluation Metrics" button
2. See comprehensive performance metrics:
   - Accuracy, Precision, Recall, F1 Score
   - Confusion matrices
   - Training history (for Neural Network)
   - Comparison charts

## 🔧 Troubleshooting

### Dataset Not Found
- The app will automatically use synthetic data
- To use real data, ensure `data/S2/S2.mat` exists

### Models Not Training
- Check that data is loaded first
- Ensure sufficient memory (Neural Network requires more memory)
- Try reducing the dataset size if memory is limited

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

## 📊 Features

### Models Included
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Machine with RBF kernel
- **Neural Network**: Deep learning model with 4 hidden layers
- **Logistic Regression**: Linear classifier
- **Gradient Boosting**: Sequential ensemble learning

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Training History (Neural Network)

## 🎯 Project Structure

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
└── data/                  # Dataset directory
    └── S2/                # Subject S2 data
```

## 📝 Notes

- The app uses Subject S2 from WESAD dataset for prototyping
- Synthetic data is used when real data is not available
- All models are trained on the same data for fair comparison
- Models are saved automatically (optional)

## 🆘 Need Help?

1. Check the README.md for detailed documentation
2. Run `python test_models.py` to verify installation
3. Ensure all dependencies are installed correctly

