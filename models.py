"""
Machine Learning Models for Stress Prediction
Includes multiple models: Random Forest, SVM, Neural Network, etc.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os


class StressPredictionModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_names = [
            'RandomForest',
            'SVM',
            'NeuralNetwork',
            'LogisticRegression',
            'GradientBoosting'
        ]
        self.results = {}

    # -------------------------------------------------------------
    def create_neural_network(self, input_dim):
        """Create a deep neural network model"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # ✅ Explicit metric objects (prevents 'str' object errors)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        return model

    # -------------------------------------------------------------
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train all models on the data"""
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"✅ Label distribution before training: {dict(zip(unique_classes, counts))}")

        # Handle single-class case
        if len(unique_classes) < 2:
            print("⚠ Only one class present — generating synthetic balancing.")
            X = np.vstack([X, X + np.random.normal(0, 0.1, X.shape)])
            y = np.concatenate([
                y,
                np.ones_like(y) if unique_classes[0] == 0 else np.zeros_like(y)
            ])
            print(f"✅ Synthetic balancing complete: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Split safely
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        for model_name in self.model_names:
            print(f"Training {model_name}...")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler

            # ---------------------------------------------------------
            if model_name == 'NeuralNetwork':
                model = self.create_neural_network(X_train.shape[1])

                # ✅ Ensure y are float32 arrays for Keras
                y_train_nn = np.asarray(y_train, dtype=np.float32)
                y_test_nn = np.asarray(y_test, dtype=np.float32)

                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )

                history = model.fit(
                    X_train_scaled, y_train_nn,
                    epochs=50, batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Predictions
                y_pred_proba = model.predict(X_test_scaled, verbose=0).reshape(-1)
                y_pred = (y_pred_proba > 0.5).astype(int)

                self.models[model_name] = model
                self.results[model_name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                continue

            # ---------------------------------------------------------
            # Classical ML Models
            if model_name == 'RandomForest':
                model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
            elif model_name == 'SVM':
                model = SVC(kernel='rbf', probability=True, random_state=random_state)
            elif model_name == 'LogisticRegression':
                model = LogisticRegression(max_iter=1000, random_state=random_state)
            elif model_name == 'GradientBoosting':
                model = GradientBoostingClassifier(n_estimators=100, random_state=random_state, max_depth=5)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # ✅ Safe handling for models missing second class in predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test_scaled)
                y_pred_proba = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(probs))
            else:
                y_pred_proba = np.zeros(len(y_pred))

            self.models[model_name] = model
            self.results[model_name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

        return self.results

    # -------------------------------------------------------------
    def evaluate_models(self):
        """Evaluate all trained models"""
        evaluations = {}
        for model_name in self.model_names:
            if model_name not in self.results:
                continue

            y_pred = self.results[model_name]['y_pred']
            y_prob = self.results[model_name].get('y_prob', None)  # for AUC

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            cm = confusion_matrix(self.y_test, y_pred)

            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            balanced_acc = balanced_accuracy_score(self.y_test, y_pred)
            mcc = matthews_corrcoef(self.y_test, y_pred)
            kappa = cohen_kappa_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_prob) if y_prob is not None else np.nan

            evaluations[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'auc': auc,
            'mcc': mcc,
            'kappa': kappa,
            'confusion_matrix': cm,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
        return evaluations

    # -------------------------------------------------------------
    def predict(self, X, model_name='RandomForest'):
        """Make predictions using a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        scaler = self.scalers[model_name]
        X_scaled = scaler.transform(X)
        model = self.models[model_name]

        if model_name == 'NeuralNetwork':
            y_pred_proba = model.predict(X_scaled, verbose=0).reshape(-1)
            y_pred = (y_pred_proba > 0.5).astype(int)
            return y_pred, y_pred_proba

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)
            y_pred_proba = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(probs))
        else:
            y_pred_proba = np.zeros(len(X_scaled))

        y_pred = model.predict(X_scaled)
        return y_pred, y_pred_proba

    # -------------------------------------------------------------
    def save_models(self, save_dir='saved_models'):
        os.makedirs(save_dir, exist_ok=True)
        for model_name in self.model_names:
            if model_name in self.models:
                if model_name == 'NeuralNetwork':
                    self.models[model_name].save(os.path.join(save_dir, f'{model_name}.h5'))
                else:
                    joblib.dump(self.models[model_name], os.path.join(save_dir, f'{model_name}.pkl'))

                if model_name in self.scalers:
                    joblib.dump(self.scalers[model_name], os.path.join(save_dir, f'{model_name}_scaler.pkl'))

    # -------------------------------------------------------------
    def load_models(self, save_dir='saved_models'):
        for model_name in self.model_names:
            if model_name == 'NeuralNetwork':
                model_path = os.path.join(save_dir, f'{model_name}.h5')
                if os.path.exists(model_path):
                    self.models[model_name] = keras.models.load_model(model_path)
            else:
                model_path = os.path.join(save_dir, f'{model_name}.pkl')
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)

            scaler_path = os.path.join(save_dir, f'{model_name}_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
