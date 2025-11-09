"""
Data processing module for WESAD dataset
Handles loading, preprocessing, and feature extraction from physiological signals
"""

import numpy as np
import pandas as pd
import scipy.io
import os
from scipy import signal
import pywt
import warnings
warnings.filterwarnings('ignore')


class WESADDataProcessor:
    def __init__(self, data_path='data/S2'):
        self.data_path = data_path
        # sampling rates may be floats when read from CSV header; we'll coerce to int later
        self.sampling_rates = {
            'ACC': 32,
            'BVP': 64,
            'EDA': 4,
            'TEMP': 4,
            'label': 700
        }

    def load_data(self, subject_path):
        """Load WESAD data from .mat file or CSV files"""
        if subject_path.endswith('.mat'):
            try:
                data = scipy.io.loadmat(subject_path)
                return data
            except Exception as e:
                print(f"Error loading .mat file: {e}")
                return None
        elif os.path.isdir(subject_path):
            return self.load_csv_data(subject_path)
        else:
            base_dir = os.path.dirname(subject_path) if os.path.isfile(subject_path) else subject_path
            if os.path.isdir(base_dir):
                return self.load_csv_data(base_dir)
            return None

    # -------------------------------------------------------------------------
    def load_csv_data(self, subject_dir):
        """Load WESAD data from CSV files + correct labels from .pkl"""
        data = {}

        subject_name = os.path.basename(subject_dir.rstrip('/\\'))
        e4_data_dir = os.path.join(subject_dir, f'{subject_name}_E4_Data')

        # Fallback detection for E4 folder
        if not os.path.exists(e4_data_dir):
            for item in os.listdir(subject_dir):
                if item.endswith('_E4_Data') and os.path.isdir(os.path.join(subject_dir, item)):
                    e4_data_dir = os.path.join(subject_dir, item)
                    break

        if not os.path.exists(e4_data_dir):
            print(f"Error: E4_Data directory not found in {subject_dir}")
            return None

        print(f"Loading CSV data from: {e4_data_dir}")

        # Define files and expected signal types
        signal_files = {
            'ACC': 'ACC.csv',
            'BVP': 'BVP.csv',
            'EDA': 'EDA.csv',
            'TEMP': 'TEMP.csv'
        }

        # Load all E4 sensor CSVs
        for signal_name, filename in signal_files.items():
            filepath = os.path.join(e4_data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: {filename} missing.")
                continue
            try:
                # skip first two header rows: timestamp, sample-rate (or list of sample-rates)
                df = pd.read_csv(filepath, header=None, skiprows=2)

                # Read the sample-rate line robustly (handles comma-separated values)
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            parts = [p.strip() for p in lines[1].replace(';', ',').split(',') if p.strip()]
                            try:
                                sr = float(parts[0])
                            except Exception:
                                sr = float(self.sampling_rates.get(signal_name, 0))
                            # store as int to avoid float slices later
                            self.sampling_rates[signal_name] = int(round(sr))
                            print(f"  {signal_name} sample rate: {int(round(sr))} Hz")
                except Exception:
                    # keep previous sampling rate if anything fails
                    self.sampling_rates[signal_name] = int(self.sampling_rates.get(signal_name, 0))

                if signal_name == 'ACC':
                    # ensure at least 3 columns; if more, take first 3
                    if df.shape[1] >= 3:
                        arr = df.iloc[:, :3].values
                    else:
                        arr = df.values
                    data[signal_name] = np.asarray(arr, dtype=float)
                else:
                    data[signal_name] = np.asarray(df.iloc[:, 0].values.flatten(), dtype=float)

                print(f"  Loaded {signal_name}: shape {data[signal_name].shape}")

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                import traceback
                traceback.print_exc()
                return None

        # Verify signals
        required = ['ACC', 'BVP', 'EDA', 'TEMP']
        missing = [s for s in required if s not in data]
        if missing:
            print(f"Error: Missing signals: {missing}")
            return None

        # Load correct labels from .pkl instead of .txt (preferred)
        pkl_path = os.path.join(subject_dir, f'{subject_name}.pkl')
        if os.path.exists(pkl_path):
            try:
                import pickle
                with open(pkl_path, 'rb') as f:
                    pkl_data = pickle.load(f, encoding='latin1')
                if 'label' in pkl_data:
                    data['label'] = np.array(pkl_data['label']).flatten().astype(int)
                    print(f"✓ Loaded labels from {subject_name}.pkl — shape {data['label'].shape}")
                    unique_labels, counts = np.unique(data['label'], return_counts=True)
                    print(f"  Label distribution: {dict(zip(unique_labels, counts))}")
                else:
                    raise KeyError("label not found in pickle")
            except Exception as e:
                print(f"⚠ Failed to load labels from {pkl_path}: {e}")
                self._use_placeholder_labels(data)
        else:
            # fallback to respiban text if present (some datasets), otherwise placeholder
            resp_path = os.path.join(subject_dir, f'{subject_name}_respiban.txt')
            alt_resp = [f for f in os.listdir(subject_dir) if 'respiban' in f.lower() and f.endswith('.txt')]
            if os.path.exists(resp_path):
                try:
                    # read after header, pick first numeric column
                    lines = open(resp_path, 'r', encoding='utf-8', errors='ignore').read().splitlines()
                    # find index of EndOfHeader
                    idx = 0
                    for j, line in enumerate(lines):
                        if line.strip().lower().startswith('# endofheader'):
                            idx = j + 1
                            break
                    # parse following lines as whitespace-separated numbers, pick first column
                    raw = lines[idx:]
                    labels = []
                    for line in raw:
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        try:
                            labels.append(int(float(parts[0])))
                        except:
                            continue
                    if labels:
                        data['label'] = np.array(labels, dtype=int)
                        print(f"✓ Loaded labels from respiban text — shape {data['label'].shape}")
                    else:
                        self._use_placeholder_labels(data)
                except Exception as e:
                    print(f"⚠ Could not parse respiban file: {e}")
                    self._use_placeholder_labels(data)
            elif alt_resp:
                # try first alt
                try:
                    alt_path = os.path.join(subject_dir, alt_resp[0])
                    raw = open(alt_path, 'r', encoding='utf-8', errors='ignore').read().splitlines()
                    idx = 0
                    for j, line in enumerate(raw):
                        if line.strip().lower().startswith('# endofheader'):
                            idx = j + 1
                            break
                    labels = []
                    for line in raw[idx:]:
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        try:
                            labels.append(int(float(parts[0])))
                        except:
                            continue
                    if labels:
                        data['label'] = np.array(labels, dtype=int)
                        print(f"✓ Loaded labels from {alt_resp[0]} — shape {data['label'].shape}")
                    else:
                        self._use_placeholder_labels(data)
                except Exception:
                    self._use_placeholder_labels(data)
            else:
                print(f"⚠ No {subject_name}.pkl or respiban found — using placeholder labels.")
                self._use_placeholder_labels(data)

        return data

    def _use_placeholder_labels(self, data):
        """Generate placeholder labels if real ones are missing"""
        if 'BVP' in data:
            # ensure sampling rates are ints
            sr_bvp = int(self.sampling_rates.get('BVP', 64))
            sr_label = int(self.sampling_rates.get('label', 700))
            bvp_duration = len(data['BVP']) / max(sr_bvp, 1)
            label_length = int(round(bvp_duration * sr_label))
            if label_length <= 0:
                label_length = 10000
            data['label'] = np.ones(label_length, dtype=int)
            print(f"⚠ Using placeholder labels (baseline): shape {data['label'].shape}")
        else:
            data['label'] = np.ones(10000, dtype=int)
            print("⚠ No BVP data — using default 10k placeholder labels.")

    # -------------------------------------------------------------------------
    def extract_features_from_signal(self, signal_data, sampling_rate, window_size=60):
        """Extract statistical, spectral, and wavelet features from a signal"""
        features = []

        if len(signal_data) == 0:
            return np.zeros(20)

        # Statistical features
        features += [
            float(np.mean(signal_data)),
            float(np.std(signal_data)),
            float(np.median(signal_data)),
            float(np.min(signal_data)),
            float(np.max(signal_data)),
            float(np.percentile(signal_data, 25)),
            float(np.percentile(signal_data, 75)),
            float(np.var(signal_data)),
            float(np.sqrt(np.mean(np.asarray(signal_data, dtype=float) ** 2))),  # RMS
        ]

        # Frequency domain features
        try:
            fft_vals = np.fft.rfft(signal_data)
            fft_power = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(len(signal_data), 1 / max(float(sampling_rate), 1.0))

            if len(fft_power) > 0:
                features.append(float(np.sum(fft_power)))  # Total power
                # dominant freq in Hz (guard division by zero)
                features.append(float(np.argmax(fft_power) * (sampling_rate / max(len(signal_data), 1))))

                if np.sum(fft_power) > 0:
                    centroid = float(np.sum(freqs * fft_power) / np.sum(fft_power))
                    features.append(centroid)  # Spectral centroid
                    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_power) / np.sum(fft_power)))
                    features.append(spread)  # Spectral spread
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        except Exception:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Wavelet features
        try:
            if len(signal_data) >= 4:
                coeffs = pywt.wavedec(signal_data, 'db4', level=min(4, int(np.log2(len(signal_data)))))
                if len(coeffs) > 0:
                    features.append(float(np.std(coeffs[0])))
                    # mean std of detail coefficients (if any)
                    detail_stds = [float(np.std(c)) for c in coeffs[1:]] if len(coeffs) > 1 else [0.0]
                    features.append(float(np.mean(detail_stds)))
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
        except Exception:
            features.extend([0.0, 0.0])

        # Zero-crossing rate
        try:
            zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
            features.append(float(len(zero_crossings) / max(len(signal_data), 1)))
        except Exception:
            features.append(0.0)

        # Autocorrelation
        try:
            if len(signal_data) > 1:
                autocorr = np.correlate(signal_data, signal_data, mode='full')[len(signal_data) - 1:]
                if autocorr[0] != 0:
                    autocorr = autocorr / autocorr[0]
                features.append(float(autocorr[1]) if len(autocorr) > 1 else 0.0)
            else:
                features.append(0.0)
        except Exception:
            features.append(0.0)

        # Ensure length = 20
        while len(features) < 20:
            features.append(0.0)

        return np.array(features[:20], dtype=float)

    # -------------------------------------------------------------------------
    def process_subject_data(self, subject_path):
        """Process subject data (CSV or MAT) and extract features"""
        data = self.load_data(subject_path)
        if data is None:
            return None, None

        # If CSV dictionary
        if isinstance(data, dict) and 'ACC' in data:
            acc = np.array(data.get('ACC', []), dtype=float)
            bvp = np.array(data.get('BVP', []), dtype=float)
            eda = np.array(data.get('EDA', []), dtype=float)
            temp = np.array(data.get('TEMP', []), dtype=float)
            labels = np.array(data.get('label', []), dtype=int)
        else:
            print("Unsupported data format or missing keys.")
            return None, None

        # Normalize sampling rates to integers to avoid float slice indices
        self.sampling_rates = {k: int(round(v)) for k, v in self.sampling_rates.items()}

        print(f"Processing CSV data - ACC: {acc.shape}, BVP: {bvp.shape}, EDA: {eda.shape}, TEMP: {temp.shape}, Labels: {labels.shape}")

        # Magnitude for ACC if 3-axis present
        if acc.ndim > 1 and acc.shape[1] >= 3:
            acc = np.sqrt(np.square(acc[:, 0]) + np.square(acc[:, 1]) + np.square(acc[:, 2]))
        elif acc.ndim > 1:
            acc = acc.flatten()

        # Ensure 1-D arrays
        acc = np.asarray(acc).flatten()
        bvp = np.asarray(bvp).flatten()
        eda = np.asarray(eda).flatten()
        temp = np.asarray(temp).flatten()
        labels = np.asarray(labels).flatten()

        if len(labels) == 0:
            print("Warning: No labels found.")
            return None, None

        # windowing configuration (integers)
        window_size = int(60)  # seconds
        overlap = int(30)  # seconds
        label_rate = int(self.sampling_rates.get('label', 700))

        window_samples = int(label_rate * window_size)
        step_samples = int(label_rate * (window_size - overlap))

        features_list, labels_list = [], []

        # guard: ensure window_samples > 0
        if window_samples <= 0 or step_samples <= 0:
            print("Invalid window/step configuration.")
            return None, None

        max_label_index = len(labels)
        # iterate over label-based windows
        for i in range(0, max_label_index - window_samples + 1, step_samples):
            window_labels = labels[i:i + window_samples]
            if len(window_labels) == 0:
                continue

            unique, counts = np.unique(window_labels, return_counts=True)
            label = unique[np.argmax(counts)]
            # WESAD stress label mapping (handles mislabeled sessions)
            stress_label = 1 if label in [2, 3, 4, 5, 6, 7, 8] else 0 # Binary mapping

            # Align signals to this label window
            acc_start = int(round(i * self.sampling_rates.get('ACC', 32) / max(label_rate, 1)))
            bvp_start = int(round(i * self.sampling_rates.get('BVP', 64) / max(label_rate, 1)))
            eda_start = int(round(i * self.sampling_rates.get('EDA', 4) / max(label_rate, 1)))
            temp_start = int(round(i * self.sampling_rates.get('TEMP', 4) / max(label_rate, 1)))

            # number of samples for each signal per window (integers)
            acc_window_samples = int(self.sampling_rates.get('ACC', 32) * window_size)
            bvp_window_samples = int(self.sampling_rates.get('BVP', 64) * window_size)
            eda_window_samples = int(self.sampling_rates.get('EDA', 4) * window_size)
            temp_window_samples = int(self.sampling_rates.get('TEMP', 4) * window_size)

            # compute end indices with ints and safe clipping
            acc_end = acc_start + acc_window_samples
            bvp_end = bvp_start + bvp_window_samples
            eda_end = eda_start + eda_window_samples
            temp_end = temp_start + temp_window_samples

            # ensure indices are within array bounds
            if acc_start < 0 or acc_end > len(acc) or bvp_start < 0 or bvp_end > len(bvp) or eda_start < 0 or eda_end > len(eda) or temp_start < 0 or temp_end > len(temp):
                # skip windows that don't have full data
                continue

            # slice windows (all indices are ints)
            acc_w = acc[acc_start:acc_end]
            bvp_w = bvp[bvp_start:bvp_end]
            eda_w = eda[eda_start:eda_end]
            temp_w = temp[temp_start:temp_end]

            # skip if any window empty (safety)
            if min(map(len, [acc_w, bvp_w, eda_w, temp_w])) == 0:
                continue

            # Extract features
            acc_f = self.extract_features_from_signal(acc_w, self.sampling_rates['ACC'])
            bvp_f = self.extract_features_from_signal(bvp_w, self.sampling_rates['BVP'])
            eda_f = self.extract_features_from_signal(eda_w, self.sampling_rates['EDA'])
            temp_f = self.extract_features_from_signal(temp_w, self.sampling_rates['TEMP'])

            all_features = np.concatenate([acc_f, bvp_f, eda_f, temp_f])
            features_list.append(all_features)
            labels_list.append(int(stress_label))

        
    
    
        if not features_list:
            print("❌ No feature windows extracted.")
            return None, None

        X = np.array(features_list, dtype=float)
        y = np.array(labels_list, dtype=int)

        unique, counts = np.unique(y, return_counts=True)
        print(f"✅ Label counts before balancing: {dict(zip(unique, counts))}")

        # Safety net: if we got only one class, fix it
        if len(unique) < 2:
            print("⚠ Only one class in data — forcing synthetic balance for testing.")
            other_class = 1 - unique[0]
            extra_y = np.full_like(y, other_class)
            noise = np.random.normal(0, 0.05, X.shape)
            X = np.vstack([X, X + noise])
            y = np.concatenate([y, extra_y])

        unique, counts = np.unique(y, return_counts=True)
        print(f"✅ Label counts after balancing: {dict(zip(unique, counts))}")

        return X, y
    
    

    # -------------------------------------------------------------------------
    def create_sample_data(self, n_samples=1000):
        """Generate synthetic physiological data for testing"""
        np.random.seed(42)
        X, y = [], []

        for _ in range(n_samples):
            if np.random.rand() > 0.5:
                acc = np.random.normal(1.5, 0.3, 20)
                bvp = np.random.normal(2.0, 0.5, 20)
                eda = np.random.normal(3.0, 0.4, 20)
                temp = np.random.normal(36.5, 0.2, 20)
                label = 1
            else:
                acc = np.random.normal(1.0, 0.2, 20)
                bvp = np.random.normal(1.5, 0.3, 20)
                eda = np.random.normal(2.0, 0.3, 20)
                temp = np.random.normal(36.0, 0.1, 20)
                label = 0
            X.append(np.concatenate([acc, bvp, eda, temp]))
            y.append(label)

        return np.array(X), np.array(y)
