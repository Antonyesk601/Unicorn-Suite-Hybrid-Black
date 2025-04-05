import numpy as np
from scipy import signal
import pickle
from joblib import load
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model


def preprocess_eeg(X, sampling_freq=250, notch_freq=60, lowcut=0.5, highcut=30, scaling_factor=50e-6):
    """
    Preprocess EEG data by applying a notch filter and a bandpass filter,
    and scale the data to a specified range.

    Parameters:
    - X: numpy array, shape (trials, channels, samples)
        Input EEG data array.
    - sampling_freq: int, optional
        Sampling frequency of the EEG data in Hz (default: 250 Hz).
    - notch_freq: float, optional
        Frequency of the notch filter in Hz (default: 60 Hz).
    - lowcut: float, optional
        Low cutoff frequency of the bandpass filter in Hz (default: 0.5 Hz).
    - highcut: float, optional
        High cutoff frequency of the bandpass filter in Hz (default: 30 Hz).
    - scaling_factor: float, optional
        Scaling factor to adjust the amplitude range of the EEG data (default: 50e-6).

    Returns:
    - filtered_eeg_data: numpy array, shape (trials, channels, samples)
        Preprocessed EEG data after applying notch and bandpass filters,
        and scaling to the specified range.
    """
    # Design the notch filter
    Q = 30  # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs=sampling_freq)

    # Design the bandpass filter
    nyquist_freq = 0.5 * sampling_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b_bandpass, a_bandpass = signal.butter(4, [low, high], btype='band')

    # Initialize filtered EEG data array
    filtered_eeg_data = np.zeros_like(X)

    # Apply notch and bandpass filters to each trial and channel
    for trial in range(X.shape[0]):
        for channel in range(X.shape[1]):
            # Apply notch filter
            eeg_notch_filtered = signal.filtfilt(b_notch, a_notch, X[trial, channel, :])
            # Apply bandpass filter
            filtered_eeg_data[trial, channel, :] = signal.filtfilt(b_bandpass, a_bandpass, eeg_notch_filtered)

    # Scale the filtered EEG data to ±50 µV
    filtered_eeg_data *= scaling_factor

    return filtered_eeg_data


# Load XGBoost model
with open('/kaggle/working/xgb_model.pkl', 'rb') as f:
    loaded_xgb = pickle.load(f)

# Load scaler and label encoder
scaler = load('/kaggle/working/minmaxscaling.pkl')
label_encoder = load('/kaggle/working/label_encoder.pkl')

# Load autoencoder model
loaded_autoencoder = load_model('/kaggle/working/encoder_model.h5')


# Assuming X is your EEG data array, shape (trials, channels, samples)
EEG_data = preprocess_eeg(X)

# Reshape EEG data for autoencoder input
EEG_reshaped = EEG_data.reshape(EEG_data.shape[0], -1)

# Use autoencoder to predict decoded EEG data
decoded_EEG_data = loaded_autoencoder.predict(EEG_reshaped)

# Transform decoded data using the loaded scaler
X_scaled = scaler.transform(decoded_EEG_data)

# Predict using XGBoost model
y_pred_encoded = loaded_xgb.predict(X_scaled)

# Inverse transform predictions using label encoder
predictions = label_encoder.inverse_transform(y_pred_encoded)

# Example usage or further processing with predictions
print("Predictions:")
print(predictions)
