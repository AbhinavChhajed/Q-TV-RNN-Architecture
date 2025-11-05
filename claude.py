# %%
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Layer
import tensorflow as tf

# %%
# Configuration parameters
PERIOD_SIZE = 50
STEP_SIZE = 100
MEDIAN_Q_THRESHOLD = 0.5
LOWER_Q_THRESHOLD = 0.1
UPPER_Q_THRESHOLD = 0.9
ANOMALY_RES = [15, 11]  # Ground truth anomaly values
RANDOM_SEEDS = [42]  # For ensemble diversity
LOOK_BACK = PERIOD_SIZE
TW = STEP_SIZE
MULTI = LOOK_BACK // TW

# %%
# Custom Swish Activation (CORRECTED)
class Swish(Layer):
    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        # Correct Swish formula: x * sigmoid(beta * x)
        return inputs * K.sigmoid(self.beta * inputs)

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

# %%
# Utility Functions
def verify_stationarity(dataset):
    """Check if time series is stationary using ADF test"""
    is_stationary = True
    test_results = adfuller(dataset)

    print(f"ADF test statistic: {test_results[0]:.4f}")
    print(f"p-value: {test_results[1]:.4f}")
    print("Critical thresholds:")

    critical_value = None
    for i, (key, value) in enumerate(test_results[4].items()):
        print(f"\t{key}: {value:.3f}")
        if i == 0:
            critical_value = value

    if test_results[0] > critical_value:
        print('Series is non-stationary')
        is_stationary = False
    else:
        print('Series is stationary')
    
    return is_stationary


def create_dataset(dataset, look_back=1, tw=3):
    """Create quantile-based features for training"""
    dataX, dataY = [], []  # q50 (median)
    dataUpperX, dataUpperY = [], []  # q90 (upper)
    dataLowerX, dataLowerY = [], []  # q10 (lower)
    multi = look_back // tw
    
    for i in range(len(dataset) - look_back - 1):
        q50X, q90X, q10X = [], [], []
        a = dataset[i + 1:(i + look_back + 1)]
        
        # Calculate target quantiles
        c = np.quantile(a, MEDIAN_Q_THRESHOLD)
        u = np.quantile(a, UPPER_Q_THRESHOLD)
        l = np.quantile(a, LOWER_Q_THRESHOLD)
        
        # Create sub-window features
        for j in range(0, len(a), tw):
            window = a[j:j + tw]
            q50X.append(np.quantile(window, MEDIAN_Q_THRESHOLD))
            q90X.append(np.quantile(window, UPPER_Q_THRESHOLD))
            q10X.append(np.quantile(window, LOWER_Q_THRESHOLD))
        
        dataX.append(q50X)
        dataY.append(c)
        dataUpperX.append(q90X)
        dataUpperY.append(u)
        dataLowerX.append(q10X)
        dataLowerY.append(l)
    
    return (np.array(dataX), np.array(dataY), 
            np.array(dataUpperX), np.array(dataUpperY), 
            np.array(dataLowerX), np.array(dataLowerY))


def quantile_loss(q):
    """Proper quantile loss function for training"""
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e))
    return loss


def create_quantile_model(quantile, input_shape, seed=42, units=64):
    """Create improved LSTM model with proper architecture"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units // 2),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1),
        Swish(beta=1.5)  # Single activation layer
    ])
    
    model.compile(
        loss=quantile_loss(quantile),
        optimizer='adam',
        metrics=['mse']
    )
    
    return model


def predict_ensemble(models, input_data):
    """Average predictions from multiple models"""
    predictions = [model.predict(input_data, verbose=0) for model in models]
    return np.mean(predictions, axis=0)


def identify_alpha(dataset):
    """Calculate trend stability metric"""
    alpha_detection = []
    prev_slope = 1
    
    for m in range(0, len(dataset), PERIOD_SIZE):
        period_dataset = dataset[m:m + PERIOD_SIZE]
        if len(period_dataset) < 2:
            continue
        # Extract scalar values from arrays
        start_val = float(period_dataset[0][0]) if period_dataset[0].ndim > 0 else float(period_dataset[0])
        end_val = float(period_dataset[-1][0]) if period_dataset[-1].ndim > 0 else float(period_dataset[-1])
        slope = (end_val - start_val) / len(period_dataset)
        alpha = slope / prev_slope if prev_slope != 0 else 1
        alpha_detection.append(float(alpha))
        prev_slope = slope if slope != 0 else 1
    
    return float(np.abs(np.mean(alpha_detection))) if alpha_detection else 1.0

# %%
# Load and preprocess data
print("="*60)
print("LOADING AND PREPROCESSING DATA")
print("="*60)

np.random.seed(7)
dataframe = read_csv('Q-TV-RNN/Q-data/speed_t4013_train.csv', usecols=[2], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# Check for NaN/Inf
assert not np.isnan(dataset).any(), "Dataset contains NaN values"
assert not np.isinf(dataset).any(), "Dataset contains Inf values"

print(f"\nDataset shape: {dataset.shape}")
print(f"Dataset range: [{dataset.min():.2f}, {dataset.max():.2f}]")

# Verify stationarity
print("\nStationarity Test:")
stationary = verify_stationarity(dataset)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train/validation/test sets (60/20/20)
train_size = int(len(dataset) * 0.6)
val_size = int(len(dataset) * 0.2)
train = dataset[0:train_size, :]
val = dataset[train_size:train_size + val_size, :]
test = dataset[train_size + val_size:, :]

print(f"\nTrain size: {len(train)}")
print(f"Validation size: {len(val)}")
print(f"Test size: {len(test)}")

# Calculate alpha (trend metric)
alpha = identify_alpha(dataset)
print(f"\nAlpha (trend stability): {alpha:.4f}")

# %%
# Create datasets
print("\n" + "="*60)
print("CREATING QUANTILE DATASETS")
print("="*60)

trainX, trainY, trainXU, trainYU, trainXL, trainYL = create_dataset(train, LOOK_BACK, TW)
valX, valY, valXU, valYU, valXL, valYL = create_dataset(val, LOOK_BACK, TW)
testX, testY, testXU, testYU, testXL, testYL = create_dataset(test, LOOK_BACK, TW)

# Reshape for LSTM [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

trainXU = np.reshape(trainXU, (trainXU.shape[0], trainXU.shape[1], 1))
valXU = np.reshape(valXU, (valXU.shape[0], valXU.shape[1], 1))
testXU = np.reshape(testXU, (testXU.shape[0], testXU.shape[1], 1))

trainXL = np.reshape(trainXL, (trainXL.shape[0], trainXL.shape[1], 1))
valXL = np.reshape(valXL, (valXL.shape[0], valXL.shape[1], 1))
testXL = np.reshape(testXL, (testXL.shape[0], testXL.shape[1], 1))

print(f"Training input shape: {trainX.shape}")
print(f"Validation input shape: {valX.shape}")
print(f"Test input shape: {testX.shape}")

# %%
# Train ensemble models
print("\n" + "="*60)
print("TRAINING ENSEMBLE MODELS")
print("="*60)

input_shape = (MULTI, 1)
epochs = 100
batch_size = 32

# Callbacks
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=1)
]

# Train Q50 (median) models
print("\n--- Training Q50 (Median) Models ---")
models_q50 = []
for i, seed in enumerate(RANDOM_SEEDS):
    print(f"\nModel {i+1}/3 (seed={seed}):")
    model = create_quantile_model(MEDIAN_Q_THRESHOLD, input_shape, seed)
    history = model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    models_q50.append(model)
    print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")

# Train Q10 (lower) models
print("\n--- Training Q10 (Lower Bound) Models ---")
models_q10 = []
for i, seed in enumerate(RANDOM_SEEDS):
    print(f"\nModel {i+1}/3 (seed={seed}):")
    model = create_quantile_model(LOWER_Q_THRESHOLD, input_shape, seed)
    history = model.fit(
        trainXL, trainYL,
        validation_data=(valXL, valYL),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    models_q10.append(model)
    print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")

# Train Q90 (upper) models
print("\n--- Training Q90 (Upper Bound) Models ---")
models_q90 = []
for i, seed in enumerate(RANDOM_SEEDS):
    print(f"\nModel {i+1}/3 (seed={seed}):")
    model = create_quantile_model(UPPER_Q_THRESHOLD, input_shape, seed)
    history = model.fit(
        trainXU, trainYU,
        validation_data=(valXU, valYU),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    models_q90.append(model)
    print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")

print("\n✓ All ensemble models trained successfully!")

# %%
# Determine optimal threshold from validation set
print("\n" + "="*60)
print("LEARNING OPTIMAL THRESHOLD FROM VALIDATION SET")
print("="*60)

val_predictions_q50 = predict_ensemble(models_q50, valX)
val_predictions_q10 = predict_ensemble(models_q10, valXL)
val_predictions_q90 = predict_ensemble(models_q90, valXU)

val_errors = []
for i in range(len(valX)):
    actual = val[LOOK_BACK + i + 1]
    iqr = val_predictions_q90[i] - val_predictions_q10[i]
    error = abs(actual - val_predictions_q50[i]) / (iqr + 1e-6)
    val_errors.append(error[0])

# Use 95th percentile of validation errors as threshold
threshold_multiplier = np.quantile(val_errors, 0.95)
print(f"Learned threshold multiplier: {threshold_multiplier:.4f}")

# %%
# Inference on test set
print("\n" + "="*60)
print("RUNNING INFERENCE ON TEST SET")
print("="*60)

# Load labeled test data
dataframe = read_csv('Q-TV-RNN/Q-data/speed_t4013_labelled.csv', usecols=[2], engine='python')
dataset_test = dataframe.values
dataset_test = dataset_test.astype('float32')
dataset_test = scaler.transform(dataset_test)

print(f"Test dataset size: {len(dataset_test)}")

anomalies = []
anomaly_indices = []
i = 0
j = LOOK_BACK

while j < len(dataset_test) - 1:
    q50_array, q10_array, q90_array = [], [], []
    
    temp = dataset_test[i:j]
    
    # Extract quantile features
    for m in range(0, len(temp), STEP_SIZE):
        window = temp[m:m + STEP_SIZE]
        q50_array.append([np.quantile(window, MEDIAN_Q_THRESHOLD)])
        q90_array.append([np.quantile(window, UPPER_Q_THRESHOLD)])
        q10_array.append([np.quantile(window, LOWER_Q_THRESHOLD)])
    
    # Prepare inputs
    final_q50_array = np.array([q50_array])
    final_q10_array = np.array([q10_array])
    final_q90_array = np.array([q90_array])
    
    # Ensemble predictions
    q50_predict = predict_ensemble(models_q50, final_q50_array)
    q10_predict = predict_ensemble(models_q10, final_q10_array)
    q90_predict = predict_ensemble(models_q90, final_q90_array)
    
    # Anomaly detection with learned threshold
    iqr = q90_predict - q10_predict
    ucl = q50_predict + threshold_multiplier * iqr
    lcl = q50_predict - threshold_multiplier * iqr
    
    actual_value = dataset_test[j + 1]
    
    if actual_value > ucl or actual_value < lcl:
        anomalies.append(actual_value)
        anomaly_indices.append(j + 1)
    
    j += 1
    i += 1

print(f"Detected {len(anomalies)} anomalies")

# %%
# Post-processing
print("\n" + "="*60)
print("POST-PROCESSING RESULTS")
print("="*60)

# Convert back to original scale
if len(anomalies) > 0:
    anomalies_array = np.array(anomalies).reshape(-1, 1)
    anomalies_array = scaler.inverse_transform(anomalies_array)
    anomalies_unique = np.unique(anomalies_array)
else:
    anomalies_unique = np.array([])

print(f"Unique detected anomalies: {len(anomalies_unique)}")
if len(anomalies_unique) > 0:
    print(f"Anomaly values (first 10): {anomalies_unique.flatten()[:10]}")

# %%
# Visualization
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

plt.figure(figsize=(14, 6))
plt.plot(dataset_test, label="Test Dataset (Normalized)", alpha=0.7, linewidth=1)

if len(anomaly_indices) > 0:
    plt.scatter(anomaly_indices, [dataset_test[idx] for idx in anomaly_indices], 
                color='red', s=80, marker='x', 
                label=f'Detected Anomalies ({len(anomaly_indices)})', 
                zorder=5, linewidths=2)

plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Normalized Speed', fontsize=12)
plt.title('Quantile RNN Anomaly Detection Results', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Evaluation Metrics
print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)

ground_truth = np.array(ANOMALY_RES)
print(f"Ground truth anomaly values: {ground_truth}")
print(f"Detected unique anomaly values: {anomalies_unique.flatten()}")

# Calculate confusion matrix components
truep = np.intersect1d(anomalies_unique, ground_truth)
tp = len(truep)
fp = len(anomalies_unique) - tp
fn = len(ground_truth) - tp
tn = 1670 - tp - fp - fn  # Total data points

print(f"\n--- Confusion Matrix ---")
print(f"True Positives (TP):  {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN):  {tn}")

# Calculate metrics
if tp + fp > 0:
    precision = tp / (tp + fp)
    print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
else:
    precision = 0
    print("\nPrecision: 0.0000 (no anomalies detected)")

if tp + fn > 0:
    recall = tp / (tp + fn)
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
else:
    recall = 0
    print("Recall: 0.0000 (no ground truth anomalies)")

if precision + recall > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
else:
    f1 = 0
    print("F1-Score: 0.0000 (0.00%)")

accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed report
print(f"\n--- Detailed Report ---")
if len(truep) > 0:
    print(f"Correctly detected: {truep}")

missed = np.setdiff1d(ground_truth, truep)
if len(missed) > 0:
    print(f"Missed anomalies: {missed}")

false_detections = np.setdiff1d(anomalies_unique, ground_truth)
if len(false_detections) > 0:
    print(f"False positives (first 10): {false_detections.flatten()[:10]}")

print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)