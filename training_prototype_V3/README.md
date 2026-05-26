# AQI Autoencoder + LSTM Training

This script trains an air quality prediction model using a Time-Variant Autoencoder and an LSTM.

The autoencoder compresses selected sensor/weather features into a lower-dimensional latent representation. These latent features are then combined with lag features, cyclic time features, and binary features before being passed into the LSTM to predict future AQI values.

## How to Run

Run the script normally from the terminal:

```bash
python3 fileName.py
```

Replace `fileName.py` with the actual name of the Python file.

Example:

```bash
python3 train_aqi_lstm.py
```

## Required Imports / Libraries

This script uses the following Python libraries:

```python
logging
os
time
pickle
warnings
json
dataclasses
typing

numpy
pandas
matplotlib
torch
sklearn
```

Install the main external libraries with:

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

## Dataset Path

The dataset path is controlled in the `Config` class:

```python
all_data_path: str = "../datasets/all_features_training.csv"
```

Change this path if the dataset is stored somewhere else.

## Main Parameters to Adjust

Most training settings are located inside the `Config` class.

### Data Splitting

```python
val_size: float = 0.20
random_state: int = 10
```

- `val_size`: percentage of data used for validation.
- `random_state`: used for reproducibility.

### Sequence Settings

```python
lookback: int = 24
horizon: int = 1
```

- `lookback`: number of previous time steps used as input.
- `horizon`: how many hours ahead the model predicts.

For example, `lookback = 24` means the LSTM uses the previous 24 hours of data.

### Autoencoder Settings

```python
latent_dim: int = 8
ae_epochs: int = 50
ae_lr: float = 1e-3
ae_batch_size: int = 1048
```

- `latent_dim`: size of the compressed feature representation.
- `ae_epochs`: number of training epochs for the autoencoder.
- `ae_lr`: autoencoder learning rate.
- `ae_batch_size`: batch size used while training the autoencoder.

Increasing `latent_dim` keeps more information but may reduce compression.

### LSTM Settings

```python
lstm_hidden: int = 96
lstm_layers: int = 1
lstm_dropout: float = 1e-4
lstm_epochs: int = 50
lstm_lr: float = .00099
patience: int = 5
lstm_batch_size = 4096
```

- `lstm_hidden`: number of hidden units in the LSTM.
- `lstm_layers`: number of LSTM layers.
- `lstm_dropout`: dropout used between LSTM layers.
- `lstm_epochs`: maximum number of LSTM training epochs.
- `lstm_lr`: LSTM learning rate.
- `patience`: early stopping patience.
- `lstm_batch_size`: batch size used for LSTM training.

### Dropped Columns

Columns that are not needed are removed here:

```python
cols_to_drop: List[str] = field(default_factory=lambda: [
    'wind_speed_100m', 'month', 'day', 'hour',
    'day_of_week', 'day_of_year', 'month_sin', 'month_cos',
])
```

Add or remove columns from this list depending on which features should be excluded.

## Output Files

Trained models and results are saved in:

```python
save_dir: str = "saved_models"
```

The script saves:

```text
saved_models/
├── lstm_ae16.pt
├── ae_16.pt
├── scalers.pkl
└── results_summary.csv
```

- `lstm_ae16.pt`: saved LSTM model, configuration, metrics, and loss curves.
- `ae_16.pt`: saved autoencoder weights.
- `scalers.pkl`: saved scaler object.
- `results_summary.csv`: RMSE, MAE, and R² results.

## Code Structure

The code is organized into specific classes and methods for readability:

- `Config`: stores file paths and training parameters.
- `AQI_LSTM`: defines the LSTM model.
- `TimeVariantAutoencoder`: defines the autoencoder model.
- `AIQTrainingPipeline`: loads, cleans, splits, scales, and prepares the data.
- `AEReducer`: trains the autoencoder and creates latent features.
- `LSTMTrainer`: trains, predicts, and evaluates the LSTM.
- `EarlyStopping`: stops training when validation loss stops improving.
- `save_artifacts`: saves the trained models, scaler, and results.
- `run`: controls the full training pipeline.

Comments are included throughout the script to explain how the code works, especially for feature preparation, scaling, sequence creation, and model training.

## Training Flow

The script follows this general process:

1. Load the dataset.
2. Remove missing values.
3. Drop unneeded columns.
4. Identify feature types:
   - cyclic features
   - binary features
   - lag features
   - regular sensor/weather features
5. Split data by ZIP code while preserving time order.
6. Scale selected features.
7. Train the autoencoder.
8. Extract latent features from the autoencoder.
9. Combine latent, lag, cyclic, and binary features.
10. Create LSTM time sequences.
11. Train the LSTM.
12. Evaluate the model.
13. Save the trained models and results.

## Notes

The model automatically uses CUDA if a GPU is available:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Otherwise, it runs on CPU.
