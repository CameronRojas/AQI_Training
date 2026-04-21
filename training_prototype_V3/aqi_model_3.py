# Use average of the mean for each feature to get gaussian distribution for the autoencoder, which is what it learns best. This is a common practice to improve AE performance when features have different scales or distributions.

import logging
import os
import time
import pickle
import warnings
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    # filename='autoencoder_training.log',
                    # filemode='w',
                    format='%(levelname)s: %(message)s',
                    force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f'Logger name: {logger.name} and level: {logger.level}')

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    all_data_path: str = "/content/all_features_all_data.csv"

    # Split/scaling choices
    val_size: float = 0.20
    random_state: int = 10
    scalers = [MinMaxScaler(), StandardScaler()]
    # Sequence settings for LSTM
    lookback: int = 24 # The number of previous sequences to "lookback" on
    horizon: int = 1 # The number of hour(s) to predict ahead

    # AE settings
    latent_dim: int = 8
    ae_layers = [[16, 4], [8]]
    ae_hidden_dims: List[int] = field(default_factory=list)  # single bottleneck layer only
    ae_epochs: int = 50
    ae_lr: float = 1e-3
    ae_batch_size: int = 1048

    # LSTM settings
    lstm_hidden: int = 96
    lstm_layers: int = 1
    lstm_dropout: float = 1e-4
    lstm_epochs: int = 50
    lstm_lr: float = .00099
    patience: int = 5
    lstm_batch_size = 4096

    seed: int = 42
    show_plots: bool = True
    save_dir: str = "saved_models"

###########################################################################
    # THESE COLUMNS ARE IRRELEVANT AND ARE DROPPED BECAUSE NOT NEEDED
    cols_to_drop: List[str] = field(default_factory=lambda: [
        'wind_speed_100m', 'month', 'day', 'hour',
        'day_of_week', 'day_of_year', 'month_sin', 'month_cos']
        )
###########################################################################
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Models
# ============================================================

class AQI_LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TimeVariantAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, scaler=StandardScaler()):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(),
        )
        if isinstance(scaler, MinMaxScaler):
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        recon = self.decoder(encoded)
        return recon, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

# ============================================================
# Data pipeline
# ============================================================

class AIQTrainingPipeline:
    """Prepare aiq_training_cam using the notebook's feature/scaling logic."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dataset_df = self.load_data(cfg.all_data_path)

        if self.dataset_df is None:
            raise ValueError("Failed to load data. Please check the file path and contents.")
        self.feature_cols: List[str] = []
        self.cyclic_cols = None
        self.binary_cols = None
        self.lag_cols = None
        self.target_col = "us_aqi"

        self.remove_nan()
        self.remove_uninformative_features()
        self.identify_features()

        self.train_df, self.val_df = self.split_data()

    def load_data(self, dataset_path):
        dataset = pd.read_csv(dataset_path)

        if dataset.empty:
            logger.warning("The loaded DataFrame is empty. Please check the file path and contents.")
            return None
        else:
            logger.info(f"Data loaded successfully with shape: {dataset.shape}")
            logger.info(f'DataFrame columns: {dataset.columns.tolist()}')

        return dataset

    def remove_nan(self):
        initial_shape = self.dataset_df.shape
        self.dataset_df.dropna(inplace=True)
        logger.info(f"Removed NaN values. Shape changed from {initial_shape} to {self.dataset_df.shape}")

    def remove_uninformative_features(self):
        self.dataset_df.drop(columns=self.cfg.cols_to_drop, inplace=True, errors='ignore')
        logger.info(f"Removed uninformative features: {self.cfg.cols_to_drop}")

    def identify_cyclic_features(self):
        cyclic_postfixes = ['_sin', '_cos']
        self.cyclic_cols = [col for col in self.dataset_df.columns if any(col.endswith(postfix) for postfix in cyclic_postfixes)]
        logger.info(f"Identified cyclic features: {self.cyclic_cols}")

    def identify_binary_features(self):
        self.binary_cols = [col for col in self.dataset_df.columns if self.dataset_df[col].nunique() == 2 and col not in self.cyclic_cols]
        logger.info(f"Identified binary features: {self.binary_cols}")

    def identify_lag_features(self):
        lag_prefixes = [
            'us_aqi_past_',
            'pm2_5_past_',
            'ozone_past_',
            'wind_speed_100m_past_',
            'wind_direction_100m_sin_past_',
            'wind_direction_100m_cos_past_',
        ]
        self.lag_cols = [col for col in self.dataset_df.columns if any(col.startswith(prefix) for prefix in lag_prefixes)]
        logger.info(f"Identified lag features: {self.lag_cols}")

    def identify_feature_cols(self):
        remove_cols = ['zip', 'time', 'us_aqi'] + self.cyclic_cols + self.binary_cols + self.lag_cols
        self.feature_cols = [col for col in self.dataset_df.columns if col not in remove_cols]
        logger.info(f"Feature columns: {self.feature_cols}")
        logger.info(f"Number of feature columns: {len(self.feature_cols)}")
    
    def identify_features(self):
        self.identify_cyclic_features()
        self.identify_binary_features()
        self.identify_lag_features()
        self.identify_feature_cols()

    def split_data(self):
        train_parts = []
        val_parts = []

        for zip_code, group in self.dataset_df.groupby("zip"):
            group = group.sort_values("time").reset_index(drop=True)
            split_idx_train = int(len(group) * (1 - self.cfg.val_size))

            train_parts.append(group.iloc[:split_idx_train].copy())
            val_parts.append(group.iloc[split_idx_train:].copy())

        train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
        val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)

        return train_df, val_df
    
    def scale_features(self, unscaled_train_df: pd.DataFrame, unscaled_val_df: pd.DataFrame, scaler) -> Tuple[np.ndarray, np.ndarray]:
        scaled_train = scaler.fit_transform(unscaled_train_df[self.feature_cols])
        scaled_val = scaler.transform(unscaled_val_df[self.feature_cols])

        return scaled_train, scaled_val

    @property
    def input_dim(self) -> int:
        return len(self.feature_cols)

    # Create 24 hour window sequences for LSTM training, grouped by zip code to maintain temporal integrity. Each sequence includes the past `lookback` hours of features
    def make_lstm_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        lookback, horizon = self.cfg.lookback, self.cfg.horizon
        Xs, ys = [], []

        for zip_code, group in df.groupby("zip"):
            group = group.sort_values("time").reset_index(drop=True)

            X = group.drop(columns=[self.target_col, 'zip', 'time']).to_numpy(dtype=np.float32)
            y = group[self.target_col].to_numpy(dtype=np.float32)

            if len(group) < lookback + horizon:
                continue
            
            for i in range(len(group) - lookback - horizon + 1):
                Xs.append(X[i:i+lookback]) # past `lookback` hours of features
                ys.append(y[i+lookback+horizon-1]) # target is the AQI at the end of the horizon

        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
# ============================================================
# AE reducer
# ============================================================

class AEReducer:
    def __init__(self, cfg: Config, input_dim: int, scaler=StandardScaler()):
        self.cfg = cfg
        self.model = TimeVariantAutoencoder(input_dim=input_dim, latent_dim=cfg.latent_dim, scaler=scaler)
        self.train_history: List[float] = []

    def fit(self, X_train: np.ndarray):
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
            batch_size=self.cfg.ae_batch_size,
            shuffle=True,
            drop_last=False,
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.ae_lr, weight_decay=1e-5)
        self.model.to(self.cfg.device).train()
        self.train_history = []

        for epoch in range(self.cfg.ae_epochs):
            total = 0.0
            t0 = time.time()
            for (batch,) in loader:
                batch = batch.to(self.cfg.device)
                optimizer.zero_grad()
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total += loss.item()

            avg = total / len(loader)
            self.train_history.append(avg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    [AE] Epoch {epoch+1:3d}/{self.cfg.ae_epochs}  loss={avg:.6f}  ({time.time()-t0:.1f}s)")

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.cfg.device)
            z = self.model.encode(X_t)
        return z.cpu().numpy()

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

# ============================================================
# LSTM trainer
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None

    def step(self, model: nn.Module, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class LSTMTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _loader(self, X, y, shuffle: bool):
        return DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).unsqueeze(1),
            ),
            batch_size=self.cfg.lstm_batch_size,
            shuffle=shuffle,
        )

    def train(self, model, X_train, y_train, X_val, y_val, name="LSTM"):
        train_dl = self._loader(X_train, y_train, True)
        val_dl = self._loader(X_val, y_val, False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.lstm_lr, weight_decay=1e-5)
        model.to(self.cfg.device)
        earlystopper = EarlyStopping(patience=self.cfg.patience)

        train_hist, val_hist = [], []

        for ep in range(self.cfg.lstm_epochs):
            model.train()

            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)

                pred = model(xb) # forward pass

                optimizer.zero_grad()
                loss = criterion(pred, yb)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients that exceed norm of 1.0, which is [gradient * (1 / L2 norm of gradients)]
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dl)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.cfg.device), yb.to(self.cfg.device)
                    val_loss += criterion(model(xb), yb).item()
            val_loss /= len(val_dl)

            train_hist.append(train_loss)
            val_hist.append(val_loss)

            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"    [{name}] Epoch {ep+1:3d}/{self.cfg.lstm_epochs}  train={train_loss:.6f}  test={val_loss:.6f}")

            if(earlystopper.step(model, val_loss)):
                print(f"    [{name}] Early stopping at epoch {ep+1}")
                break

        if earlystopper.best_state is not None:
            model.load_state_dict(earlystopper.best_state)
        return train_hist, val_hist

    @torch.no_grad()
    def predict(self, model, X):
        model.eval().to(self.cfg.device)
        preds = []
        for i in range(0, len(X), self.cfg.lstm_batch_size):
            batch = torch.tensor(X[i:i+self.cfg.lstm_batch_size], dtype=torch.float32).to(self.cfg.device)
            preds.append(model(batch).cpu().numpy())
        return np.concatenate(preds).flatten()

    def evaluate(self, y_true, y_pred, label=""):
        yt = y_true.reshape(-1) # ensure 1D for metric calculations
        yp = y_pred.reshape(-1)

        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(1 - np.sum((yt - yp) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-10))
        print(f"    [{label}] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
        return {"label": label, "rmse": rmse, "mae": mae, "r2": r2, "y_true": yt, "y_pred": yp}

# ============================================================
# Save helpers
# ============================================================

def save_artifacts(save_dir: str, model: AQI_LSTM, ae_reducer: AEReducer,
                   pipeline: AIQTrainingPipeline, cfg: Config, metrics: Dict,
                   loss_curves: Tuple[List[float], List[float]],
                   lstm_input_dim: int):
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": lstm_input_dim,
            "hidden_size": cfg.lstm_hidden,
            "num_layers": cfg.lstm_layers,
            "lookback": cfg.lookback,
            "latent_dim": cfg.latent_dim,
            "feature_cols": pipeline.feature_cols,
            "target_col": pipeline.target_col,
        },
        "metrics": metrics,
        "train_loss": loss_curves[0],
        "val_loss": loss_curves[1],
    }, os.path.join(save_dir, "lstm_ae16.pt"))

    torch.save(ae_reducer.model.state_dict(), os.path.join(save_dir, "ae_16.pt"))

    with open(os.path.join(save_dir, "scalers.pkl"), "wb") as f:
        pickle.dump({
            "scaler": cfg.scalers[1], # StandardScaler for LSTM input
        }, f)

    pd.DataFrame([{
        "Model": metrics["label"],
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R2": metrics["r2"],
    }]).to_csv(os.path.join(save_dir, "results_summary.csv"), index=False)

# ============================================================
# Run
# ============================================================

def run(cfg: Config):
    seed_everything(cfg.seed)
    print(f"Device: {cfg.device}\n")

    print("=" * 60)
    print("  DATA PIPELINE")
    print("=" * 60)
    pipeline = AIQTrainingPipeline(cfg)

    print("\n" + "=" * 60)
    print("  TRAIN TIME-VARIANT AUTOENCODER")
    print("=" * 60)
    # First scale features for the AE, not including cyclic/binary/lag features
    X_train_ae, X_val_ae = pipeline.scale_features(pipeline.train_df[pipeline.feature_cols], pipeline.val_df[pipeline.feature_cols], scaler=StandardScaler())

    # Train AE and get latent features
    ae = AEReducer(cfg, input_dim=X_train_ae.shape[1]) # Defaults to StandardScaler in AEReducer, which is what we want for the AE
    X_train_latent = ae.fit_transform(X_train_ae)
    X_val_latent = ae.transform(X_val_ae)
    latent_cols = [f"z{i}" for i in range(cfg.latent_dim)] # latent_dim is 8, so this will create z0, z1, ..., z7

    # Need to scale the features for the LSTM as well, but we want to preserve the original scaler before the AE transformation
    base_columns_to_scale = pipeline.train_df.drop(columns=['zip', 'time'] + pipeline.cyclic_cols + pipeline.binary_cols + pipeline.lag_cols).columns

    scaler = cfg.scalers[1] # StandardScaler for LSTM input
    scaler.fit(pipeline.train_df[base_columns_to_scale]) # dont scale cyclic, binary, or lag features(yet)

    col_idx = {col: idx for idx, col in enumerate(base_columns_to_scale)}

    aqi_lag_cols= [col for col in pipeline.lag_cols if col.startswith('us_aqi_past_')]
    print(f"Number of AQI lag features: {len(aqi_lag_cols)}")

    pm2_5_lag_cols = [col for col in pipeline.lag_cols if col.startswith('pm2_5_past_')]
    ozone_lag_cols = [col for col in pipeline.lag_cols if col.startswith('ozone_past_')]

    def scale_lag_group(train_df, val_df, lag_cols, base_col, scaler, col_idx):
        idx = col_idx[base_col]
        train_scaled = (train_df[lag_cols].to_numpy() - scaler.mean_[idx]) / scaler.scale_[idx]
        val_scaled = (val_df[lag_cols].to_numpy() - scaler.mean_[idx]) / scaler.scale_[idx]
        return train_scaled, val_scaled

    aqi_lags_scaled_train, aqi_lags_scaled_val = scale_lag_group(pipeline.train_df, pipeline.val_df, aqi_lag_cols, 'us_aqi', scaler, col_idx)

    # Changed this to only use the first 8 lag features
    pm2_5_lags_scaled_train, pm2_5_lags_scaled_val = scale_lag_group(pipeline.train_df, pipeline.val_df, pm2_5_lag_cols[:8], 'pm2_5', scaler, col_idx)
    ozone_lags_scaled_train, ozone_lags_scaled_val = scale_lag_group(pipeline.train_df, pipeline.val_df, ozone_lag_cols[:8], 'ozone', scaler, col_idx)

    # Combine latent features with scaled lag features for LSTM input (binary/cyclic features not concatenated yet)
    latent_lag_scaled_train = np.concatenate([X_train_latent, aqi_lags_scaled_train, pm2_5_lags_scaled_train, ozone_lags_scaled_train], axis=1)
    latent_lag_scaled_val = np.concatenate([X_val_latent, aqi_lags_scaled_val, pm2_5_lags_scaled_val, ozone_lags_scaled_val], axis=1)

    latent_lag_df = pd.DataFrame(latent_lag_scaled_train, columns=latent_cols + aqi_lag_cols + pm2_5_lag_cols[:8] + ozone_lag_cols[:8])
    latent_val_lag_df = pd.DataFrame(latent_lag_scaled_val, columns=latent_cols + aqi_lag_cols + pm2_5_lag_cols[:8] + ozone_lag_cols[:8])
    print(f"Latent+Lag train shape: {latent_lag_df.shape}")

    # Combine latent features with original binary, scaled lag features, and zip/time for LSTM training.
    lstm_train_df = pd.concat([latent_lag_df, pipeline.train_df[pipeline.binary_cols + pipeline.cyclic_cols + ['zip', 'time', 'us_aqi']]], axis=1)
    lstm_val_df = pd.concat([latent_val_lag_df, pipeline.val_df[pipeline.binary_cols + pipeline.cyclic_cols + ['zip', 'time', 'us_aqi']]], axis=1)
    print(f"LSTM train shape: {lstm_train_df.shape}")
    
    # Create sequences for LSTM training
    X_train_lstm, y_train_lstm = pipeline.make_lstm_sequences(lstm_train_df)
    X_val_lstm, y_val_lstm = pipeline.make_lstm_sequences(lstm_val_df)

    print(f"Latent train seq: {X_train_lstm.shape}")
    print(f"Latent val  seq: {X_val_lstm.shape}")

    print("\n" + "=" * 60)
    print("  TRAIN LSTM")
    print("=" * 60)
    model = AQI_LSTM(
        input_dim=X_train_lstm.shape[2],
        hidden_size=cfg.lstm_hidden,
        num_layers=cfg.lstm_layers,
        dropout=cfg.lstm_dropout
    )
    trainer = LSTMTrainer(cfg)
    loss_curves = trainer.train(
        model,
        X_train_lstm,
        y_train_lstm,
        X_val_lstm,
        y_val_lstm,
        name="LSTM+AE16",
    )

    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    preds = trainer.predict(model, X_val_lstm)
    metrics = trainer.evaluate(y_val_lstm, preds, label="LSTM+AE16")

    save_artifacts(cfg.save_dir, model, ae, pipeline, cfg, metrics, loss_curves, X_train_lstm.shape[2])

    return {
        "pipeline": pipeline,
        "ae": ae,
        "model": model,
        "metrics": metrics,
        "loss_curves": loss_curves,
        "X_train_lstm": X_train_lstm,
        "y_train_lstm": y_train_lstm,
        "X_val_lstm": X_val_lstm,
        "y_val_lstm": y_val_lstm,
    }


if __name__ == "__main__":
    cfg = Config()
    results = run(cfg)