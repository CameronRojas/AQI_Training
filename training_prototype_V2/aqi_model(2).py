# Use average of the mean for each feature to get gaussian distribution for the autoencoder, which is what it learns best. This is a common practice to improve AE performance when features have different scales or distributions.

import os
import time
import pickle
import warnings
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

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    all_data_path: str = "/content/all_features_all_data.csv"
    target_col: str = "us_aqi"

    # Split/scaling choices
    val_size: float = 0.20
    random_state: int = 10

    # Sequence settings for LSTM
    lookback: int = 24 # The number of previous sequences to "lookback" on
    horizon: int = 1 # The number of hour(s) to predict ahead

    # AE settings
    latent_dim: int = 22
    ae_hidden_dims: List[int] = field(default_factory=list)  # single bottleneck layer only
    ae_epochs: int = 50
    ae_lr: float = 1e-3
    ae_batch_size: int = 1048

    # LSTM settings
    lstm_hidden: int = 512
    lstm_layers: int = 3
    lstm_dropout: float = 1e-3
    lstm_epochs: int = 100
    lstm_lr: float = .00099
    patience: int = 5
    lstm_batch_size = 4096

    seed: int = 42
    show_plots: bool = True
    save_dir: str = "saved_models"

    # Exact drop pattern from notebook feature selection
    cols_to_drop: List[str] = field(default_factory=lambda: [
        'latitude_y', 'longitude_y', 'city', 'state', 'month',
        'day', 'hour', 'day_of_week', 'day_of_year', 'us_aqi'
    ])

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
    def __init__(self, input_dim: int, latent_dim: int = 22):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
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
        self.feature_cols: List[str] = []
        self.ae_scaler = MinMaxScaler() # Use MinMaxScaler for autoencoder, StandardScaler for LSTM sequences (after AE transformation)
        self.lstm_scaler = StandardScaler()

        self.train_df = None
        self.val_df_df = None

        self.X_train = self.X_val = None
        self.y_train = self.y_val = None

        self.X_train_seq = self.X_val_seq = None
        self.y_train_seq = self.y_val_seq = None

        self.lag_data_cols = None

    def prepare(self) -> "AIQTrainingPipeline":
        all_data_df, self.lag_data_cols = self._load_and_engineer()
        self.train_df, self.val_df = self._split_scale(all_data_df)

        # AE gets raw features, LSTM gets AE-reduced features + past AQI sequences
        self.X_train = self.train_df[self.feature_cols].values.astype(np.float32)
        self.X_val = self.val_df[self.feature_cols].values.astype(np.float32)

        self.y_train = self.train_df[self.cfg.target_col].values.reshape(-1, 1).astype(np.float32)
        self.y_val = self.val_df[self.cfg.target_col].values.reshape(-1, 1).astype(np.float32)

        # grouped sequences for LSTM
        self.X_train_seq, self.y_train_seq = self.make_sequences(self.train_df)
        self.X_val_seq, self.y_val_seq = self.make_sequences(self.val_df)

        print(f"Train rows: {self.train_df.shape}, Validation rows: {self.val_df.shape}")
        print(f"Train seq: {self.X_train_seq.shape}, Validation seq: {self.X_val_seq.shape}")
        return self

    def _load_and_engineer(self) -> pd.DataFrame:
        all_data_df = pd.read_csv(self.cfg.all_data_path)

        all_data_df.dropna(inplace=True) # Drop rows with missing values to ensure clean training data for the autoencoder and LSTM

        lag_prefixes = [
            'us_aqi_past_',
            'pm2_5_past_',
            'ozone_past_',
            'wind_speed_10m_past_',
            'wind_direction_10m_sin_past_',
            'wind_direction_10m_cos_past_',
        ]

        lagged_features_to_remove = [
            c for c in all_data_df.columns
            if any(c.startswith(prefix) for prefix in lag_prefixes)
        ]
        
        # IMPORTANT: do NOT drop zip yet
        x = all_data_df.drop(columns=self.cfg.cols_to_drop)
        lag_features_removed_df = x.drop(columns=lagged_features_to_remove)

        # zip stays only for grouping, not as a model feature
        self.feature_cols = [c for c in lag_features_removed_df.columns if c != "zip" and c != 'time']
        self._df = all_data_df

        print(f"Loaded: {all_data_df.shape}")
        print(f"Target: {self.cfg.target_col}")
        print(f"Features ({len(self.feature_cols)}): {self.feature_cols}")
        return all_data_df, lagged_features_to_remove

    def _split_scale(self, all_data_df: pd.DataFrame):
        train_parts = []
        val_parts = []

        for zip_code, group in all_data_df.groupby("zip"):
            group = group.sort_values("time").reset_index(drop=True)
            split_idx_train = int(len(group) * (1 - self.cfg.val_size))

            train_parts.append(group.iloc[:split_idx_train].copy())
            val_parts.append(group.iloc[split_idx_train:].copy())

        train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
        val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)

        train_df[self.feature_cols] = self.ae_scaler.fit_transform(
            train_df[self.feature_cols].astype(np.float32)
        )
        val_df[self.feature_cols] = self.ae_scaler.transform(
            val_df[self.feature_cols].astype(np.float32)
        )

        return train_df, val_df

    @property
    def input_dim(self) -> int:
        return len(self.feature_cols)

    # Create 24 hour window sequences for LSTM training, grouped by zip code to maintain temporal integrity. Each sequence includes the past `lookback` hours of features
    def make_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        lookback, horizon = self.cfg.lookback, self.cfg.horizon
        Xs, ys = [], []

        for zip_code, group in df.groupby("zip"):
            group = group.sort_values("time").reset_index(drop=True)

            X = group[self.feature_cols].values.astype(np.float32)
            y = group[self.cfg.target_col].values.astype(np.float32)

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
    def __init__(self, cfg: Config, input_dim: int):
        self.cfg = cfg
        self.model = TimeVariantAutoencoder(input_dim=input_dim, latent_dim=cfg.latent_dim)
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
            "target_col": cfg.target_col,
        },
        "metrics": metrics,
        "train_loss": loss_curves[0],
        "val_loss": loss_curves[1],
    }, os.path.join(save_dir, "lstm_ae22.pt"))

    torch.save(ae_reducer.model.state_dict(), os.path.join(save_dir, "ae_22.pt"))

    with open(os.path.join(save_dir, "scalers.pkl"), "wb") as f:
        pickle.dump({
            "ae_scaler": pipeline.ae_scaler,
            "lstm_scaler" : pipeline.lstm_scaler
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
    pipeline = AIQTrainingPipeline(cfg).prepare()

    print("\n" + "=" * 60)
    print("  TRAIN TIME-VARIANT AUTOENCODER")
    print("=" * 60)
    ae = AEReducer(cfg, input_dim=pipeline.X_train.shape[1])
    X_train_latent = ae.fit_transform(pipeline.X_train)
    X_val_latent = ae.transform(pipeline.X_val)

    # X_train_latent = pipeline.lstm_scaler.fit_transform(X_train_latent) # Scale the latent features for LSTM training using StandardScaler
    # X_val_latent = pipeline.lstm_scaler.transform(X_val_latent)

    latent_cols = [f"z{i}" for i in range(cfg.latent_dim)] # latent_dim is 32, so this will create z0, z1, ..., z21

    train_latent_df = pipeline.train_df[["zip", "time", cfg.target_col]].copy() # Get the original train_df structure with zip, time, and target_col
    val_latent_df = pipeline.val_df[["zip", "time", cfg.target_col]].copy()

    train_latent_df[latent_cols] = X_train_latent # Add the latent features to the train dataframe
    val_latent_df[latent_cols] = X_val_latent

    train_latent_df[pipeline.lag_data_cols] = pipeline.train_df[pipeline.lag_data_cols]
    val_latent_df[pipeline.lag_data_cols] = pipeline.val_df[pipeline.lag_data_cols]

    original_feature_cols = pipeline.feature_cols.copy() # Save original feature columns to restore later
    pipeline.feature_cols = latent_cols + pipeline.lag_data_cols

    try:
        X_train_seq, y_train_seq = pipeline.make_sequences(train_latent_df) # Generate sequences using the latent features
        X_val_seq, y_val_seq = pipeline.make_sequences(val_latent_df)
    finally:
        pipeline.feature_cols = original_feature_cols

    #X_train_seq.shape is (num_samples, lookback, num_features + 1) where num_features is the number of latent features (22) and the +1 is for the lagged target variable included in the sequence

    print(f"Latent train seq: {X_train_seq.shape}")
    print(f"Latent val  seq: {X_val_seq.shape}")

    print("\n" + "=" * 60)
    print("  TRAIN LSTM")
    print("=" * 60)
    model = AQI_LSTM(
        input_dim=X_train_seq.shape[2],
        hidden_size=cfg.lstm_hidden,
        num_layers=cfg.lstm_layers,
        dropout=cfg.lstm_dropout,
    )
    trainer = LSTMTrainer(cfg)
    loss_curves = trainer.train(
        model,
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        name="LSTM+AE22"
    )

    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    preds = trainer.predict(model, X_val_seq)
    metrics = trainer.evaluate(y_val_seq, preds, label="LSTM+AE22")

    save_artifacts(cfg.save_dir, model, ae, pipeline, cfg, metrics, loss_curves, X_train_seq.shape[2])

    return {
        "pipeline": pipeline,
        "ae": ae,
        "model": model,
        "metrics": metrics,
        "loss_curves": loss_curves,
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
    }


if __name__ == "__main__":
    cfg = Config()
    results = run(cfg)