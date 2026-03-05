import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')

class Config:
    DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
    
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 512
    EPOCHS = 120
    PATIENCE = 20
    LEARNING_RATE = 5e-4
    
    WINDOW_SIZE = 9
    HIDDEN_DIM = 192
    MAX_FUTURE_HORIZON = 94
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=13):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.SEED)

def height_to_feet(height_str):
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

def add_advanced_features(df):
    """Enhanced feature engineering"""
    print("Adding advanced features...")
    df = df.copy()
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    # Distance Rate Features
    if 'distance_to_ball' in df.columns:
        df['distance_to_ball_change'] = df.groupby(gcols)['distance_to_ball'].diff().fillna(0)
        df['distance_to_ball_accel'] = df.groupby(gcols)['distance_to_ball_change'].diff().fillna(0)
        df['time_to_intercept'] = (df['distance_to_ball'] / 
                                    (np.abs(df['distance_to_ball_change']) + 0.1)).clip(0, 10)
    
    # Target Alignment Features
    if 'ball_direction_x' in df.columns:
        df['velocity_alignment'] = (
            df['velocity_x'] * df['ball_direction_x'] +
            df['velocity_y'] * df['ball_direction_y']
        )
        df['velocity_perpendicular'] = (
            df['velocity_x'] * (-df['ball_direction_y']) +
            df['velocity_y'] * df['ball_direction_x']
        )
        if 'acceleration_x' in df.columns:
            df['accel_alignment'] = (
                df['acceleration_x'] * df['ball_direction_x'] +
                df['acceleration_y'] * df['ball_direction_y']
            )
    
    # Multi-Window Rolling
    for window in [3, 5, 10]:
        for col in ['velocity_x', 'velocity_y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_roll{window}'] = df.groupby(gcols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{col}_std{window}'] = df.groupby(gcols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                ).fillna(0)
    
    # Extended Lag Features
    for lag in [4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(gcols)[col].shift(lag).fillna(0)
    
    # Velocity Change Features
    if 'velocity_x' in df.columns:
        df['velocity_x_change'] = df.groupby(gcols)['velocity_x'].diff().fillna(0)
        df['velocity_y_change'] = df.groupby(gcols)['velocity_y'].diff().fillna(0)
        df['speed_change'] = df.groupby(gcols)['s'].diff().fillna(0)
        df['direction_change'] = df.groupby(gcols)['dir'].diff().fillna(0)
        df['direction_change'] = df['direction_change'].apply(
            lambda x: x if abs(x) < 180 else x - 360 * np.sign(x)
        )
    
    # Field Position Features
    df['dist_from_left'] = df['y']
    df['dist_from_right'] = 53.3 - df['y']
    df['dist_from_sideline'] = np.minimum(df['dist_from_left'], df['dist_from_right'])
    df['dist_from_endzone'] = np.minimum(df['x'], 120 - df['x'])
    
    # Role-Specific Features
    if 'is_receiver' in df.columns and 'velocity_alignment' in df.columns:
        df['receiver_optimality'] = df['is_receiver'] * df['velocity_alignment']
        df['receiver_deviation'] = df['is_receiver'] * np.abs(df.get('velocity_perpendicular', 0))
    if 'is_coverage' in df.columns and 'closing_speed' in df.columns:
        df['defender_closing_speed'] = df['is_coverage'] * df['closing_speed']
    
    # Time Features
    df['frames_elapsed'] = df.groupby(gcols).cumcount()
    df['normalized_time'] = df.groupby(gcols)['frames_elapsed'].transform(
        lambda x: x / (x.max() + 1)
    )
    
    return df

def prepare_combined_features(input_df, output_df=None, test_template=None, is_training=True, window_size=10):
    """COMBINED: Advanced features + enhanced preprocessing WITH LAST POSITIONS"""
    print(f"Preparing COMBINED sequences (window_size={window_size})...")
    
    input_df = input_df.copy()
    
    # BASIC FEATURES
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    
    # Enhanced motion features
    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    o_rad = np.deg2rad(input_df['o'].fillna(0))
    
    input_df['velocity_x'] = input_df['s'] * np.sin(dir_rad)
    input_df['velocity_y'] = input_df['s'] * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.sin(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.cos(dir_rad)
    input_df['orientation_x'] = np.sin(o_rad)
    input_df['orientation_y'] = np.cos(o_rad)
    
    # Enhanced roles
    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    input_df['is_rusher'] = (input_df['player_role'] == 'Pass Rusher').astype(int)
    
    # Field position (enhanced)
    input_df['field_x_norm'] = (input_df['x'] - Config.FIELD_X_MIN) / (Config.FIELD_X_MAX - Config.FIELD_X_MIN)
    input_df['field_y_norm'] = (input_df['y'] - Config.FIELD_Y_MIN) / (Config.FIELD_Y_MAX - Config.FIELD_Y_MIN)
    input_df['distance_to_sideline'] = np.minimum(input_df['y'], 53.3 - input_df['y'])
    input_df['distance_to_endzone'] = np.minimum(input_df['x'], 120 - input_df['x'])
    
    # Physics features
    mass_kg = input_df['player_weight'].fillna(200.0) / 2.20462
    input_df['momentum_x'] = input_df['velocity_x'] * mass_kg
    input_df['momentum_y'] = input_df['velocity_y'] * mass_kg
    input_df['kinetic_energy'] = 0.5 * mass_kg * (input_df['s'] ** 2)
    
    # Ball features
    if 'ball_land_x' in input_df.columns:
        ball_dx = input_df['ball_land_x'] - input_df['x']
        ball_dy = input_df['ball_land_y'] - input_df['y']
        input_df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        input_df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        input_df['ball_direction_x'] = ball_dx / (input_df['distance_to_ball'] + 1e-6)
        input_df['ball_direction_y'] = ball_dy / (input_df['distance_to_ball'] + 1e-6)
        input_df['closing_speed'] = (
            input_df['velocity_x'] * input_df['ball_direction_x'] +
            input_df['velocity_y'] * input_df['ball_direction_y']
        )
    
    # Sort for temporal features
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    # Enhanced temporal features
    for lag in [1, 2, 3, 5]:
        input_df[f'x_lag{lag}'] = input_df.groupby(gcols)['x'].shift(lag)
        input_df[f'y_lag{lag}'] = input_df.groupby(gcols)['y'].shift(lag)
        input_df[f'velocity_x_lag{lag}'] = input_df.groupby(gcols)['velocity_x'].shift(lag)
        input_df[f'velocity_y_lag{lag}'] = input_df.groupby(gcols)['velocity_y'].shift(lag)
        input_df[f's_lag{lag}'] = input_df.groupby(gcols)['s'].shift(lag)
    
    # Multiple EMA smoothing
    for alpha in [0.1, 0.3, 0.5]:
        input_df[f'velocity_x_ema_{alpha}'] = input_df.groupby(gcols)['velocity_x'].transform(
            lambda x: x.ewm(alpha=alpha, adjust=False).mean()
        )
        input_df[f'velocity_y_ema_{alpha}'] = input_df.groupby(gcols)['velocity_y'].transform(
            lambda x: x.ewm(alpha=alpha, adjust=False).mean()
        )
    
    # ADVANCED FEATURES
    input_df = add_advanced_features(input_df)
    
    # COMBINED FEATURE LIST
    feature_cols = [
        # Core tracking (8)
        'x', 'y', 's', 'a', 'o', 'dir', 'frame_id',
        'ball_land_x', 'ball_land_y',
        
        # Player attributes (2)
        'player_height_feet', 'player_weight',
        
        # Enhanced motion (7)
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
        'orientation_x', 'orientation_y',
        'kinetic_energy',
        
        # Roles (6)
        'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer', 'is_rusher',
        
        # Field position (6)
        'field_x_norm', 'field_y_norm', 
        'dist_from_sideline', 'dist_from_endzone',
        'distance_to_sideline', 'distance_to_endzone',
        
        # Ball interaction (5)
        'distance_to_ball', 'angle_to_ball', 'ball_direction_x', 'ball_direction_y', 'closing_speed',
        
        # Enhanced temporal (20)
        'x_lag1', 'y_lag1', 'velocity_x_lag1', 'velocity_y_lag1', 's_lag1',
        'x_lag2', 'y_lag2', 'velocity_x_lag2', 'velocity_y_lag2', 's_lag2',
        'x_lag3', 'y_lag3', 'velocity_x_lag3', 'velocity_y_lag3', 's_lag3',
        'x_lag5', 'y_lag5', 'velocity_x_lag5', 'velocity_y_lag5', 's_lag5',
        
        # Multiple EMAs (6)
        'velocity_x_ema_0.1', 'velocity_y_ema_0.1',
        'velocity_x_ema_0.3', 'velocity_y_ema_0.3', 
        'velocity_x_ema_0.5', 'velocity_y_ema_0.5',
        
        # Advanced features
        'distance_to_ball_change', 'distance_to_ball_accel', 'time_to_intercept',
        'velocity_alignment', 'velocity_perpendicular', 'accel_alignment',
        'velocity_x_change', 'velocity_y_change', 'speed_change', 'direction_change',
        'receiver_optimality', 'receiver_deviation', 'defender_closing_speed',
        'frames_elapsed', 'normalized_time',
        
        # Rolling features (selective)
        'velocity_x_roll5', 'velocity_y_roll5', 's_roll5', 'a_roll5',
        'velocity_x_std5', 'velocity_y_std5', 's_std5', 'a_std5',
    ]
    
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in input_df.columns]
    print(f"Using {len(feature_cols)} COMBINED features")
    
    # CREATE SEQUENCES
    input_df.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df.groupby(level=['game_id', 'play_id', 'nfl_id'])
    
    target_rows = output_df if is_training else test_template
    target_groups = target_rows[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, last_positions = [], [], [], [], [], []
    
    for _, row in tqdm(target_groups.iterrows(), total=len(target_groups)):
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        
        try:
            group_df = grouped.get_group(key) 
        except KeyError:
            continue
        
        input_window = group_df.tail(window_size)
        
        if len(input_window) < window_size:
            if is_training:
                continue
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)
        
        # Enhanced imputation
        input_window = input_window.fillna(method='ffill').fillna(method='bfill')
        input_window = input_window.fillna(group_df.mean(numeric_only=True))
        
        seq = input_window[feature_cols].values
        
        if np.isnan(seq).any():
            if is_training:
                continue
            seq = np.nan_to_num(seq, nan=0.0)
        
        sequences.append(seq)
        
        # Store last positions for metric calculation
        last_x = input_window.iloc[-1]['x']
        last_y = input_window.iloc[-1]['y']
        last_positions.append((last_x, last_y))
        
        if is_training:
            out_grp = output_df[
                (output_df['game_id']==row['game_id']) &
                (output_df['play_id']==row['play_id']) &
                (output_df['nfl_id']==row['nfl_id'])
            ].sort_values('frame_id')
            
            dx = out_grp['x'].values - last_x
            dy = out_grp['y'].values - last_y
            
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_frame_ids.append(out_grp['frame_id'].values)
        
        sequence_ids.append({
            'game_id': key[0],
            'play_id': key[1],
            'nfl_id': key[2],
            'frame_id': input_window.iloc[-1]['frame_id']
        })
    
    print(f"Created {len(sequences)} sequences")
    
    if is_training:
        return sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, feature_cols, last_positions
    return sequences, sequence_ids, feature_cols, last_positions


class EnhancedSeqModel(nn.Module):
    def __init__(self, input_dim, horizon):
        super().__init__()
        self.horizon = horizon
        
        self.gru = nn.GRU(input_dim, 192, num_layers=3, batch_first=True, dropout=0.2, bidirectional=False)
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GELU(),
        )
        
        self.pool_ln = nn.LayerNorm(192)
        self.pool_attn = nn.MultiheadAttention(192, num_heads=8, batch_first=True, dropout=0.1)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 192))
        
        self.head = nn.Sequential(
            nn.Linear(192 + 128, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, horizon * 2)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        h, _ = self.gru(x)
        
        h_conv = self.conv1d(h.transpose(1, 2)).transpose(1, 2)
        h_conv_pool = h_conv.mean(dim=1)
        
        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)
        h_norm = self.pool_ln(h)
        ctx, _ = self.pool_attn(q, h_norm, h_norm)
        ctx = ctx.squeeze(1)
        
        combined = torch.cat([ctx, h_conv_pool], dim=1)
        
        out = self.head(combined)
        out = out.view(B, 2, self.horizon)
        
        out = torch.cumsum(out, dim=2)
        
        return out[:, 0, :], out[:, 1, :]

# CORRECTED METRIC FUNCTIONS
def compute_rmse(pred_dx, pred_dy, target_dx, target_dy, mask):
    """Calculate RMSE - CORRECTED VERSION
    Official formula: sqrt(0.5 * (MSE_x + MSE_y))
    """
    squared_errors_x = ((pred_dx - target_dx)**2) * mask
    squared_errors_y = ((pred_dy - target_dy)**2) * mask
    
    # Sum of squared errors divided by sum of mask (number of valid predictions)
    mse_x = squared_errors_x.sum() / (mask.sum() + 1e-8)
    mse_y = squared_errors_y.sum() / (mask.sum() + 1e-8)
    
    # Competition formula: sqrt(0.5 * (MSE_x + MSE_y))
    combined_mse = 0.5 * (mse_x + mse_y)
    return torch.sqrt(combined_mse).item()

def calculate_oof_rmse(sequences, targets_dx, targets_dy, oof_predictions, last_positions):
    """Calculate overall OOF RMSE - CORRECTED VERSION
    Official formula: sqrt(0.5 * (MSE_x + MSE_y))
    """
    all_squared_errors_x = []
    all_squared_errors_y = []
    total_samples = 0
    
    for i in range(len(sequences)):
        target_dx = targets_dx[i]
        target_dy = targets_dy[i]
        pred_dx = oof_predictions[i, :len(target_dx), 0]
        pred_dy = oof_predictions[i, :len(target_dy), 1]
        last_x, last_y = last_positions[i]
        
        # Convert displacements to absolute positions
        pred_x = last_x + pred_dx
        pred_y = last_y + pred_dy
        target_x = last_x + target_dx
        target_y = last_y + target_dy
        
        # Calculate squared errors
        squared_errors_x = (pred_x - target_x) ** 2
        squared_errors_y = (pred_y - target_y) ** 2
        
        all_squared_errors_x.extend(squared_errors_x)
        all_squared_errors_y.extend(squared_errors_y)
        total_samples += len(target_dx)
    
    # Compute MSE separately for x and y
    mse_x = np.sum(all_squared_errors_x) / total_samples
    mse_y = np.sum(all_squared_errors_y) / total_samples
    
    # Apply competition formula
    oof_rmse = np.sqrt(0.5 * (mse_x + mse_y))
    
    return oof_rmse


class EnhancedTemporalLoss(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.05, velocity_weight=0.1):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
        self.velocity_weight = velocity_weight
        self.huber = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, pred_dx, pred_dy, target_dx, target_dy, mask):
        L = pred_dx.size(1)
        t = torch.arange(L, device=pred_dx.device).float()
        time_weights = torch.exp(-self.time_decay * t).view(1, L)
        
        # Position loss with time decay
        loss_dx = self.huber(pred_dx, target_dx) * time_weights
        loss_dy = self.huber(pred_dy, target_dy) * time_weights
        
        masked_loss_dx = (loss_dx * mask).sum() / (mask.sum() + 1e-8)
        masked_loss_dy = (loss_dy * mask).sum() / (mask.sum() + 1e-8)
        
        # Competition-style position loss: 0.5 * (loss_x + loss_y)
        position_loss = 0.5 * (masked_loss_dx + masked_loss_dy)
        
        # Optional velocity consistency
        if self.velocity_weight > 0:
            pred_velocity_x = torch.diff(pred_dx, dim=1, prepend=torch.zeros_like(pred_dx[:, :1]))
            pred_velocity_y = torch.diff(pred_dy, dim=1, prepend=torch.zeros_like(pred_dy[:, :1]))
            target_velocity_x = torch.diff(target_dx, dim=1, prepend=torch.zeros_like(target_dx[:, :1]))
            target_velocity_y = torch.diff(target_dy, dim=1, prepend=torch.zeros_like(target_dy[:, :1]))
            
            velocity_loss = (
                self.huber(pred_velocity_x, target_velocity_x).mean() +
                self.huber(pred_velocity_y, target_velocity_y).mean()
            ) * self.velocity_weight
            
            total_loss = position_loss + velocity_loss
        else:
            total_loss = position_loss
        
        return total_loss

def prepare_targets_enhanced(batch_dx, batch_dy, max_h):
    """Prepare targets with proper masking"""
    tensors_dx, tensors_dy, masks = [], [], []
    for dx_arr, dy_arr in zip(batch_dx, batch_dy):
        L = len(dx_arr)
        padded_dx = np.pad(dx_arr, (0, max_h - L), constant_values=0).astype(np.float32)
        padded_dy = np.pad(dy_arr, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        tensors_dx.append(torch.tensor(padded_dx))
        tensors_dy.append(torch.tensor(padded_dy))
        masks.append(torch.tensor(mask))
    return torch.stack(tensors_dx), torch.stack(tensors_dy), torch.stack(masks)

def train_model_combined(X_train, y_dx_train, y_dy_train, X_val, y_dx_val, y_dy_val, input_dim, horizon, config):
    device = config.DEVICE
    model = EnhancedSeqModel(input_dim, horizon).to(device)
    
    criterion = EnhancedTemporalLoss(delta=0.5, time_decay=0.05, velocity_weight=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LEARNING_RATE, 
        epochs=config.EPOCHS, steps_per_epoch=len(X_train)//config.BATCH_SIZE+1
    )
    
    train_batches = []
    for i in range(0, len(X_train), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by_dx, by_dy, bm = prepare_targets_enhanced(
            [y_dx_train[j] for j in range(i, end)],
            [y_dy_train[j] for j in range(i, end)], 
            horizon
        )
        train_batches.append((bx, by_dx, by_dy, bm))
    
    val_batches = []
    for i in range(0, len(X_val), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by_dx, by_dy, bm = prepare_targets_enhanced(
            [y_dx_val[j] for j in range(i, end)],
            [y_dy_val[j] for j in range(i, end)],
            horizon
        )
        val_batches.append((bx, by_dx, by_dy, bm))
    
    best_rmse, best_state, bad = float('inf'), None, 0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        
        for bx, by_dx, by_dy, bm in train_batches:
            bx, by_dx, by_dy, bm = bx.to(device), by_dx.to(device), by_dy.to(device), bm.to(device)
            pred_dx, pred_dy = model(bx)
            loss = criterion(pred_dx, pred_dy, by_dx, by_dy, bm)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses, val_rmses = [], []
        with torch.no_grad():
            for bx, by_dx, by_dy, bm in val_batches:
                bx, by_dx, by_dy, bm = bx.to(device), by_dx.to(device), by_dy.to(device), bm.to(device)
                pred_dx, pred_dy = model(bx)
                loss = criterion(pred_dx, pred_dy, by_dx, by_dy, bm)
                rmse = compute_rmse(pred_dx, pred_dy, by_dx, by_dy, bm)
                val_losses.append(loss.item())
                val_rmses.append(rmse)
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_rmse = np.mean(val_rmses)
        
        if epoch % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_rmse={val_rmse:.4f}, lr={lr:.2e}")
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_rmse


def main():
    config = Config()
    
    print("=" * 90)
    print("🚀 MAIN PIPELINE WITH CORRECTED METRIC 🚀".center(90))
    print("=" * 90)
    print("\n🔗 Key Corrections:\n")
    print("  ✅ Competition Metric: sqrt(0.5 * (MSE_x + MSE_y))")
    print("  ✅ Proper OOF Calculation with absolute positions")
    print("  ✅ Enhanced loss function aligned with competition")
    print("  ✅ Last position tracking for accurate metric calculation\n")
    print("=" * 90)
    
    # Load data
    print("\n[1/4] Loading data...")
    train_input_files = [config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)]
    train_output_files = [config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
    
    train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
    train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
    
    test_input = pd.read_csv(config.DATA_DIR / "test_input.csv")
    test_template = pd.read_csv(config.DATA_DIR / "test.csv")
    
    # Prepare combined sequences WITH LAST POSITIONS
    print("\n[2/4] Preparing COMBINED sequences with position tracking...")
    (sequences, targets_dx, targets_dy, targets_frame_ids, 
     sequence_ids, feature_cols, last_positions) = prepare_combined_features(
        train_input, train_output, is_training=True, window_size=config.WINDOW_SIZE
    )
    
    sequences = np.array(sequences, dtype=object)
    targets_dx = np.array(targets_dx, dtype=object)
    targets_dy = np.array(targets_dy, dtype=object)
    last_positions = np.array(last_positions)
    
    print(f"Feature dimension: {sequences[0].shape[-1]}")
    
    # Train with combined approach
    print("\n[3/4] Training COMBINED model...")
    groups = np.array([d['game_id'] for d in sequence_ids])
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    
    models, scalers, fold_rmses = [], [], []
    oof_predictions = np.zeros((len(sequences), config.MAX_FUTURE_HORIZON, 2))
    
    for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
        print(f"\nFold {fold}/{config.N_FOLDS}")
        
        X_tr = sequences[tr]
        X_va = sequences[va]
        
        scaler = StandardScaler()
        scaler.fit(np.vstack([s for s in X_tr]))
        
        X_tr_scaled = np.stack([scaler.transform(s) for s in X_tr])
        X_va_scaled = np.stack([scaler.transform(s) for s in X_va])
        
        model, val_rmse = train_model_combined(
            X_tr_scaled, targets_dx[tr], targets_dy[tr], 
            X_va_scaled, targets_dx[va], targets_dy[va],
            X_tr[0].shape[-1], config.MAX_FUTURE_HORIZON, config
        )
        
        # Store OOF predictions for validation set
        model.eval()
        with torch.no_grad():
            X_va_tensor = torch.tensor(X_va_scaled.astype(np.float32)).to(config.DEVICE)
            pred_dx, pred_dy = model(X_va_tensor)
            oof_predictions[va, :, 0] = pred_dx.cpu().numpy()
            oof_predictions[va, :, 1] = pred_dy.cpu().numpy()
        
        models.append(model)
        scalers.append(scaler)
        fold_rmses.append(val_rmse)
        
        print(f"Fold {fold} completed with val_RMSE: {val_rmse:.4f}")
    
    # Calculate overall OOF RMSE using CORRECTED metric
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    for fold, rmse in enumerate(fold_rmses, 1):
        print(f"Fold {fold} RMSE: {rmse:.4f}")
    
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    print(f"\nMean CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    # Calculate full OOF RMSE using CORRECTED metric with absolute positions
    oof_rmse = calculate_oof_rmse(sequences, targets_dx, targets_dy, oof_predictions, last_positions)
    print(f"Overall OOF RMSE: {oof_rmse:.4f}")
    print("="*80 + "\n")
    
    # Predict
    print("\n[4/4] Generating final predictions...")
    test_sequences, test_ids, _, test_last_positions = prepare_combined_features(
        test_input, test_template=test_template, is_training=False, window_size=config.WINDOW_SIZE
    )
    
    X_test = np.array(test_sequences, dtype=object)
    test_last_x = np.array([pos[0] for pos in test_last_positions])
    test_last_y = np.array([pos[1] for pos in test_last_positions])
    
    # Ensemble predictions
    all_dx, all_dy = [], []
    
    for model, sc in zip(models, scalers):
        X_scaled = np.stack([sc.transform(s) for s in X_test])
        X_tensor = torch.tensor(X_scaled.astype(np.float32)).to(config.DEVICE)
        
        model.eval()
        with torch.no_grad():
            dx, dy = model(X_tensor)
            all_dx.append(dx.cpu().numpy())
            all_dy.append(dy.cpu().numpy())
    
    ens_dx = np.mean(all_dx, axis=0)
    ens_dy = np.mean(all_dy, axis=0)
    
    # Create submission
    rows = []
    H = ens_dx.shape[1]
    
    for i, sid in enumerate(test_ids):
        fids = test_template[
            (test_template['game_id'] == sid['game_id']) &
            (test_template['play_id'] == sid['play_id']) &
            (test_template['nfl_id'] == sid['nfl_id'])
        ]['frame_id'].sort_values().tolist()
        
        for t, fid in enumerate(fids):
            tt = min(t, H - 1)
            px = np.clip(test_last_x[i] + ens_dx[i, tt], Config.FIELD_X_MIN, Config.FIELD_X_MAX)
            py = np.clip(test_last_y[i] + ens_dy[i, tt], Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
            
            rows.append({
                'id': f"{sid['game_id']}_{sid['play_id']}_{sid['nfl_id']}_{fid}",
                'x': float(px),
                'y': float(py)
            })
    
    submission = pd.DataFrame(rows)
    submission.to_csv("submission.csv", index=False)
    
    print("🏁 FINAL SUBMISSION SUMMARY".center(70))
    print("=" * 70)
    
    print(f"\n ✓ Submission saved with CORRECTED METRIC!")
    print(f"   ├─ Rows: {len(submission)}")
    print(f"   ├─ Features used: {len(feature_cols)}")
    print(f"   ├─ OOF RMSE: {oof_rmse:.4f}")
    print(f"   └─ Expected LB RMSE: {oof_rmse:.4f} (±0.01)")
    
    print("\n Metric Corrections Applied:")
    print(f"   • Competition formula: sqrt(0.5 * (MSE_x + MSE_y))")
    print(f"   • Proper absolute position calculation")
    print(f"   • Correct OOF RMSE computation")
    print(f"   • Enhanced loss alignment")
    
    print("=" * 70)
    
    return submission

if __name__ == "__main__":
    main()