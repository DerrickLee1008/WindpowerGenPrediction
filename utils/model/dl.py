
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm(input_timesteps: int, input_dim: int) -> Sequential:
    """
    Simple 2-layer LSTM for regression on sequence inputs.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(input_timesteps, input_dim)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model
