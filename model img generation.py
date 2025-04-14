from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

# Define model
model = Sequential([
    BatchNormalization(input_shape=(70, 20)),  # Adjust input shape as needed
    LSTM(128, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)),
    LSTM(64, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dense(32, activation='gelu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(16, activation='gelu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1)  # output_dim = 1
])

# Save model plot
plot_model(model, to_file="lstm_model_plot.png", show_shapes=True, show_layer_names=True)
