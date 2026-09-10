import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Number of samples, time steps and features
samples = 400
window = 6
features = 4

# Four cyclone-related features
# Temperature, Pressure, U-Wind, V-Wind
X = np.random.rand(samples, window, features)

# Target: cyclone intensity / wind speed
y = np.random.uniform(15, 65, samples)

# Normalize input
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(
    X.reshape(-1, features)
).reshape(X.shape)

# Standardize target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(
    y.reshape(-1, 1)
)

print("Data preparation completed")
print("Input shape:", X_scaled.shape)
print("Target shape:", y_scaled.shape)