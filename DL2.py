import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

# 1. Generate synthetic data
np.random.seed(42)

# Normal transactions (1000 samples, 20 features)
normal_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 20))

# Fraudulent transactions (50 samples, different distribution)
fraud_data = np.random.uniform(low=-6, high=6, size=(50, 20))

# Combine for testing
X_test = np.concatenate([normal_data[:200], fraud_data], axis=0)  # 200 normal + 50 fraud
y_test = np.array([0]*200 + [1]*50)  # Labels (0 = normal, 1 = fraud)

# 2. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(normal_data)
X_test_scaled = scaler.transform(X_test)

# 3. Split training/validation
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 4. Build Autoencoder
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 5. Train Autoencoder
autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, X_val),
    verbose=0
)

# 6. Evaluate on test data (including frauds)
X_test_reconstructed = autoencoder.predict(X_test_scaled)
reconstruction_error = np.mean(np.power(X_test_scaled - X_test_reconstructed, 2), axis=1)

# 7. Set threshold
threshold = np.percentile(reconstruction_error, 95)
print("Reconstruction error threshold:", threshold)

# 8. Predict anomalies
predicted_labels = (reconstruction_error > threshold).astype(int)

# 9. Show results
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted_labels))
print(classification_report(y_test, predicted_labels))

# 10. Optional: Plot histogram of errors
plt.hist(reconstruction_error[y_test == 0], bins=50, alpha=0.6, label='Normal')
plt.hist(reconstruction_error[y_test == 1], bins=50, alpha=0.6, label='Fraud')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.title("Reconstruction Error")
plt.xlabel("Error")
plt.ylabel("Count")
plt.show()
