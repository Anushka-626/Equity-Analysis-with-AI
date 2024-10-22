import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# SMAPE function definition
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))

# Load data
indices_data = pd.read_excel('daily_data.xlsx')

# Convert 'date' to datetime and sort
indices_data['date'] = pd.to_datetime(indices_data['date'])
indices_data.sort_values('date', inplace=True)

# Select all 7 indices' percentage difference columns
index_columns = ['DE40_perdiff', 'FR40_perdiff', 'NL25_perdiff', 'IT40_perdiff', 'SP35_perdiff', 'UK100_perdiff', 'EU50_perdiff']
indices_data = indices_data[['date'] + index_columns]

# Fill any missing values
indices_data.fillna(method='ffill', inplace=True)

# --- Sliding window function ---
def create_sliding_window(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Calculate the midpoint for a 50/50 split
split_date = '2024-07-20'

# Split the data into training and testing sets based on date (50/50 split)
train_data = indices_data[(indices_data['date'] >= '2024-06-11') & (indices_data['date'] <= split_date)][index_columns].values
test_data = indices_data[(indices_data['date'] > split_date) & (indices_data['date'] <= '2024-08-30')][index_columns].values

# --- Fix Scaling Leakage ---
scaler = MinMaxScaler()

# Fit scaler on training data only
train_data_scaled = scaler.fit_transform(train_data)

# Apply the fitted scaler on test data
test_data_scaled = scaler.transform(test_data)

# Define the sliding window size (e.g., 24 hours for a full day lookback)
window_size = 24

# Create sliding window for train and test
X_train, y_train = create_sliding_window(train_data_scaled, window_size)
X_test, y_test = create_sliding_window(test_data_scaled, window_size)

# Convert to torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# Define a simple feed-forward neural network (FNN)
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(window_size * 7, 64)  # Input size is window_size * 7 for all 7 indices
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 7)  # Output size is 7 (for predicting all 7 indices)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Flatten the input to be of shape (batch_size, window_size * 7)
        x = x.view(x.size(0), -1)  # Flatten the input (batch size, window_size * 7)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, define loss and optimizer
model = FNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Make predictions
model.eval()
predictions = model(X_test_torch).detach().numpy()

# Inverse transform the scaled predictions and actual values
y_test_full = scaler.inverse_transform(y_test_torch.numpy().reshape(-1, 7))
predictions = scaler.inverse_transform(predictions)

# Calculate errors (MSE, MAE, SMAPE) for each index
mse_errors = []
mae_errors = []
smape_errors = []

for i, col in enumerate(index_columns):
    mse = mean_squared_error(y_test_full[:, i], predictions[:, i])
    mae = mean_absolute_error(y_test_full[:, i], predictions[:, i])
    smape_value = smape(y_test_full[:, i], predictions[:, i])
    
    mse_errors.append(mse)
    mae_errors.append(mae)
    smape_errors.append(smape_value)
    
    print(f"{col} - MSE: {mse:.4f}, MAE: {mae:.4f}, SMAPE: {smape_value:.2f}%")

# Extract the corresponding date range for the predictions
date_range = indices_data[(indices_data['date'] > split_date) & (indices_data['date'] <= '2024-08-30')]['date'].values[window_size:]

# Plot actual vs predicted percentage returns for each index with date labels
plt.figure(figsize=(14, 10))
for i, col in enumerate(index_columns):
    plt.subplot(4, 2, i + 1)
    plt.plot(date_range, y_test_full[:, i], label=f'Actual {col}', linestyle='--', color='blue')
    plt.plot(date_range, predictions[:, i], label=f'Predicted {col}', color='orange')
    plt.title(f"Actual vs Predicted for {col}")
    plt.xlabel("Date")
    plt.ylabel("Percentage Return (Diff)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

plt.show()
