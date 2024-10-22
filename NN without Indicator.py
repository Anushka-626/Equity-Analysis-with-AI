import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# Load data
indices_data = pd.read_excel(r'C:\Users\krati\OneDrive\Desktop\Lectures\Equity Analysis with AI\daily_data.xlsx')

# Convert 'date' to datetime and sort
indices_data['date'] = pd.to_datetime(indices_data['date'])
indices_data.sort_values('date', inplace=True)

# Select only DE40 and the percentage difference columns
de40_data = indices_data[['date', 'DE0_perdiff']]

# Fill any missing values
de40_data.fillna(method='ffill', inplace=True)

# --- Sliding window function ---
def create_sliding_window(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Split the data based on date for train/test
train_data = de40_data[(de40_data['date'] >= '2024-06-11') & (de40_data['date'] <= '2024-07-15')]['DE0_perdiff'].values
test_data = de40_data[(de40_data['date'] >= '2024-07-16') & (de40_data['date'] <= '2024-10-31')]['DE0_perdiff'].values

# --- Fix Scaling Leakage ---
scaler = MinMaxScaler()

# Fit scaler on training data only
train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

# Apply the fitted scaler on test data
test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

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
        self.fc1 = nn.Linear(window_size, 64)  # Input size is window_size (flattened input)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Flatten the input to be of shape (batch_size, window_size)
        x = x.view(x.size(0), -1)  # Flatten the input (batch size, window_size)
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
y_test_full = scaler.inverse_transform(y_test_torch.numpy().reshape(-1, 1))
predictions = scaler.inverse_transform(predictions)

# Extract the corresponding date range for the predictions
date_range = de40_data[(de40_data['date'] >= '2024-07-16') & (de40_data['date'] <= '2024-10-31')]['date'].values[window_size:]

# Plot actual vs predicted percentage returns with date labels
plt.figure(figsize=(10, 6))
plt.plot(date_range, y_test_full, label='Actual DE40 Percentage Return (Diff)', linestyle='--', color='blue')
plt.plot(date_range, predictions.flatten(), label='Predicted DE40 Percentage Return (Diff)', color='orange')
plt.title("Actual vs Predicted Percentage Returns for DE40 using FNN")
plt.xlabel("Date")
plt.ylabel("Percentage Return (Diff)")
plt.xticks(rotation=45)  # Rotate x-axis labels to fit better
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()