import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
# Define the folder path
folder_path = "./Tirehog Dummy Data"  # Adjust this path if necessary

# Get all Excel file names matching the pattern "ModifiedDataSample_*"
file_names = [file for file in os.listdir(folder_path) if file.startswith("ModifiedDataSample_") and file.endswith(".xlsx")]
# Print the loaded file names
print("Files found:", file_names)
# Load Excel file
file_name = 'Test.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_name, sheet_name=sheet_name)

# Display the first few rows to verify the data
print(df.head())

# Strip spaces or unwanted characters from column names
df.columns = df.columns.str.strip()

# Define feature columns and the target column
X_columns = [col for col in [
    "Desc", "Date", "Time", "C1 Temp", "C2 Temp", "C3 Temp",
    "Con1Inlet", "Con1Pressure", "Con2Pressure", "Con3Pressure",
    "Con2Inlet", "Con3Inlet", "C1Pressure", "C2Pressure", "C3Pressure",
    "Height", "BeltAdj", "Oil IN Temp", "Oil OUT Temp", "Oil Pressure",
    "WaterinTemp", "WaterFlowrate", "WGen1Temp", "WGen2Temp"
] if col in df.columns]  # Only include columns that exist in the DataFrame

y_column = "Height"  # Target column

# Verify the columns in the DataFrame
print("Columns in the DataFrame:", df.columns.tolist())

# Ensure all data in the selected columns is numeric
df[X_columns] = df[X_columns].apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
df[y_column] = pd.to_numeric(df[y_column], errors='coerce')

# Handle missing values (NaN) by replacing them with a default value, e.g., 0
df.fillna(0, inplace=True)

# Extract features (X) and target (y)
X_data = df[X_columns].values
y_data = df[y_column].values.reshape(-1, 1)

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset and dataloader
dataset = CustomDataset(X_data, y_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) # Proccesses 2 shuffled samples together

# Example: Iterate through the dataloader
for batch_idx, (features, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print("Features:", features)
    print("Targets:", targets)
'''
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(1, 10)  # Input layer to hidden
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)  # Hidden to output
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = SimpleNN()

criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SimpleNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()
'''
