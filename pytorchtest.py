import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Load Excel file (replace 'data.xlsx' with your file name)
df = pd.read_excel('CleanDataSample.xlsx', sheet_name='Sheet1')  # Change sheet name if needed

# Display the first few rows
print(df.head())

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Example data (replace with real dataset)
X_data = [[1.0], [2.0], [3.0], [4.0]]
y_data = [[2.0], [4.0], [6.0], [8.0]]

dataset = CustomDataset(X_data, y_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
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
