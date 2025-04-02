import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
#import tables as tb


"""X_columns = [col for col in [
    "Desc", "Date", "Time", "C1 Temp", "C2 Temp", "C3 Temp",
    "Con1Inlet", "Con1Pressure", "Con2Pressure", "Con3Pressure",
    "Con2Inlet", "Con3Inlet", "C1Pressure", "C2Pressure", "C3Pressure",
    "Height", "BeltAdj", "Oil IN Temp", "Oil OUT Temp", "Oil Pressure",
    "WaterinTemp", "WaterFlowrate", "WGen1Temp", "WGen2Temp"
] if col in df.columns]  # Only include columns that exist in the DataFrame"""# Questions about cleaned data: Seems to have a repeat halfway at row 2450 at value no. 2447, and resets to no. 201

fullDataSet = []  # Initialize an empty set to store unique data arrays
folderPath = "./Tirehog Dummy Data"  # Adjust this path if necessary
#Get all Excel file names matching the pattern "ModifiedDataSample_*"
for file in os.listdir(folderPath):
    if file.startswith("ModifiedDataSample") and file.endswith(".xlsx"):
        # Reads Excel file
        rawData = pd.read_excel(("./Tirehog Dummy Data/" + file), "Sheet1")

        noLabelData = rawData.iloc[3:,1:] # was [2:,1:]

        # Convert DataFrame to numpy array
        fullData = noLabelData.to_numpy()
        fullDataSet.append(fullData)
               
#Print the length of the fullDataSet
print("fullDataSet Dims", len(fullDataSet))

#Define the RNN Model
class RNNModel(nn.Module):
    def init(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).init()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # Use Xavier uniform initialization for the weights

    def forward(self, x):
        # Initialize hidden state with Xavier initialization
         
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through RNN
        out,  = self.rnn(x, h0)

        #Decode the last hidden state to the output
        out = self.fc(out[:, -1, :])  # Only take the output of the last timestep
        return out

#Hyperparameters
input_size = len(X_columns)  # Number of input features
hidden_size = 64  # Can be tuned based on experimentation
num_layers = 2  # Number of stacked RNN layers
output_size = 1  # Since we are predicting 'Height'
learning_rate = 0.001  # Typical starting value for learning rate

#Initialize the model, loss function, and optimizer
model = RNNModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Example Training Loop
num_epochs = 10  # Can be adjusted as needed
for epoch in range(num_epochs):
    for features, targets in dataloader:
        # Reshape input to (batch_size, sequence_length, input_size)
        features = features.unsqueeze(1)  # Add sequence dimension

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")