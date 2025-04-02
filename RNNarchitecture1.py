import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

"""
Data Preparation
Read all Excel files in the folder "Tirehog Dummy Data" that match
the pattern "ModifiedDataSample_*.xlsx". Each file is assumed to have
data starting on row 3 and column 1, forming a matrix of shape (D, N).
We then transpose each matrix so that the sequence length N becomes
the first dimension (now shape becomes (N, D)) before converting to a tensor.
"""
fullDataSet = [] # List to store each sample tensor
folderPath = "./Tirehog Dummy Data" # Adjust this path as needed

for file in os.listdir(folderPath):
    if file.startswith("ModifiedDataSample") and file.endswith(".xlsx"):
        filePath = os.path.join(folderPath, file)
        rawData = pd.read_excel(filePath, "Sheet1")
        # Adjust indices as needed (here we skip the first 3 rows and the first column).
        noLabelData = rawData.iloc[3:, 1:]
        fullData = noLabelData.to_numpy()
        # fullData is shape (D, N); we need (N, D) for the RNN.
        fullDataT = fullData.T
        # Convert to a float tensor and add to the dataset list.
        fullDataSet.append(torch.tensor(fullDataT, dtype=torch.float))

print("Number of samples:", len(fullDataSet))

"""
Dataset and Collate Function for Variable-Length Sequences
Each sample is a tensor of shape (N, D) where N can vary.
The target here is defined simply as the true sequence length (N)
so that the network learns to “score” for long sequences in this example.
"""
class SequenceDataset(Dataset):
    def init(self, data_list):
    self.data_list = data_list


def __len__(self):
    return len(self.data_list)

def __getitem__(self, idx):
    # x has shape (N, D) and target is the sequence-length (N, as a float)
    x = self.data_list[idx]
    target = torch.tensor(x.size(0), dtype=torch.float)
    return x, target
def collate_fn(batch):
# Each sample in batch is a tuple (x, target)
sequences, targets = zip(*batch)
# Get the original lengths for each sample
lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
# Sort sequences by length in descending order (required for pack_padded_sequence)
lengths, idx_sort = lengths.sort(descending=True)
sequences = [sequences[i] for i in idx_sort]
targets = torch.stack([targets[i] for i in idx_sort])
# Pad sequences so that all have the same length in the batch.
padded_sequences = pad_sequence(sequences, batch_first=True)
return padded_sequences, lengths, targets

Create the dataset and dataloader.
dataset = SequenceDataset(fullDataSet)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

###############################################################

Define the RNN Model
The model takes inputs of shape (batch_size, max_seq_length, input_size)
and processes them with an RNN. A fully connected layer decodes the final
hidden state into a single scalar output.
Xavier uniform initialization is applied to both the fc layer and all RNN
weight parameters.
###############################################################
class RNNModel(nn.Module):
def init(self, input_size, hidden_size, num_layers, output_size):
super(RNNModel, self).init()
self.hidden_size = hidden_size
self.num_layers = num_layers


Collapse
    # RNN layer
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    # Fully connected layer that maps the final hidden state to the output.
    self.fc = nn.Linear(hidden_size, output_size)

    # Xavier initialization for the fully connected layer.
    nn.init.xavier_uniform_(self.fc.weight)
    if self.fc.bias is not None:
        nn.init.zeros_(self.fc.bias)

    # Xavier initialization for the RNN parameters.
    for name, param in self.rnn.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

def forward(self, x, lengths):
    # x shape: (batch_size, max_seq_length, input_size)
    # Initialize hidden state for RNN. For an RNN, it is common to start with zeros.
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    # Pack the padded sequences so that the RNN ignores the padded items.
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
    packed_out, _ = self.rnn(packed, h0)
    # Unpack the output.
    out, _ = pad_packed_sequence(packed_out, batch_first=True)

    # For each sample in the batch, extract the output corresponding to its final valid time step.
    batch_size = x.size(0)
    idx = (lengths - 1).view(-1, 1).expand(batch_size, out.size(2)).unsqueeze(1)
    last_outputs = out.gather(1, idx).squeeze(1)

    # Decode the last outputs to obtain our prediction.
    output = self.fc(last_outputs)
    return output
Determine the input size from one sample's feature dimension (D).
if len(fullDataSet) > 0:
input_size = fullDataSet[0].size(1)
else:
input_size = 10 # A default value in case fullDataSet is empty

Hyperparameters
hidden_size = 64 # Number of hidden units in the RNN; can be tuned.
num_layers = 2 # Number of stacked RNN layers.
output_size = 1 # Single scalar output (here, the predicted sequence length).
learning_rate = 0.001

Initialize model, loss function, and optimizer.
model = RNNModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss() # Mean Squared Error for regression.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

###############################################################

Training Loop
For each epoch and batch, we feed the padded sequences (and their true lengths)
into the network, compute the loss with respect to the true sequence lengths, and
update the model parameters using backpropagation.
###############################################################
num_epochs = 10
for epoch in range(num_epochs):
for features, lengths, targets in dataloader:
optimizer.zero_grad()
outputs = model(features, lengths)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")