import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Custom Dataset that uses your exact import method
class TirehogRunDataset(Dataset):
    def __init__(self, target_generation='last_point'):
        """
        Dataset that uses your exact import method for Tirehog data
        
        Args:
            target_generation: How to generate the target ('last_point' uses the last timepoint as target)
        """
        # Use your exact import method
        self.full_data_set = []  # List to store each sample tensor
        self.file_names = []     # Keep track of file names
        folder_path = "./Tirehog Dummy Data"  # Your exact path
        
        # Exact file loading logic from your code
        for file in os.listdir(folder_path):
            if file.startswith("ModifiedDataSample") and file.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file)
                try:
                    raw_data = pd.read_excel(file_path, "Sheet1")
                    # Exact indexing from your code
                    no_label_data = raw_data.iloc[3:, 1:]
                    full_data = no_label_data.to_numpy()
                    # Transpose exactly as in your code
                    full_data_t = full_data.T
                    # Convert to tensor exactly as in your code
                    self.full_data_set.append(torch.tensor(full_data_t, dtype=torch.float))
                    self.file_names.append(file)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
        
        print(f"Loaded {len(self.full_data_set)} samples using your exact import method")
        
        # Create input sequences and targets
        self.runs = []
        for i, data in enumerate(self.full_data_set):
            if len(data) < 10:  # Skip very short sequences
                continue
                
            # For training: use 80% as input, last point as target
            split_idx = int(len(data) * 0.8)
            input_seq = data[:split_idx]
            
            # Target is the last row (assumed to be optimal parameters)
            target = data[-1]
            
            # Store sequence length for weighting
            seq_length = len(input_seq)
            self.runs.append((input_seq, target, seq_length, self.file_names[i]))
    
    def __len__(self):
        return len(self.runs)
    
    def __getitem__(self, idx):
        return self.runs[idx]

# Collate function for DataLoader to handle variable-length sequences
def collate_fn(batch):
    # batch is a list of tuples: (sequence, target, seq_length, file_name)
    sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    file_names = [item[3] for item in batch]
    
    # Pad sequences to the length of the longest run in batch
    padded_seqs = pad_sequence(sequences, batch_first=True)
    
    return padded_seqs, lengths, targets, file_names

# Define the GRU-based model with batch normalization and dropout
class ParameterOptimizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(ParameterOptimizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # GRU layers with dropout
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization after GRU
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        # A fully connected network with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_length, input_dim)
        # lengths: tensor of the actual sequence lengths (batch_size)
        
        # Apply batch normalization to each feature
        batch_size, seq_len, features = x.size()
        x_reshaped = x.reshape(-1, features)
        x_bn = self.input_bn(x_reshaped)
        x = x_bn.reshape(batch_size, seq_len, features)
        
        # Pack the padded sequence for efficient processing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        
        # Take the hidden state from the last layer
        last_hidden = hidden[-1]  # shape: (batch_size, hidden_dim)
        
        # Apply batch normalization
        normalized = self.bn(last_hidden)
        
        # First dense layer with ReLU and dropout
        fc1_out = self.fc1(normalized)
        fc1_bn = self.fc_bn(fc1_out)
        fc1_relu = torch.relu(fc1_bn)
        fc1_drop = self.dropout(fc1_relu)
        
        # Add residual connection
        fc_combined = fc1_drop + normalized
        
        # Output layer
        output = self.fc2(fc_combined)
        
        return output

# Function to train the model
def train_model(model_save_path="parameter_optimizer_model.pth", 
                batch_size=8, num_epochs=50, learning_rate=0.001, 
                hidden_dim=64, num_layers=2, dropout=0.3, patience=10):
    
    # Create dataset using your exact import method
    dataset = TirehogRunDataset()
    
    # If no runs were found, exit
    if len(dataset) == 0:
        print("No valid runs found in the dataset. Exiting.")
        return None
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=min(batch_size, len(dataset)),
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Get dimensions from the first sample
    sample = dataset[0]
    input_dim = sample[0].shape[1]  # Feature dimension
    output_dim = sample[1].shape[0]  # Target dimension
    
    print(f"Model dimensions: input={input_dim}, hidden={hidden_dim}, output={output_dim}")
    
    # Create model
    model = ParameterOptimizer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Use MSELoss with reduction='none' to allow for weighting
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for padded_seqs, lengths, targets, file_names in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(padded_seqs, lengths)
            
            # Compute loss for each element
            loss_all = criterion(outputs, targets)  # shape: (batch_size, output_dim)
            
            # Average loss for each sample across all parameters
            loss_per_sample = loss_all.mean(dim=1)  # shape: (batch_size)
            
            # Weight each sample's loss by its sequence length
            weights = lengths.float() / lengths.float().mean()
            weighted_loss = (loss_per_sample * weights).mean()
            
            # Backward pass and optimize
            weighted_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += weighted_loss.item()
            batch_count += 1
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / max(1, batch_count)
        losses.append(avg_loss)
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'num_layers': num_layers
            }, model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    return model

# Function to predict optimal parameters for new data
def predict_optimal_parameters(model_path="parameter_optimizer_model.pth"):
    """
    Predict optimal parameters using your import method
    """
    # Load the trained model
    checkpoint = torch.load(model_path)
    
    # Create model with the same architecture
    model = ParameterOptimizer(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim'],
        num_layers=checkpoint['num_layers']
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Use your exact import method
    folder_path = "./Tirehog Dummy Data"
    results = []
    
    for file in os.listdir(folder_path):
        if file.startswith("ModifiedDataSample") and file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            try:
                # Exact import method
                raw_data = pd.read_excel(file_path, "Sheet1")
                no_label_data = raw_data.iloc[3:, 1:]
                full_data = no_label_data.to_numpy()
                full_data_t = full_data.T
                data_tensor = torch.tensor(full_data_t, dtype=torch.float)
                
                # Process with model
                with torch.no_grad():
                    # Use the actual sequence length
                    seq_length = torch.tensor([len(data_tensor)], dtype=torch.long)
                    # Add batch dimension
                    data_batch = data_tensor.unsqueeze(0)  # shape becomes [1, seq_len, features]
                    # Forward pass
                    output = model(data_batch, seq_length)
                    
                    # Create result entry
                    result = {
                        'file_name': file,
                        'sequence_length': len(data_tensor)
                    }
                    
                    # Add each predicted parameter
                    for i in range(output.shape[1]):
                        result[f'optimal_param_{i+1}'] = output[0, i].item()
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing {file} for prediction: {str(e)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Main execution function
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory for results
    os.makedirs("results", exist_ok=True)
    
    # Train the model
    print("Training model...")
    model = train_model(
        model_save_path="results/parameter_optimizer_model.pth",
        batch_size=8,
        num_epochs=50,
        learning_rate=0.001,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        patience=10
    )
    
    if model is None:
        print("Training failed. Exiting.")
        return
    
    # Predict optimal parameters
    print("Predicting optimal parameters...")
    predictions = predict_optimal_parameters(
        model_path="results/parameter_optimizer_model.pth"
    )
    
    if len(predictions) == 0:
        print("No predictions generated. Exiting.")
        return
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions.to_csv(f"results/optimal_parameters_{timestamp}.csv", index=False)
    print(f"Predictions saved to results/optimal_parameters_{timestamp}.csv")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(predictions.head())

if __name__ == "__main__":
    main()