import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Custom Dataset that loads runs from an Excel file
class MachineRunDataset(Dataset):
    def __init__(self, excel_path, run_id_col='Date', time_col='Time', 
                 exclude_cols=['No.', 'Desc', 'Date', 'Time', 'Millitm'], 
                 scaler=None, train_mode=True):
        """
        Args:
            excel_path: Path to the Excel file
            run_id_col: Column that identifies different runs
            time_col: Column used to sort the sequence
            exclude_cols: Columns not to be used as features
            scaler: Optional scaler for feature normalization
            train_mode: If True, split data for training; if False, use all data as input
        """
        # Load the Excel file
        df = pd.read_excel(excel_path)
        
        # Identify the feature columns (all measurement columns except excluded ones)
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Using {len(self.feature_cols)} features: {self.feature_cols}")
        
        # Initialize scaler if not provided
        self.scaler = scaler
        if self.scaler is None and train_mode:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.feature_cols])
        
        # Group by run identifier
        self.runs = []
        grouped = df.groupby(run_id_col)
        
        for run_id, group in grouped:
            # Sort by time to ensure sequence is in correct order
            group = group.sort_values(by=[time_col])
            
            # Require at least 10 timepoints per run
            if len(group) < 10:
                print(f"Skipping run {run_id} with only {len(group)} timepoints")
                continue
            
            # Scale features if scaler is provided
            if self.scaler is not None:
                features = self.scaler.transform(group[self.feature_cols])
            else:
                features = group[self.feature_cols].values
            
            if train_mode:
                # For training: use 80% as input, last point as target
                split_idx = int(len(group) * 0.8)
                input_seq = torch.tensor(features[:split_idx], dtype=torch.float32)
                
                # Target is the last row (assumed to be optimal parameters)
                target = torch.tensor(features[-1], dtype=torch.float32)
            else:
                # For inference: use all data as input
                input_seq = torch.tensor(features, dtype=torch.float32)
                target = torch.zeros(len(self.feature_cols), dtype=torch.float32)  # Dummy target
            
            # Store sequence length for weighting
            seq_length = len(input_seq)
            self.runs.append((input_seq, target, seq_length, run_id))
    
    def __len__(self):
        return len(self.runs)
    
    def __getitem__(self, idx):
        return self.runs[idx]
    
    def get_feature_names(self):
        return self.feature_cols
    
    def get_scaler(self):
        return self.scaler

# Collate function for the DataLoader to handle variable-length sequences
def collate_fn(batch):
    # batch is a list of tuples: (sequence, target, seq_length, run_id)
    sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    run_ids = [item[3] for item in batch]
    
    # Pad the sequences to the length of the longest run in the batch
    padded_seqs = pad_sequence(sequences, batch_first=True)
    
    return padded_seqs, lengths, targets, run_ids

# Define the enhanced GRU-based model with batch normalization and dropout
class EnhancedParameterOptimizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(EnhancedParameterOptimizer, self).__init__()
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
        
        # Apply batch normalization to each feature across the time dimension
        batch_size, seq_len, features = x.size()
        x_reshaped = x.reshape(-1, features)
        x_bn = self.input_bn(x_reshaped)
        x = x_bn.reshape(batch_size, seq_len, features)
        
        # Pack the padded sequence for efficient processing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        
        # hidden has shape (num_layers, batch_size, hidden_dim)
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
        fc_combined = fc1_drop + normalized  # Residual connection
        
        # Output layer
        output = self.fc2(fc_combined)
        
        return output

# Function to train the model
def train_model(excel_path, model_save_path="parameter_optimizer_model.pth", 
                batch_size=16, num_epochs=100, learning_rate=0.001, 
                hidden_dim=128, num_layers=2, dropout=0.3, patience=10):
    
    # Create dataset
    dataset = MachineRunDataset(excel_path)
    
    # If no runs were found, exit
    if len(dataset) == 0:
        print("No valid runs found in the dataset. Exiting.")
        return None
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Get dimensions from the dataset
    input_dim = len(dataset.get_feature_names())
    output_dim = input_dim  # Predicting the same parameters
    
    # Create model
    model = EnhancedParameterOptimizer(
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
        
        for padded_seqs, lengths, targets, run_ids in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(padded_seqs, lengths)
            
            # Compute loss for each element
            loss_all = criterion(outputs, targets)  # shape: (batch_size, output_dim)
            
            # Average loss for each sample across all parameters
            loss_per_sample = loss_all.mean(dim=1)  # shape: (batch_size)
            
            # Weight each sample's loss by its sequence length
            # Longer runs (with higher length values) will contribute proportionately more
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
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        
        # Update learning rate based on validation loss
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
                'feature_names': dataset.get_feature_names(),
                'scaler': dataset.get_scaler()
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
    
    return model, dataset.get_scaler(), dataset.get_feature_names()

# Function to predict optimal parameters for a new run
def predict_optimal_parameters(model, scaler, feature_names, input_data, run_id_col='Date', time_col='Time'):
    """
    Predict optimal parameters for a new run
    
    Args:
        model: Trained model
        scaler: Scaler used during training
        feature_names: List of feature column names
        input_data: DataFrame containing the new run data
        run_id_col: Column that identifies different runs
        time_col: Column used to sort the sequence
    
    Returns:
        DataFrame with predicted optimal parameters
    """
    # Create a dataset for inference
    exclude_cols = [col for col in input_data.columns if col not in feature_names]
    inference_dataset = MachineRunDataset(
        input_data, 
        run_id_col=run_id_col,
        time_col=time_col,
        exclude_cols=exclude_cols,
        scaler=scaler,
        train_mode=False
    )
    
    inference_loader = DataLoader(
        inference_dataset, 
        batch_size=1,  # Process one run at a time
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Set model to evaluation mode
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for padded_seqs, lengths, _, run_ids in inference_loader:
            # Forward pass
            outputs = model(padded_seqs, lengths)
            
            # Convert predictions back to original scale
            if scaler is not None:
                predictions = scaler.inverse_transform(outputs.numpy())
            else:
                predictions = outputs.numpy()
            
            # Create result dictionary
            result = {
                'run_id': run_ids[0],
                'sequence_length': lengths.item()
            }
            
            # Add predicted parameters
            for i, feature in enumerate(feature_names):
                result[f'optimal_{feature}'] = predictions[0, i]
            
            results.append(result)
    
    return pd.DataFrame(results)

# Main execution function
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Path to your Excel file
    excel_path = "machine_data.xlsx"  # Replace with your actual file path
    
    # Create output directory for results
    os.makedirs("results", exist_ok=True)
    
    # Train the model
    print("Training model...")
    model, scaler, feature_names = train_model(
        excel_path=excel_path,
        model_save_path="results/parameter_optimizer_model.pth",
        batch_size=16,
        num_epochs=100,
        learning_rate=0.001,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        patience=10
    )
    
    if model is None:
        print("Training failed. Exiting.")
        return
    
    # Load the best model for inference
    checkpoint = torch.load("results/parameter_optimizer_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # For demonstration, we'll use the same data for prediction
    # In a real scenario, you would use new data
    print("Predicting optimal parameters...")
    predictions = predict_optimal_parameters(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        input_data=pd.read_excel(excel_path)
    )
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions.to_csv(f"results/optimal_parameters_{timestamp}.csv", index=False)
    print(f"Predictions saved to results/optimal_parameters_{timestamp}.csv")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(predictions.head())

if __name__ == "__main__":
    main()