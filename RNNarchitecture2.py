import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Helper function to safely read Excel files
def safe_read_excel(file_path, sheet_name="Sheet1"):
    """
    Safely read Excel files with proper error handling
    """
    try:
        # Read as strings first
        raw_data = pd.read_excel(
            file_path, 
            sheet_name=sheet_name,
            engine="openpyxl",
            dtype=str  # Read everything as strings first
        )
        
        # Skip the first 3 rows and first column
        no_label_data = raw_data.iloc[3:, 1:]
        
        # Convert to numeric, errors='coerce' will convert non-numeric to NaN
        numeric_data = no_label_data.apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values with 0 or another appropriate value
        numeric_data = numeric_data.fillna(0)
        
        # Get column names (these are your actual parameters)
        parameter_names = numeric_data.columns.tolist()
        
        # Convert to numpy array - NOT transposing here
        data = numeric_data.to_numpy(dtype=np.float32)
        
        return data, parameter_names, True
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None, None, False

# Custom Dataset that correctly handles the data structure
class TirehogRunDataset(Dataset):
    def __init__(self):
        """
        Dataset that correctly handles the Tirehog data structure
        """
        self.runs = []  # List to store processed runs
        self.file_names = []  # Keep track of file names
        self.parameter_names = None  # Store parameter names
        folder_path = "./Tirehog Dummy Data"  # Your exact path
        
        # Process each file
        for file in os.listdir(folder_path):
            if file.startswith("ModifiedDataSample") and file.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file)
                
                # Use the safe read function
                data, param_names, success = safe_read_excel(file_path)
                
                if success and data is not None:
                    # Store parameter names if not already stored
                    if self.parameter_names is None:
                        self.parameter_names = param_names
                    
                    # Check if data is valid
                    if data.size == 0 or np.isnan(data).any():
                        print(f"Warning: {file} contains invalid data, skipping")
                        continue
                    
                    # Convert to tensor - shape is (time_steps, num_parameters)
                    data_tensor = torch.tensor(data, dtype=torch.float)
                    
                    # Store the run
                    self.runs.append(data_tensor)
                    self.file_names.append(file)
                    print(f"Successfully loaded {file} with shape {data_tensor.shape} - {len(param_names)} parameters")
        
        print(f"Loaded {len(self.runs)} runs with {len(self.parameter_names)} parameters each")
    
    def __len__(self):
        return len(self.runs)
    
    def __getitem__(self, idx):
        return self.runs[idx], self.file_names[idx]
    
    def get_parameter_names(self):
        return self.parameter_names
    
    def get_num_parameters(self):
        return len(self.parameter_names) if self.parameter_names else 0

# Collate function for DataLoader to handle variable-length sequences
def collate_fn(batch):
    # batch is a list of tuples: (run_tensor, file_name)
    runs = [item[0] for item in batch]
    file_names = [item[1] for item in batch]
    
    # Get sequence lengths
    lengths = torch.tensor([len(run) for run in runs], dtype=torch.long)
    
    # Pad sequences to the length of the longest run in batch
    padded_runs = pad_sequence(runs, batch_first=True)
    
    return padded_runs, lengths, file_names

# Define the GRU-based model for parameter optimization
class ParameterOptimizer(nn.Module):
    def __init__(self, num_parameters, hidden_dim=64, num_layers=2, dropout=0.3):
        super(ParameterOptimizer, self).__init__()
        self.num_parameters = num_parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU to process the sequence of parameters
        self.gru = nn.GRU(
            input_size=num_parameters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer to predict the next set of parameters
        self.fc = nn.Linear(hidden_dim, num_parameters)
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_length, num_parameters)
        # lengths: tensor of the actual sequence lengths (batch_size)
        
        # Pack the padded sequence for efficient processing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Process with GRU
        _, hidden = self.gru(packed)
        
        # hidden has shape (num_layers, batch_size, hidden_dim)
        # Take the hidden state from the last layer
        last_hidden = hidden[-1]  # shape: (batch_size, hidden_dim)
        
        # Predict the next set of parameters
        next_params = self.fc(last_hidden)  # shape: (batch_size, num_parameters)
        
        return next_params
    
    def generate_sequence(self, seed_sequence, sequence_length=100):
        """
        Generate a sequence of optimal parameters
        
        Args:
            seed_sequence: Initial sequence to start generation from
            sequence_length: Length of the sequence to generate
        
        Returns:
            Generated sequence of parameters
        """
        self.eval()
        with torch.no_grad():
            # Start with the seed sequence
            current_sequence = seed_sequence.clone()
            
            # Generate new steps one by one
            generated_sequence = []
            
            # Use the last step of the seed as our first input
            current_input = current_sequence[-1:].unsqueeze(0)  # Shape: [1, 1, num_parameters]
            
            for _ in range(sequence_length):
                # Predict the next set of parameters
                next_params = self(current_input, torch.tensor([1]))
                
                # Add to our generated sequence
                generated_sequence.append(next_params.squeeze(0).numpy())
                
                # Update the input for the next step
                current_input = next_params.unsqueeze(1)  # Shape: [1, 1, num_parameters]
            
            return np.array(generated_sequence)

# Function to train the model
def train_model(model_save_path="parameter_optimizer_model.pth", 
                batch_size=8, num_epochs=50, learning_rate=0.001, 
                hidden_dim=64, num_layers=2, dropout=0.3, patience=10):
    
    # Create dataset
    dataset = TirehogRunDataset()
    
    # If no runs were found, exit
    if len(dataset) == 0:
        print("No valid runs found in the dataset. Exiting.")
        return None, None
    
    # Get the number of parameters
    num_parameters = dataset.get_num_parameters()
    parameter_names = dataset.get_parameter_names()
    
    if num_parameters == 0:
        print("No parameters found in the dataset. Exiting.")
        return None, None
    
    print(f"Training model with {num_parameters} parameters")
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=min(batch_size, len(dataset)),
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Create model
    model = ParameterOptimizer(
        num_parameters=num_parameters,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
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
        
        for padded_runs, lengths, file_names in dataloader:
            optimizer.zero_grad()
            
            # For each run, we'll use all but the last time step as input
            # and the last time step as the target
            inputs = padded_runs[:, :-1, :]
            targets = padded_runs[:, -1, :]
            
            # Adjust lengths for the input (one less than the original)
            input_lengths = lengths - 1
            
            # Skip any sequences that are too short after removing the last step
            valid_indices = input_lengths > 0
            if not valid_indices.any():
                continue
                
            # Filter to only valid sequences
            inputs = inputs[valid_indices]
            targets = targets[valid_indices]
            input_lengths = input_lengths[valid_indices]
            
            # Forward pass
            outputs = model(inputs, input_lengths)
            
            # Compute loss - weight by sequence length to prioritize longer runs
            weights = input_lengths.float() / input_lengths.float().mean()
            loss = criterion(outputs, targets)
            weighted_loss = (loss * weights.unsqueeze(1)).mean()
            
            # Backward pass and optimize
            weighted_loss.backward()
            
            # Gradient clipping
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
                'num_parameters': num_parameters,
                'parameter_names': parameter_names,
                'hidden_dim': hidden_dim,
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
    
    return model, parameter_names

# Function to generate optimal parameters
def generate_optimal_parameters(model_path="parameter_optimizer_model.pth", sequence_length=100):
    """
    Generate a sequence of optimal parameters
    
    Args:
        model_path: Path to the trained model
        sequence_length: Length of the sequence to generate
    
    Returns:
        DataFrame with the generated sequence
    """
    # Load the trained model
    checkpoint = torch.load(model_path)
    
    # Get model parameters
    num_parameters = checkpoint['num_parameters']
    parameter_names = checkpoint['parameter_names']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    
    # Create model with the same architecture
    model = ParameterOptimizer(
        num_parameters=num_parameters,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load a sample run to use as seed
    dataset = TirehogRunDataset()
    if len(dataset) == 0:
        print("No valid runs found. Cannot generate parameters.")
        return pd.DataFrame()
    
    # Use the longest run as seed
    longest_idx = 0
    longest_len = 0
    for i in range(len(dataset)):
        run, _ = dataset[i]
        if len(run) > longest_len:
            longest_len = len(run)
            longest_idx = i
    
    seed_run, seed_file = dataset[longest_idx]
    print(f"Using {seed_file} (length {len(seed_run)}) as seed for generation")
    
    # Generate a sequence of optimal parameters
    generated_sequence = model.generate_sequence(seed_run, sequence_length)
    
    # Create a DataFrame with the generated sequence
    result_df = pd.DataFrame(generated_sequence, columns=parameter_names)
    
    # Add a time step column
    result_df.insert(0, 'time_step', range(len(result_df)))
    
    return result_df

# Main execution function
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory for results
    os.makedirs("results", exist_ok=True)
    
    # Train the model
    print("Training model...")
    model, parameter_names = train_model(
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
    
    # Generate optimal parameters
    print("Generating optimal parameters...")
    optimal_parameters = generate_optimal_parameters(
        model_path="results/parameter_optimizer_model.pth",
        sequence_length=200  # Generate a longer sequence
    )
    
    if len(optimal_parameters) == 0:
        print("No parameters generated. Exiting.")
        return
    
    # Save the generated parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimal_parameters.to_csv(f"results/optimal_parameters_{timestamp}.csv", index=False)
    print(f"Optimal parameters saved to results/optimal_parameters_{timestamp}.csv")
    
    # Print sample of the generated parameters
    print("\nSample of generated optimal parameters:")
    print(optimal_parameters.head())
    
    # Also save as Excel for easier viewing
    optimal_parameters.to_excel(f"results/optimal_parameters_{timestamp}.xlsx", index=False)
    print(f"Optimal parameters also saved as Excel: results/optimal_parameters_{timestamp}.xlsx")

if __name__ == "__main__":
    main()