import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from double_pendulum import DoublePendulum # Assuming it's in the same dir

# Hyperparameters
NUM_TRAJECTORIES = 2000 # Increased from 500 (~98k transitions)
TIME_STEPS = 50      # steps per trajectory
DT = 0.05            # Time step resolution
BATCH_SIZE = 128
EPOCHS = 300         # Increased from 100
LEARNING_RATE = 1e-3

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. GENERATE TRAINING DATA
def generate_data(num_trajectories, time_steps, dt):
    print("Generating dataset from physics solver...")
    X_data = [] # Inputs: [m1, m2, l1, l2, theta1, theta2, omega1, omega2]
    Y_data = [] # Targets: Residuals [d_theta1, d_theta2, d_omega1, d_omega2]
    
    for i in range(num_trajectories):
        if i % 200 == 0:
            print(f"  Generating trajectory {i}/{num_trajectories}")
            
        # Randomize system parameters slightly around typical values
        m1 = np.random.uniform(0.5, 2.0)
        m2 = np.random.uniform(0.5, 2.0)
        l1 = np.random.uniform(0.5, 2.0)
        l2 = np.random.uniform(0.5, 2.0)
        
        # Randomize initial conditions
        theta10 = np.random.uniform(-np.pi, np.pi)
        theta20 = np.random.uniform(-np.pi, np.pi)
        omega10 = np.random.uniform(-2.0, 2.0)
        omega20 = np.random.uniform(-2.0, 2.0)
        y0 = [theta10, theta20, omega10, omega20]
        
        dp = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2)
        
        # Simulate trajectory
        t_span = (0, time_steps * dt)
        t_eval = np.linspace(0, time_steps * dt, time_steps)
        
        try:
            sol = dp.solve(t_span, y0, t_eval=t_eval)
            if sol.success:
                y_traj = sol.y.T # Shape: (time_steps, 4)
                
                for t in range(time_steps - 1):
                    curr = y_traj[t]
                    nxt = y_traj[t+1]
                    
                    # Compute residuals (difference)
                    # For angles, we need to handle the -pi to pi wrap-around nicely
                    d_theta1 = (nxt[0] - curr[0] + np.pi) % (2 * np.pi) - np.pi
                    d_theta2 = (nxt[1] - curr[1] + np.pi) % (2 * np.pi) - np.pi
                    d_omega1 = nxt[2] - curr[2]
                    d_omega2 = nxt[3] - curr[3]
                    
                    x_feature = [m1, m2, l1, l2] + curr.tolist()
                    y_residual = [d_theta1, d_theta2, d_omega1, d_omega2]
                    
                    X_data.append(x_feature)
                    Y_data.append(y_residual)
        except Exception:
            pass # Skip failed or extremely stiff trajectories

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float32)
    return X_tensor, Y_tensor

# 2. DEFINE THE NEURAL NETWORK (ResNet architecture)
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(x + self.block(x))

class DynamicsMLP(nn.Module):
    def __init__(self):
        super(DynamicsMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.output_layer = nn.Linear(256, 4)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

def main():
    # Generate Data
    X, Y = generate_data(NUM_TRAJECTORIES, TIME_STEPS, DT)
    print(f"Generated {X.shape[0]} transition pairs.")
    
    # Feature Normalization (Standardization)
    x_mean, x_std = X.mean(dim=0), X.std(dim=0) + 1e-8
    y_mean, y_std = Y.mean(dim=0), Y.std(dim=0) + 1e-8
    
    torch.save({'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}, "scalers.pt")
    
    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std
    
    # Validation split
    dataset = TensorDataset(X_norm, Y_norm)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize Model
    model = DynamicsMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # AdamW with weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
    # Train Loop
    print("Starting Training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / train_size
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
        epoch_val_loss = val_loss / val_size
        val_losses.append(epoch_val_loss)
        
        scheduler.step(epoch_val_loss)
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
    # Save Model
    torch.save(model.state_dict(), "pendulum_mlp.pth")
    print("Model saved to pendulum_mlp.pth")
    
    # Plot Training Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Dynamics ResNet (Residual Prediction)')
    plt.savefig('training_loss.png')
    print("Training loss curve saved to training_loss.png")

if __name__ == "__main__":
    main()
