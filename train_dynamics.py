import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from double_pendulum import DoublePendulum # Assuming it's in the same dir

# Hyperparameters
NUM_TRAJECTORIES = 500
TIME_STEPS = 50      # steps per trajectory
DT = 0.05            # Time step resolution
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. GENERATE TRAINING DATA
def generate_data(num_trajectories, time_steps, dt):
    print("Generating dataset from physics solver...")
    X_data = [] # Inputs: [m1, m2, l1, l2, theta1, theta2, omega1, omega2]
    Y_data = [] # Targets: [theta1', theta2', omega1', omega2'] at t+dt
    
    for i in range(num_trajectories):
        if i % 50 == 0:
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
        
        # Try solving. Sometimes random initial conditions may cause stiffness or errors
        try:
            sol = dp.solve(t_span, y0, t_eval=t_eval)
            if sol.success:
                y_traj = sol.y.T # Shape: (time_steps, 4)
                
                # Create (state_t, state_t+dt) pairs
                for t in range(time_steps - 1):
                    current_state = y_traj[t]
                    next_state = y_traj[t+1]
                    
                    x_feature = [m1, m2, l1, l2] + current_state.tolist()
                    y_target = next_state.tolist()
                    
                    # Normalize angles to [-pi, pi] to avoid predicting 4pi vs 2pi differences
                    y_target[0] = (y_target[0] + np.pi) % (2 * np.pi) - np.pi
                    y_target[1] = (y_target[1] + np.pi) % (2 * np.pi) - np.pi
                    
                    X_data.append(x_feature)
                    Y_data.append(y_target)
        except Exception as e:
            pass # Skip failed or extremely stiff trajectories

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float32)
    return X_tensor, Y_tensor

# 2. DEFINE THE NEURAL NETWORK
class DynamicsMLP(nn.Module):
    def __init__(self):
        super(DynamicsMLP, self).__init__()
        # Input: 8 features -> Hidden Layers -> Output: 4 features
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    # Generate Data
    X, Y = generate_data(NUM_TRAJECTORIES, TIME_STEPS, DT)
    print(f"Generated {X.shape[0]} transition pairs.")
    
    # Validation split (90/10)
    dataset = TensorDataset(X, Y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize Model
    model = DynamicsMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
            
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
    plt.title('Training Dynamics MLP')
    plt.savefig('training_loss.png')
    print("Training loss curve saved to training_loss.png")

if __name__ == "__main__":
    main()
