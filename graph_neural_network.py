import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# Import the data generator from train_dynamics to ensure the exact same dataset
from train_dynamics import generate_data, NUM_TRAJECTORIES, TIME_STEPS, DT, BATCH_SIZE, EPOCHS, LEARNING_RATE

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. DEFINE GNN ARCHITECTURE

class NodeEncoder(nn.Module):
    def __init__(self, node_dim):
        super(NodeEncoder, self).__init__()
        # Input features per node: [m, l, sin(theta), cos(theta), omega]
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, node_dim),
            nn.LayerNorm(node_dim)
        )
        
    def forward(self, x):
        # x is [batch_size, 4] with features [m, l, theta, omega]
        m = x[:, 0:1]
        l = x[:, 1:2]
        theta = x[:, 2:3]
        omega = x[:, 3:4]
        feats = torch.cat([m, l, torch.sin(theta), torch.cos(theta), omega], dim=1)
        return self.net(feats)

class EdgeModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(EdgeModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim),
            nn.LayerNorm(edge_dim)
        )
        
    def forward(self, src, dest):
        # Source and destination node embeddings
        out = torch.cat([src, dest], dim=-1)
        return self.net(out)

class NodeModel(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(NodeModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, node_dim),
            nn.LayerNorm(node_dim)
        )
        
    def forward(self, node, edge_msg):
        out = torch.cat([node, edge_msg], dim=-1)
        return self.net(out)

class DynamicsGNN(nn.Module):
    """
    Graph Neural Network for learning the Hamiltonian vector field of a Double Pendulum.
    
    Architecture Explanation:
    - Nodes: The double pendulum has 2 moving masses. Thus, the system is represented 
      as a graph with 2 nodes. 
        * Node 1 corresponds to inner mass: (m1, l1, theta1, omega1).
        * Node 2 corresponds to outer mass: (m2, l2, theta2, omega2).
    - Edges: The masses interact mechanically (coupled through the rods and hinges). 
      We establish bidirectional edges between Node 1 and Node 2.
    - Node Encoder: Transforms the raw localized state variables into a higher dimensional 
      hidden representation. To handle rotational invariants (periodicity of angles), we map 
      `theta` into `sin(theta)` and `cos(theta)` inside the encoder.
    - Message Passing: A sequence of spatial interactions where:
      1. Edge messages are computed using the MLP on concatenated source/destination node features.
      2. Node embeddings are updated by aggregating their previous state with incoming edge messages.
    - Readout: A linear layer maps the final node embeddings to the predicted local phase space 
      derivatives (d_theta, d_omega).
      
    Why this architecture?
    Because the underlying physics of multi-body systems originates from localized pairwise interactions 
    (forces/constraints between adjacent bodies), Graph Neural Networks possess the exact 
    inductive bias needed to learn their Hamiltonian/Lagrangian vector fields effectively and generally better than MLPs.
    """
    def __init__(self, num_message_passing=3):
        super(DynamicsGNN, self).__init__()
        self.num_mp = num_message_passing
        self.node_dim = 64
        self.edge_dim = 64
        
        self.node_encoder = NodeEncoder(self.node_dim)
        self.edge_model = EdgeModel(self.node_dim, self.edge_dim)
        self.node_model = NodeModel(self.node_dim, self.edge_dim)
        
        # Output [d_theta, d_omega] for each node
        self.readout = nn.Linear(self.node_dim, 2)
        
    def forward(self, x):
        # x shape: [batch_size, 8]
        # x variables: [m1, m2, l1, l2, theta1, theta2, omega1, omega2]
        
        # Extract features for node 1 and node 2
        n1_features = torch.stack([x[:, 0], x[:, 2], x[:, 4], x[:, 6]], dim=1)
        n2_features = torch.stack([x[:, 1], x[:, 3], x[:, 5], x[:, 7]], dim=1)
        
        h1 = self.node_encoder(n1_features)
        h2 = self.node_encoder(n2_features)
        
        for _ in range(self.num_mp):
            # Message Passing - Compute Edge Interactions
            # Edge 1->2 (message sent to node 2)
            e12 = self.edge_model(src=h1, dest=h2)
            # Edge 2->1 (message sent to node 1)
            e21 = self.edge_model(src=h2, dest=h1)
            
            # Node Feature Update (using residual connection)
            h1 = h1 + self.node_model(node=h1, edge_msg=e21)
            h2 = h2 + self.node_model(node=h2, edge_msg=e12)
            
        # Readout predicted vector field components
        out1 = self.readout(h1) # [d_theta1, d_omega1]
        out2 = self.readout(h2) # [d_theta2, d_omega2]
        
        # Match target format from generate_data: [d_theta1, d_theta2, d_omega1, d_omega2]
        d_theta1, d_omega1 = out1[:, 0], out1[:, 1]
        d_theta2, d_omega2 = out2[:, 0], out2[:, 1]
        
        return torch.stack([d_theta1, d_theta2, d_omega1, d_omega2], dim=1)

# 2. TRAINING LOOP

def main():
    # Generate Data
    X, Y = generate_data(NUM_TRAJECTORIES, TIME_STEPS, DT)
    print(f"Generated {X.shape[0]} transition pairs.")
    
    # Feature Normalization
    x_mean, x_std = X.mean(dim=0), X.std(dim=0) + 1e-8
    y_mean, y_std = Y.mean(dim=0), Y.std(dim=0) + 1e-8
    
    # Save scalers specifically for the GNN
    torch.save({'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}, "scalers_gnn.pt")
    
    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std
    
    # Validation split
    dataset = TensorDataset(X_norm, Y_norm)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize GNN Model
    model = DynamicsGNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
    # Train Loop
    print("Starting Training GNN...")
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
    torch.save(model.state_dict(), "pendulum_gnn.pth")
    print("GNN Model saved to pendulum_gnn.pth")
    
    # Plot Training Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Dynamics GNN (Hamiltonian Vector Field)')
    plt.savefig('training_loss_gnn.png')
    print("Training loss curve saved to training_loss_gnn.png")

if __name__ == "__main__":
    main()
