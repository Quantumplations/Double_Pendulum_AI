import numpy as np
import torch
import matplotlib.pyplot as plt
from triple_pendulum import ChainPendulum
from train_dynamics import DT
from graph_neural_network import DynamicsGNN

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gnn_forward_N_nodes(model, features_norm, N):
    """
    Manually replicates the GNN forward pass for an N-node chain graph.
    The original model hardcoded N=2 nodes. This generalizes the message passing.
    """
    # 1. Encode all nodes
    h = []
    for i in range(N):
        # Extract [m_i, l_i, theta_i, omega_i] for node i
        n_features = torch.stack([
            features_norm[:, i],         # mass
            features_norm[:, N + i],     # length
            features_norm[:, 2*N + i],   # theta
            features_norm[:, 3*N + i]    # omega
        ], dim=1)
        h.append(model.node_encoder(n_features))
        
    # 2. Sequential Message Passing
    for _ in range(model.num_mp):
        # We need to collect all updates before applying them to simulate parallel message passing
        h_updates = [0] * N
        
        for i in range(N - 1):
            # Edge i <-> i+1
            h_src = h[i]
            h_dst = h[i+1]
            
            # Message from src to dst
            e_src_dst = model.edge_model(src=h_src, dest=h_dst)
            # Message from dst to src
            e_dst_src = model.edge_model(src=h_dst, dest=h_src)
            
            # Accumulate node updates
            h_updates[i] = h_updates[i] + model.node_model(node=h[i], edge_msg=e_dst_src)
            h_updates[i+1] = h_updates[i+1] + model.node_model(node=h[i+1], edge_msg=e_src_dst)
            
        # Apply updates with residual connection
        for i in range(N):
            h[i] = h[i] + h_updates[i]
            
    # 3. Readout for each node
    predicted_derivatives = []
    for i in range(N):
        out = model.readout(h[i])
        predicted_derivatives.append(out[:, 0]) # d_theta
        
    for i in range(N):
        out = model.readout(h[i])
        predicted_derivatives.append(out[:, 1]) # d_omega
        
    return torch.stack(predicted_derivatives, dim=1)


def main():
    print("Loading Trained GNN Model & Scalers (trained ONLY on Double Pendulum)...")
    model = DynamicsGNN().to(device)
    try:
        model.load_state_dict(torch.load("pendulum_gnn.pth", map_location=device, weights_only=True))
        scalers = torch.load("scalers_gnn.pt", map_location=device, weights_only=True)
        x_mean, x_std = scalers['x_mean'].to(device), scalers['x_std'].to(device)
        y_mean, y_std = scalers['y_mean'].to(device), scalers['y_std'].to(device)
    except FileNotFoundError:
        print("Error: pendulum_gnn.pth or scalers_gnn.pt not found. Run graph_neural_network.py first.")
        return
    model.eval()

    # N = 3 for Triple Pendulum
    N = 3
    masses = [1.0, 1.0, 1.0]
    lengths = [1.0, 1.0, 1.0]
    
    # [theta1, theta2, theta3, omega1, omega2, omega3]
    y0 = [np.pi/3, 0.0, -np.pi/3, 0.0, 0.0, 0.0]
    
    # Target structure the GNN expects for normalizer:
    # Double pendulum was: [m1, m2, l1, l2, theta1, theta2, omega1, omega2] (size 8)
    # We must pad our 3-pendulum inputs to trick the double-pendulum normalizer, 
    # but the generic N-node forward pass will just reuse the single-feature scaler values.
    
    # The normalizer mean/std is size 8.
    # Feature order for generic N: [m1..mN, l1..lN, theta1..thetaN, omega1..omegaN]
    
    # We will compute the rolling mean/std of the double pendulum features and apply that 
    # uniform scalar average to all nodes in the triple pendulum.
    m_mean = x_mean[0:2].mean()
    m_std = x_std[0:2].mean()
    
    l_mean = x_mean[2:4].mean()
    l_std = x_std[2:4].mean()
    
    theta_mean = x_mean[4:6].mean()
    theta_std = x_std[4:6].mean()
    
    omega_mean = x_mean[6:8].mean()
    omega_std = x_std[6:8].mean()
    
    y_dtheta_mean = y_mean[0:2].mean()
    y_dtheta_std = y_std[0:2].mean()
    
    y_domega_mean = y_mean[2:4].mean()
    y_domega_std = y_std[2:4].mean()
    
    
    # Time parameters for rollout
    trajectory_seconds = 3.0
    steps = int(trajectory_seconds / DT)
    t_span = (0, trajectory_seconds)
    t_eval = np.linspace(0, trajectory_seconds, steps)
    
    print(f"1. Generating Ground Truth trajectory using Scipy ODE Solver for Triple Pendulum...")
    dp = ChainPendulum(masses, lengths)
    sol = dp.solve(t_span, y0, t_eval=t_eval)
    
    if not sol.success:
        print("Solver failed. Use different initial conditions.")
        return
        
    true_trajectory = sol.y.T # shape: (steps, 6)
    
    print("2. Generating Zero-Shot AI Rollout via GNN...")
    gnn_trajectory = np.zeros((steps, 2*N)) # 6 variables
    gnn_trajectory[0] = y0
    
    current_state = np.array(y0)
    with torch.no_grad():
        for t in range(1, steps):
            # Input features shape [batch_size, 4N]
            # [m1, m2, m3, l1, l2, l3, theta1, theta2, theta3, omega1, omega2, omega3]
            features_raw = np.concatenate((masses, lengths, current_state))
            features = torch.tensor(features_raw, dtype=torch.float32).to(device).unsqueeze(0)
            
            # Normalize manually using the average scalers
            features_norm = features.clone()
            features_norm[:, 0:N] = (features_norm[:, 0:N] - m_mean) / m_std
            features_norm[:, N:2*N] = (features_norm[:, N:2*N] - l_mean) / l_std
            features_norm[:, 2*N:3*N] = (features_norm[:, 2*N:3*N] - theta_mean) / theta_std
            features_norm[:, 3*N:4*N] = (features_norm[:, 3*N:4*N] - omega_mean) / omega_std
            
            # Predict residual (delta) for all 3 nodes!
            residual_norm = gnn_forward_N_nodes(model, features_norm, N).squeeze(0)
            
            # Un-normalize manually
            residual = residual_norm.clone()
            residual[0:N] = (residual[0:N] * y_dtheta_std) + y_dtheta_mean
            residual[N:2*N] = (residual[N:2*N] * y_domega_std) + y_domega_mean
            
            residual = residual.cpu().numpy()
            
            # Compute next state
            next_state = current_state + residual
            
            # Normalize predicted angles
            for i in range(N):
                next_state[i] = (next_state[i] + np.pi) % (2 * np.pi) - np.pi
            
            gnn_trajectory[t] = next_state
            current_state = next_state
            
    print("3. Plotting comparison...")
    
    # Plotting Angles
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    for i in range(N):
        axs[i].plot(t_eval, true_trajectory[:, i], 'b-', label=f'True Theta {i+1} (Physics)')
        axs[i].plot(t_eval, gnn_trajectory[:, i], 'r--', label=f'Predicted Theta {i+1} (Zero-Shot GNN)')
        axs[i].set_ylabel(f'Theta {i+1} (rad)')
        axs[i].legend()
        axs[i].grid(True)
        
    axs[0].set_title(f'ZERO SHOT: Double-Pendulum GNN rolling out a Triple Pendulum ({trajectory_seconds}s)')
    axs[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('triple_pendulum_rollout.png')
    print("Saved plot to triple_pendulum_rollout.png")

if __name__ == "__main__":
    main()
