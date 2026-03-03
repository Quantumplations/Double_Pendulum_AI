import numpy as np
import torch
import matplotlib.pyplot as plt
from double_pendulum import DoublePendulum
from train_dynamics import DynamicsMLP, DT

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("Loading Trained Model & Scalers...")
    model = DynamicsMLP().to(device)
    try:
        model.load_state_dict(torch.load("pendulum_mlp.pth", map_location=device, weights_only=True))
        scalers = torch.load("scalers.pt", map_location=device, weights_only=True)
        x_mean, x_std = scalers['x_mean'].to(device), scalers['x_std'].to(device)
        y_mean, y_std = scalers['y_mean'].to(device), scalers['y_std'].to(device)
    except FileNotFoundError:
        print("Error: pendulum_mlp.pth or scalers.pt not found. Run train_dynamics.py first.")
        return
    model.eval()

    # Define a test scenario: parameters the model may or may not have seen exactly
    m1, m2, l1, l2 = 0.6, 0.33, 0.46, 1.1
    
    # Unseen initial conditions: [theta1, theta2, omega1, omega2] 
    y0 = [np.pi/5, -np.pi/7, -3.1, 2.4]
    
    # Time parameters for rollout
    trajectory_seconds = 5.0
    steps = int(trajectory_seconds / DT)
    t_span = (0, trajectory_seconds)
    t_eval = np.linspace(0, trajectory_seconds, steps)
    
    print(f"1. Generating Ground Truth trajectory using Scipy ODE Solver...")
    dp = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2)
    sol = dp.solve(t_span, y0, t_eval=t_eval)
    
    if not sol.success:
        print("Solver failed. Use different initial conditions.")
        return
        
    true_trajectory = sol.y.T # shape: (steps, 4)
    
    print("2. Generating AI Rollout via Autoregressive Simulation...")
    ai_trajectory = np.zeros((steps, 4))
    ai_trajectory[0] = y0
    
    current_state = np.array(y0)
    with torch.no_grad():
        for t in range(1, steps):
            # Input features: [m1, m2, l1, l2, theta1, theta2, omega1, omega2]
            features = torch.tensor([m1, m2, l1, l2] + current_state.tolist(), dtype=torch.float32).to(device)
            features_norm = (features - x_mean) / x_std
            
            # Predict residual (delta)
            residual_norm = model(features_norm.unsqueeze(0)).squeeze(0)
            residual = (residual_norm * y_std) + y_mean
            residual = residual.cpu().numpy()
            
            # Compute next state
            next_state = current_state + residual
            
            # Normalize predicted angles
            next_state[0] = (next_state[0] + np.pi) % (2 * np.pi) - np.pi
            next_state[1] = (next_state[1] + np.pi) % (2 * np.pi) - np.pi
            
            ai_trajectory[t] = next_state
            current_state = next_state
            
    print("3. Plotting comparison...")
    
    # Plotting Angles
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Theta 1
    axs[0].plot(t_eval, true_trajectory[:, 0], 'b-', label='True Theta 1 (Physics)')
    axs[0].plot(t_eval, ai_trajectory[:, 0], 'r--', label='Predicted Theta 1 (AI)')
    axs[0].set_ylabel('Theta 1 (rad)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title(f'Enhanced ResNet Rollout vs Physics over {trajectory_seconds}s')
    
    # Theta 2
    axs[1].plot(t_eval, true_trajectory[:, 1], 'g-', label='True Theta 2 (Physics)')
    axs[1].plot(t_eval, ai_trajectory[:, 1], 'm--', label='Predicted Theta 2 (AI)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Theta 2 (rad)')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('rollout_comparison.png')
    print("Saved plot to rollout_comparison.png")
    print("Close the window to end script.")
    plt.show()

if __name__ == "__main__":
    main()
