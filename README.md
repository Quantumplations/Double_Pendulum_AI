# Double Pendulum AI

This project explores the famously chaotic dynamics of a double pendulum system using a PyTorch ResNet to predict its motion.

## The Double Pendulum System

The double pendulum consists of two pendulums, with one attached to the end of the other. The state of the system is fully described by 8 parameters: the masses ($m_1, m_2$), the rod lengths ($l_1, l_2$), the angles from the vertical ($\theta_1, \theta_2$), and the angular velocities ($\omega_1, \omega_2$).

Below is a schematic representation of the system taken from Wikipedia:

<img width="294" height="398" alt="image" src="https://github.com/user-attachments/assets/bb1f53d0-1403-4a06-a8cb-1db7ea96ae7c" />


## Project Structure

*   `double_pendulum.py`: Contains the highly accurate `scipy.integrate.solve_ivp` (DOP853) numerical solver to generate true physics trajectories and real-time animations of the chaotic motion.
*   `train_dynamics.py`: The data generation and PyTorch training pipeline. We generate 100,000 continuous transitions and train a Deep Residual Network (ResNet) to predict the $\delta$ (change) in state for an infinitesimal timestep $dt$.
*   `evaluate_dynamics.py`: Evaluates the trained model against the physics solver using a completely unseen starting condition, generating a continuous autoregressive visual rollout to see exactly when the AI succumbs to the butterfly effect.
*   `architecture_diagram.md`: A Mermaid graph breaking down the flow of tensors through the ResNet architecture.

## AI Architecture Highlights
*   **Feature Normalization:** Uses Standard Scalers to convert all $(m_1, m_2, l_1, l_2, \theta_1, \theta_2, \omega_1, \omega_2)$ inputs into normalized zero-mean distributions.
*   **Residual Blocks:** Utilizes 3 concatenated residual blocks with Layer Normalization to smoothly learn complex topological mappings.
*   **Delta-Prediction:** Instead of predicting the absolute next state, the model correctly predicts the residual vector $x_{t+dt} - x_t$, drastically improving accuracy per timestep integration.
