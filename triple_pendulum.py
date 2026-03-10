import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ChainPendulum:
    """
    An N-link chain pendulum numerical integrator using explicit Runge-Kutta.
    Uses the generalized matrix formulation of the Euler-Lagrange equations:
    M(theta) * d_omega = C(theta, omega) + G(theta)
    """
    def __init__(self, masses, lengths, g=9.81):
        """
        Initialize the N-link pendulum.
        
        Parameters:
        masses (list or array): Masses of the bobs [m1, m2, ..., mN]
        lengths (list or array): Lengths of the massless rods [l1, l2, ..., lN]
        g (float): Acceleration due to gravity
        """
        self.m = np.array(masses, dtype=np.float64)
        self.l = np.array(lengths, dtype=np.float64)
        self.N = len(masses)
        self.g = g
        
        if len(lengths) != self.N:
            raise ValueError("Number of masses must equal number of lengths.")

    def derivatives(self, t, y):
        """
        Calculate the state space derivatives.
        y = [theta1, ..., thetaN, omega1, ..., omegaN]
        """
        theta = y[:self.N]
        omega = y[self.N:]
        
        # M is the Mass/Inertia Matrix
        M = np.zeros((self.N, self.N))
        
        # C is the Coriolis/Centrifugal Torque vector
        C = np.zeros(self.N)
        
        # G is the Gravity Torque vector
        G = np.zeros(self.N)
        
        # Calculate matrix elements
        for i in range(self.N):
            # Sum of masses from linkage i to the end (used frequently)
            mass_sum_i = np.sum(self.m[i:])
            
            # G vector
            G[i] = -mass_sum_i * self.g * self.l[i] * np.sin(theta[i])
            
            for j in range(self.N):
                # The effective mass is the sum of masses from the outer-most index (max(i, j)) 
                # to the end of the chain.
                mass_sum_ij = np.sum(self.m[max(i, j):])
                
                # M Matrix
                M[i, j] = mass_sum_ij * self.l[i] * self.l[j] * np.cos(theta[i] - theta[j])
                
                # C Vector (Centrifugal terms)
                # Note: C relates to the non-linear velocity product terms.
                C[i] += mass_sum_ij * self.l[i] * self.l[j] * omega[j]**2 * np.sin(theta[i] - theta[j])
        
        # We want to solve M * d_omega = C + G for d_omega (angular accelerations)
        # B = C + G is the Right Hand Side 
        RHS = -C + G 
        
        # Solve the linear system
        d_omega = np.linalg.solve(M, RHS)
        
        # Return [d_theta, d_omega]
        return np.concatenate((omega, d_omega))

    def solve(self, t_span, y0, method='DOP853', rtol=1e-10, atol=1e-10, t_eval=None):
        """
        Solves the equations of motion for the N-link pendulum.
        """
        sol = solve_ivp(self.derivatives, t_span, y0, method=method, 
                        rtol=rtol, atol=atol, t_eval=t_eval)
        return sol

    def get_cartesian(self, sol_y):
        """
        Converts the angle state arrays into (x, y) Cartesian coordinate arrays 
        for plotting/animation.
        
        Returns lists of arrays: [x1, x2, ...], [y1, y2, ...]
        """
        thetas = sol_y[:self.N]
        
        x = [np.zeros_like(thetas[0])]
        y = [np.zeros_like(thetas[0])]
        
        for i in range(self.N):
            x_next = x[-1] + self.l[i] * np.sin(thetas[i])
            y_next = y[-1] - self.l[i] * np.cos(thetas[i])
            x.append(x_next)
            y.append(y_next)
            
        # Remove the origin (0,0) from the returned coordinates
        return x[1:], y[1:]

def main():
    # Example usage for a Triple Pendulum
    print("Testing Triple Pendulum Solver...")
    masses = [1.0, 1.0, 1.0]
    lengths = [1.0, 1.0, 1.0]
    
    tp = ChainPendulum(masses, lengths)
    
    # y0 = [theta1, theta2, theta3, omega1, omega2, omega3]
    y0 = [np.pi/2, np.pi/4, -np.pi/4, 0, 0, 0]
    
    t_span = (0, 10)
    fps = 60
    t_eval = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) * fps)
    
    print(f"Solving dynamics for N={tp.N} from t={t_span[0]}s to t={t_span[1]}s...")
    sol = tp.solve(t_span, y0, t_eval=t_eval)
    
    if sol.success:
        print("Integration successful!")
        
        x_coords, y_coords = tp.get_cartesian(sol.y)
        
        # --- Animation Setup ---
        fig = plt.figure(figsize=(6, 6))
        
        # Max reach of the pendulum
        max_len = np.sum(lengths) * 1.05
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-max_len, max_len), ylim=(-max_len, max_len))
        ax.set_aspect('equal')
        ax.grid()
        ax.set_title(f'{tp.N}-Link Pendulum Dynamics')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        line, = ax.plot([], [], 'o-', lw=2, color='black', markersize=6)
        
        # Traces for the tip
        trace, = ax.plot([], [], '-', lw=1, color='blue', alpha=0.5)
        history_x, history_y = [], []

        def init():
            line.set_data([], [])
            trace.set_data([], [])
            return line, trace

        def animate(i):
            # Coordinates for all joints at time step i
            thisx = [0] + [x[i] for x in x_coords]
            thisy = [0] + [y[i] for y in y_coords]
            
            # Trace the location of the final tip
            history_x.append(thisx[-1])
            history_y.append(thisy[-1])
            
            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            return line, trace
            
        print("Generating animation. Close the window to exit.")
        ani = animation.FuncAnimation(fig, animate, frames=len(sol.t),
                                      interval=1000/fps, blit=True, init_func=init, repeat=False)
        plt.show()
    else:
        print(f"Integration failed: {sol.message}")

if __name__ == "__main__":
    main()
