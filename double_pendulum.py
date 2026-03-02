import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DoublePendulum:
    """
    An accurate numerical integrator for a standard double pendulum 
    using explicit Runge-Kutta methods (e.g. DOP853).
    """
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        """
        Initialize the double pendulum with arbitrary masses and lengths.
        
        Parameters:
        m1, m2 (float): Masses of the bobs
        l1, l2 (float): Lengths of the massless rods
        g (float): Acceleration due to gravity
        """
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    def derivatives(self, t, y):
        """
        Calculate the state space derivatives.
        y = [theta1, theta2, omega1, omega2]
        """
        theta1, theta2, omega1, omega2 = y
        
        delta = theta1 - theta2
        
        # Denominators for the equations
        den1 = self.l1 * (2*self.m1 + self.m2 - self.m2 * np.cos(2*delta))
        den2 = self.l2 * (2*self.m1 + self.m2 - self.m2 * np.cos(2*delta))
        
        # Angular accelerations
        num1 = (-self.g * (2*self.m1 + self.m2) * np.sin(theta1) 
                - self.m2 * self.g * np.sin(theta1 - 2*theta2) 
                - 2 * np.sin(delta) * self.m2 * (self.l2 * omega2**2 + self.l1 * omega1**2 * np.cos(delta)))
        domega1 = num1 / den1
        
        num2 = 2 * np.sin(delta) * (self.l1 * omega1**2 * (self.m1 + self.m2) 
                                    + self.g * (self.m1 + self.m2) * np.cos(theta1) 
                                    + self.l2 * omega2**2 * self.m2 * np.cos(delta))
        domega2 = num2 / den2
        
        return [omega1, omega2, domega1, domega2]

    def solve(self, t_span, y0, method='DOP853', rtol=1e-10, atol=1e-10, t_eval=None):
        """
        Solves the equations of motion for the double pendulum.
        
        Parameters:
        t_span (tuple): (t0, tf) bounds of simulation.
        y0 (array_like): Initial conditions [theta1, theta2, omega1, omega2].
        method (str): Integration method. 'DOP853' is highly accurate for non-stiff problems.
        rtol, atol (float): Relative and absolute tolerances. High accuracy defaults.
        t_eval (array_like): Times at which to store the computed solution.
        
        Returns:
        OdeResult object from scipy.integrate.solve_ivp
        """
        sol = solve_ivp(self.derivatives, t_span, y0, method=method, 
                        rtol=rtol, atol=atol, t_eval=t_eval)
        return sol

    def energy(self, y):
        """
        Calculates the total energy of the system for a given state series.
        y is an array of shape (4, N) where N is number of time steps.
        Useful for checking the accuracy of the integrator (energy should be conserved).
        """
        theta1, theta2, omega1, omega2 = y
        
        # Kinetic energy
        T = 0.5 * self.m1 * self.l1**2 * omega1**2 + \
            0.5 * self.m2 * (self.l1**2 * omega1**2 + self.l2**2 * omega2**2 + 
                             2 * self.l1 * self.l2 * omega1 * omega2 * np.cos(theta1 - theta2))
            
        # Potential energy (y=0 at the hinge)
        V = -(self.m1 + self.m2) * self.g * self.l1 * np.cos(theta1) - \
            self.m2 * self.g * self.l2 * np.cos(theta2)
            
        return T + V

def main():
    # Example usage with arbitrary parameters
    m1, m2 = 1.0, 1.5
    l1, l2 = 1.0, 1.0
    
    dp = DoublePendulum(m1=m1, m2=m2, l1=l1, l2=l2)
    
    # Arbitrary initial conditions: [theta1, theta2, omega1, omega2]
    # e.g., releasing from almost perfectly horizontal
    y0 = [np.pi/2, np.pi/2, 0.0, 0.0]
    
    # Simulation time (0 to 20 seconds)
    t_span = (0, 20)
    
    # For a smooth plot/animation, request 60 fps output
    fps = 60
    t_eval = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) * fps)
    
    print(f"Solving double pendulum dynamics from t={t_span[0]}s to t={t_span[1]}s...")
    sol = dp.solve(t_span, y0, t_eval=t_eval)
    
    if sol.success:
        print("Integration successful!")
        
        # Calculate and show maximum deviation of total energy as a check of accuracy
        E = dp.energy(sol.y)
        dE_max = np.max(np.abs(E - E[0]))
        print(f"Maximum energy drift (energy conservation error): {dE_max:.5e} J")
        
        # Calculate Cartesian coordinates for plotting
        theta1, theta2 = sol.y[0], sol.y[1]
        x1 = dp.l1 * np.sin(theta1)
        y1 = -dp.l1 * np.cos(theta1)
        
        x2 = x1 + dp.l2 * np.sin(theta2)
        y2 = y1 - dp.l2 * np.cos(theta2)
        
        # Plot the static trajectory
        plt.figure(figsize=(6, 6))
        plt.title('Double Pendulum Trajectory')
        plt.plot(x1, y1, label='Bob 1 (m1)', alpha=0.5, color='blue')
        plt.plot(x2, y2, label='Bob 2 (m2)', alpha=0.7, color='red')
        plt.plot(0, 0, 'ko', markersize=6, label='Pivot')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        print("Close the plot window to finish script.")
        plt.show()
    else:
        print(f"Integration failed: {sol.message}")

if __name__ == "__main__":
    main()
