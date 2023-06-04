import numpy as np
import matplotlib.pyplot as plt

# Mechanical properties & Q Stiffness matrix
E11 = 182.0e9 #GPa
E22 = 7.5e9 #GPa
v12 = 0.36
G12 = 2.1e9 #GPa
v21 = E22 * v12 / E11
Q11 = E11 / (1 - v12 * v21)
Q22 = E22 / (1 - v21 * v12)
Q21 = v21 * E11 / (1 - v21 * v12)
Q12 = v12 * E22 / (1 - v21 * v12)
Q66 = G12
Q = np.array([[Q11, Q12, 0],
              [Q21, Q22, 0],
              [0, 0, Q66]])

# Range of θ values from 0 to 180 degrees
theta_range = np.linspace(0, 180, 181)

def calculate_Q_macron(Q, theta):
    Q_macron = np.zeros((3, 3))
    cos_theta = np.cos(np.deg2rad(theta))
    sin_theta = np.sin(np.deg2rad(theta))

    # Transformation matrix
    m = cos_theta
    n = sin_theta
    T = np.array([[m**2, n**2, 2*m*n],
                  [n**2, m**2, -2*m*n],
                  [-m*n, m*n, m**2 - n**2]])

    # Calculate Q_macron using transformation matrix
    Q_macron = np.matmul(np.matmul(np.transpose(T), Q), T)

    return Q_macron

def plot_Q_macron(Q, theta_range):
    Q_macron_values = []
    for theta in theta_range:
        Q_macron = calculate_Q_macron(Q, theta)
        Q_macron_values.append(Q_macron)

    Q_macron_values = np.array(Q_macron_values)

    # Plot the result
    plt.figure(figsize=(15, 10))

    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, i * 3 + j + 1)
            plt.plot(theta_range, Q_macron_values[:, i, j])
            plt.title(f'Q\u0304({i + 1},{j + 1})')
            plt.xlabel('θ (Deg)')
            plt.ylabel('Q\u0304 (GPa)')

    plt.tight_layout()
    plt.show()


plot_Q_macron(Q, theta_range)
