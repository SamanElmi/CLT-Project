import numpy as np
from IPython.display import display

# Function to calculate A, B, and D matrices for a given laminate configuration
def calculate_ABD(teta, thickness, Q):
    num = len(teta)
    alpha = np.deg2rad(teta)
    z = np.arange(-thickness * num / 2, thickness * num / 2 + thickness, thickness)

    A_total = np.zeros((3, 3))
    B_total = np.zeros((3, 3))
    D_total = np.zeros((3, 3))

    for i in range(num):
        m = np.cos(alpha[i])
        n = np.sin(alpha[i])
        T = np.array([[m**2, n**2, 2*m*n],
                      [n**2, m**2, -2*m*n],
                      [-m*n, m*n, m**2 - n**2]])

        Q_macron = np.matmul(np.matmul(T, Q), T.T)

        A = Q_macron * (z[i+1] - z[i])
        B = (1/2) * Q_macron * (z[i+1]**2 - z[i]**2)
        D = (1/3) * Q_macron * (z[i+1]**3 - z[i]**3)

        A_total += A
        B_total += B
        D_total += D

    return A_total, B_total, D_total

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

# Laminate configurations for each situation
configurations = {
    '1': [0, 90, 0],
    '2': [0, -45, 45, 90, 45, -45, 0],
    '3': [0, 90, 0],
    '4': [45, -45, 0, -45, 45],
    '5': [0, 90, 0]
}

thickness = 0.001  # Laminate thickness

# Calculate A, B, and D matrices for each situation
for situation, teta in configurations.items():
    A, B, D = calculate_ABD(teta, thickness, Q)
    equal_value = configurations[situation]

    print(f'Situation {situation} (Laminate: {equal_value}):')
    print('A matrix:')
    display(A)
    print('B matrix:')
    display(B)
    print('D matrix:')
    display(D)
    print('-------------------------------------------------------------')
