import numpy as np
import matplotlib.pyplot as plt

A = np.zeros((3, 3, 90))
A_prime = np.zeros((3, 3, 90))
Exx = np.zeros(90)
Eyy = np.zeros(90)
Gxy = np.zeros(90)

# Mechanical properties & Q Stiffness matrix
E11 = 182.0e9  # GPa
E22 = 7.5e9  # GPa
v12 = 0.36
G12 = 2.1e9  # GPa
v21 = E22 * v12 / E11
Q11 = E11 / (1 - v12 * v21)
Q22 = E22 / (1 - v21 * v12)
Q21 = v21 * E11 / (1 - v21 * v12)
Q12 = v12 * E22 / (1 - v21 * v12)
Q66 = G12

for i in range(90):
    layers = [i + 1] * 8
    thickness = 0.001
    h = len(layers) * thickness

    # Calculate A matrix
    for j in range(8):
        theta = layers[j]
        alpha = theta * np.pi / 180

        m = np.cos(alpha)
        n = np.sin(alpha)

        T = np.array([[m ** 2, n ** 2, 2 * m * n],
                      [n ** 2, m ** 2, -2 * m * n],
                      [-m * n, m * n, m ** 2 - n ** 2]])

        Q = np.array([[Q11, Q12, 0],
                      [Q21, Q22, 0],
                      [0, 0, Q66]])

        Q_macron = np.matmul(np.matmul(T.T, Q), T)

        A[:, :, j] = Q_macron * thickness

    # Compute A' (inverse of A)
    A_prime[:, :, i] = np.linalg.inv(np.sum(A, axis=2))

    # Calculate Engineering constants
    Exx[i] = (1 / h) * (1 / A_prime[0, 0, i])
    Eyy[i] = (1 / h) * (1 / A_prime[1, 1, i])
    Gxy[i] = (1 / h) * (1 / A_prime[2, 2, i])

theta = np.arange(1, 91)

plt.figure(figsize=(8, 6))

plt.subplot(3, 1, 1)
plt.plot(theta, Exx)
plt.xlim([0, 90])
plt.ylabel('$E_{xx}$ (GPa)')
plt.title('Engineering constants respect to θ')

plt.subplot(3, 1, 2)
plt.plot(theta, Eyy)
plt.xlim([0, 90])
plt.ylabel('$E_{yy}$ (GPa)')

plt.subplot(3, 1, 3)
plt.plot(theta, Gxy)
plt.xlim([0, 90])
plt.ylabel('$G_{xy}$ (GPa)')

plt.xlabel('θ (Degree) ')

plt.tight_layout()
plt.show()
