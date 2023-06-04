import numpy as np
import matplotlib.pyplot as plt

# Given parameters
α1 = -4.60e-7  # /°C
α2 = 3.27e-5   # /°C
θ = np.linspace(0, 90, 100)  # Range of θ values

a = np.array([α1, α2, 0])

alpha_global = np.zeros((3, len(θ)))

αx = np.zeros(len(θ))
αy = np.zeros(len(θ))
αxy = np.zeros(len(θ))

for i in range(len(θ)):
    θrad = θ[i] * np.pi / 180
    m = np.cos(θrad)
    n = np.sin(θrad)
    T = np.array([[m**2, n**2, 2*m*n],
                  [n**2, m**2, -2*m*n],
                  [-m*n, m*n, m**2 - n**2]])
    alpha_i = np.matmul(T, a)
    alpha_global[:, i] = alpha_i
    αx[i] = alpha_i[0]
    αy[i] = alpha_i[1]
    αxy[i] = alpha_i[2]


# Plot the results
print("Part 1:")
plt.figure(figsize=(8, 6))
plt.plot(θ, αx, label='$α_x$')
plt.plot(θ, αy, label='$α_y$')
plt.plot(θ, αxy, label='$α_{xy}$')
plt.xlabel('θ (degrees)')
plt.ylabel('Thermal Expansion Coefficient (α)')
plt.title('Thermal Expansion Coefficients for a Single Ply')
plt.legend()
plt.grid(True)
plt.show()


# Part b:



# The teta range
Laminate = np.array([np.arange(0, 91), np.arange(0, -91, -1), np.arange(0, -91, -1), np.arange(0, 91)])

# the alpha vector in the local coordinate
a1 = -4.5E-7
a2 = 3.17E-5
a = np.array([a1, a2, 0])

# calculation of the alpha vector in global coordinate for each laminate
alpha_global = np.zeros((3, Laminate.shape[1]))
for i in range(Laminate.shape[1]):
    alpha_i = np.zeros(3)
    for j in range(4):
        alpha = Laminate[j, i] * np.pi / 180
        m = np.cos(alpha)
        n = np.sin(alpha)
        T = np.array([[m**2, n**2, 2*m*n],
                      [n**2, m**2, -2*m*n],
                      [-m*n, m*n, m**2-n**2]])
        alpha_i += np.matmul(T, a)
    alpha_i /= 4
    alpha_global[:, i] = alpha_i

# Plot the results
print("Part 2:")
plt.figure(figsize=(8, 6))
plt.subplot(3, 1, 1)
plt.plot(Laminate[0], alpha_global[0])
plt.title('$α_x$')
plt.xlabel('θ (degrees)')
plt.ylabel('1/C')

plt.subplot(3, 1, 2)
plt.plot(Laminate[0], alpha_global[1])
plt.title('$α_y$')
plt.xlabel('θ (degrees)')
plt.ylabel('1/C')

plt.subplot(3, 1, 3)
plt.plot(Laminate[0], alpha_global[2])
plt.title('$α_{xy}$')
plt.xlabel('θ (degrees)')
plt.ylabel('1/C')

plt.tight_layout()
plt.grid(True)
plt.show()
