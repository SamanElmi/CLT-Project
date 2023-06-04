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

# Laminate configurations for each situation
configurations = {
    'a': [90, -45, 45, 0],
}

thickness = 0.001  # Laminate thickness

# Calculate A, B, and D matrices for each situation
for situation, teta in configurations.items():
    A, B, D = calculate_ABD(teta, thickness, Q)

S = np.block([[A, B],
              [B, D]])




C = np.linalg.inv(S)

Load = np.array([[1500], [0], [0], [0], [0], [0]])
#Part a:
Strains = np.dot(C, Load)
print('Part 1: \nStrains and Curvatures=')
print("Strains:")
print("εₓ:", *Strains[0])
print("εᵧ:", *Strains[1])
print("γₓᵧ:", *Strains[2])
print("Curvatures:")
print("κₓ:", *Strains[3])
print("κᵧ:", *Strains[4])
print("κₓᵧ:", *Strains[5])
print("\nPart2: ")


#Part b



# Load vector
Nx = 1500
Ny = 0
Nxy = 0
Mx = 0
My = 0
Mxy = 0
Load = np.array([Nx, Ny, Nxy, Mx, My, Mxy])

# Angles of piles
teta = [90, 45, -45, 0]
alpha = np.deg2rad(teta)

num = len(teta)
thickness = 0.001
z = np.arange(-thickness * num / 2, thickness * num / 2 + thickness, thickness)


# Obtaining A, B, and D for each lamina
A = np.zeros((3, 3, num))
B = np.zeros((3, 3, num))
D = np.zeros((3, 3, num))

for i in range(num):
    m = np.cos(alpha[i])
    n = np.sin(alpha[i])
    T = np.array([[m**2, n**2, 2*m*n],
                  [n**2, m**2, -2*m*n],
                  [-m*n, m*n, m**2 - n**2]])
    
    Q_macron = np.matmul(np.matmul(T, Q), T.T)
    
    A[:, :, i] = Q_macron * (z[i+1] - z[i])
    B[:, :, i] = (1/2) * Q_macron * (z[i+1]**2 - z[i]**2)
    D[:, :, i] = (1/3) * Q_macron * (z[i+1]**3 - z[i]**3)

# A, B, and D for the laminate
Atotal = np.sum(A, axis=2)
Btotal = np.sum(B, axis=2)
Dtotal = np.sum(D, axis=2)

# Calculate S matrix and the strains and curvatures
S = np.block([[Atotal, Btotal],
              [Btotal, Dtotal]])
C = np.linalg.inv(S)
Strain_meter = np.matmul(C, Load)
Strain_millimeter = Strain_meter * 1000

Q_macron = np.zeros((3, 3, num))

# Obtaining A, B, and D for each lamina
for i in range(num):
    m = np.cos(alpha[i])
    n = np.sin(alpha[i])
    T = np.array([[m**2, n**2, 2*m*n],
                  [n**2, m**2, -2*m*n],
                  [-m*n, m*n, m**2 - n**2]])
    Q_macron[:, :, i] = np.matmul(np.matmul(T, Q), T.T)

stress_bot = np.zeros((3, num))
stress_top = np.zeros((3, num))

for i in range(num):
    stress_bot[:, i] = np.matmul(Q_macron[:, :, i], Strain_meter[:3] + z[i] * Strain_meter[3:])
    stress_top[:, i] = np.matmul(Q_macron[:, :, i], Strain_meter[:3] + z[i+1] * Strain_meter[3:])

stress_total = np.zeros((3, 2*num))
k = 0
for i in range(num+1):
    if i != 0 and i != num:
        stress_total[:, k] = stress_top[:, i-1]
        k += 1
    stress_total[:, k] = stress_bot[:, i-1]
    k += 1

z_plot = np.linspace(-0.002, 0.002, 2*num)

# Plot the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i in range(2*num):
    plt.plot(stress_total[0, i], z_plot[i], marker='o')
plt.title('$σ_x$')
plt.xlabel('Stress (Pa)')
plt.ylabel('z (m)')
plt.grid(True)

plt.subplot(1, 3, 2)
for i in range(2*num):
    plt.plot(stress_total[1, i], z_plot[i], marker='o')
plt.title('$σ_y$')
plt.xlabel('Stress (Pa)')
plt.ylabel('z (m)')
plt.grid(True)

plt.subplot(1, 3, 3)
for i in range(2*num):
    plt.plot(stress_total[2, i], z_plot[i], marker='o')
plt.title('$σ_{xy}$')
plt.xlabel('Stress (Pa)')
plt.ylabel('z (m)')
plt.grid(True)

plt.show()
