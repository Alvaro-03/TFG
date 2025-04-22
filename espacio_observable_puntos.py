import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

# Cargar el entorno (sin mover el robot)
env = gym.make('Ur-v5', render_mode=None)

# Número de intentos
EPS = 10000

# Listas para almacenar todos los puntos generados
all_x = [[], [], [], []]
all_y = [[], [], [], []]
all_z = [[], [], [], []]

colors = ['red', 'green', 'blue', 'purple']

# Parámetros de la esfera
esfera_centro = np.array([0.325, 0.127, 0.3])
esfera_radio = 0.3

# Generar EPS planos sin mover el robot
for episode in range(EPS):
    obs = env.reset()[0]
    base_env = env.unwrapped
    puntos = base_env.data.qpos[-12:]
    puntos = puntos.reshape(4, 3)  # 4 puntos con (x, y, z)

    for i in range(4):
        all_x[i].append(puntos[i, 0])
        all_y[i].append(puntos[i, 1])
        all_z[i].append(puntos[i, 2])

output_folder = os.getcwd()

# 📌 1. Perspectiva 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for i in range(4):
    ax.scatter(all_x[i], all_y[i], all_z[i], c=colors[i], marker='o', alpha=0.5, label=f'Punto {i+1}')

# Dibujar la esfera 3D
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = esfera_centro[0] + esfera_radio * np.cos(u) * np.sin(v)
y = esfera_centro[1] + esfera_radio * np.sin(u) * np.sin(v)
z = esfera_centro[2] + esfera_radio * np.cos(v)
ax.plot_surface(x, y, z, color='gray', alpha=0.1, linewidth=0)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Distribución de 200 Planos Generados con Esfera")
ax.legend()
plt.savefig(os.path.join(output_folder, "planos_3D.png"), dpi=300, bbox_inches='tight')
plt.close()

# 📌 2. Vista Lateral YZ
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(4):
    ax.scatter(all_y[i], all_z[i], c=colors[i], marker='o', alpha=0.5, label=f'Punto {i+1}')
circle = plt.Circle((esfera_centro[1], esfera_centro[2]), esfera_radio, color='gray', fill=False, linewidth=1)
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_xlabel("Y")
ax.set_ylabel("Z")
ax.set_title("Vista Lateral (YZ) - Sin X con Esfera")
ax.legend()
plt.savefig(os.path.join(output_folder, "planos_YZ.png"), dpi=300, bbox_inches='tight')
plt.close()

# 📌 3. Vista Lateral XZ
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(4):
    ax.scatter(all_x[i], all_z[i], c=colors[i], marker='o', alpha=0.5, label=f'Punto {i+1}')
circle = plt.Circle((esfera_centro[0], esfera_centro[2]), esfera_radio, color='gray', fill=False, linewidth=1)
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_title("Vista Lateral (XZ) - Sin Y con Esfera")
ax.legend()
plt.savefig(os.path.join(output_folder, "planos_XZ.png"), dpi=300, bbox_inches='tight')
plt.close()

# 📌 4. Vista Cenital XY
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(4):
    ax.scatter(all_x[i], all_y[i], c=colors[i], marker='o', alpha=0.5, label=f'Punto {i+1}')
circle = plt.Circle((esfera_centro[0], esfera_centro[1]), esfera_radio, color='gray', fill=False, linewidth=1)
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Vista Cenital (XY) - Sin Z con Esfera")
ax.legend()
plt.savefig(os.path.join(output_folder, "planos_XY.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Imágenes guardadas en: {output_folder}")
env.close()
