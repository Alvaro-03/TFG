import gymnasium as gym
from stable_baselines3 import SAC
import cv2
import numpy as np
import pandas as pd

# Ruta correcta del modelo entrenado
path_to_model = "models/SAC_8.zip"

# Cargar el entorno correcto
env = gym.make('Ur-v5', render_mode=None)  # Asegurarse de que es 'Ur-v3'
# env = gym.make('Ur-v5', render_mode="human")  # Asegurarse de que es 'Ur-v3'

# Cargar el modelo entrenado
model = SAC.load(path_to_model, env)

obs = env.reset()[0]
done = False
imagen_guardada = False

EPS = 100  # Número de episodios a probar
success_count = 0  # Contador de éxitos
action_average = 0
velocidad_average = 0
steps_average = 0
velocidad_total = 0
colision_total = 0
oscilacion_total = 0
distancia_total = 0
dispersion_total = 0
recompensa_total = 0

# Historial de recompensas y penalizaciones
desglose_recompensas = []

print("\n🚀 **Iniciando prueba del modelo SAC_4 en Ur-v3** 🚀\n")

for episode in range(EPS):
    obs = env.reset()[0]  # Reiniciar el entorno al inicio de cada episodio
    done = False
    step_count = 0  # Contador de pasos en el episodio
    episodio_info = {
        "reward_total": 0,
        "reward_dist": 0,
        "rw_goal": 0,
        "rw_collition": 0, 
        "action_rate_rew": 0, 
        "reward_dispersion": 0
    }
    
    print(episode)
    prev_action = [0,0,0,0,0,0]
    total_action = 0
    total_velocidades = 0

    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)  # Obtener la acción del modelo entrenado
        obs, reward, _, done, info = env.step(action)  # Aplicar la acción en el entorno

        total_action += np.square(action - prev_action).sum()
        prev_action = action

        # env.render()  # Renderizar la simulación
        step_count += 1  # Contar los pasos en el episodio

        # Sumar los valores en el episodio actual
        episodio_info["reward_total"] += reward
        episodio_info["reward_dist"] += info.get("reward_dist", 0)
        # episodio_info["reward_ctrl"] += info.get("reward_ctrl", 0)
        episodio_info["velocity_penalty_weight"] += info.get("velocity_penalty_weight", 0)
        episodio_info["rw_goal"] += info.get("rw_goal", 0)
        episodio_info["rw_collition"] += info.get("rw_collition", 0)
        episodio_info["reward_dispersion"] += info.get("reward_dispersion", 0)
        episodio_info["action_rate_rew"] += info.get("action_rate_rew", 0)

        # Acceder al entorno interno antes de obtener pos_xyT y pos_xy
        base_env = env.unwrapped 

        distancias_puntos = np.linalg.norm(base_env.pos_xyT.reshape(-1, 2) - base_env.pos_xy.reshape(-1, 2), axis=1)
        total_velocidades += base_env.data.qvel[:6]

        umbral_objetivo = 24
        if np.all(distancias_puntos < umbral_objetivo): 
            print(f"✅ Episodio {episode + 1}: Plano alineado en {step_count} pasos!")
            success_count += 1
            break  # Terminar el episodio
    
    # print(episodio_info)
    action_average += total_action/step_count
    velocidad_average += total_velocidades/step_count
    steps_average += step_count
    """
    print("Velocidad total: ", base_env.total_velocity)
    print("Colision total: ", base_env.total_collision)
    print("Oscilation total: ", base_env.total_oscilation)
    print("Distancia total: ", base_env.total_dist)
    print("Dispersion total: ", base_env.total_dispersion)
    print("Recompensa total: ", base_env.total_goal)
    """
    # velocidad_total += base_env.total_velocity
    colision_total += base_env.total_colision
    oscilacion_total += base_env.total_oscilation
    distancia_total += base_env.total_dist
    dispersion_total += base_env.total_dispersion
    recompensa_total += base_env.total_goal


print("\n📊 **Resumen Final** 📊")
print(f"🎯 Objetivo alcanzado {success_count} veces en {EPS} intentos.")
print(f"📈 Tasa de éxito: {success_count / EPS * 100:.2f}%\n")
print("Promedio de acción de: ",action_average / EPS)
print("Promedio de velocidades: ", velocidad_average / EPS)
print("Promedio de pasos: ", steps_average / EPS)
# print("Velocidad: ", velocidad_total / EPS)
print("Colision: ", colision_total / EPS)
print("Oscilation: ", oscilacion_total / EPS)
print("Distancia: ", distancia_total / EPS)
print("Dispersion: ", dispersion_total / EPS)
print("Recompensa: ", recompensa_total / EPS)


env.close()  # Cerrar el entorno después de la prueba
