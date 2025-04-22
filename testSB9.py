import gymnasium as gym
from stable_baselines3 import SAC, PPO
import cv2
import numpy as np
import pandas as pd
import random
from torch import nn 
import torch
import time

# Ruta correcta del modelo entrenado
path_to_model = "models/SAC_80.zip"

SEED = 42
CHECK_INTERVAL = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Cargar el entorno correcto
# env = gym.make('Ur-v5', render_mode=None)  # Asegurarse de que es 'Ur-v3'
env = gym.make('Ur-v5', render_mode="human")  # Asegurarse de que es 'Ur-v3'

# Cargar el modelo entrenado
model = SAC.load(path_to_model, env)

obs = env.unwrapped.reset(seed=SEED)

done = False
imagen_guardada = False

EPS = 4000  # Número de episodios a probar
success_count = 0  # Contador de éxitos
prev_action = None
prev_pose = None
action_eps = 0
action_actual = 0
max_action = 0
total_action = 0
porcentaje_osc = 0
porcentaje_vel = 0
porcentaje_disp = 0
porcenatje_dist = 0
porcentaje_par = 0
porcenatje_ratio = 0

# Variables de media para calcular porcentajes
media_osc_pos = 0
media_osc_obj = 0
media_vel = 0
media_disp = 0
media_dist = 0
media_par = 0
media_par_vel = 0
media_ctrl = 0
media_pasos = 0

# Inicialización de variables acumulativas
total_osc_pos = 0  # Oscilación en posición
total_osc_obj = 0  # Oscilación en el objetivo
total_vel = 0  # Velocidad
total_disp = 0  # Dispersión
total_dist = 0  # Distancia
total_par = 0  # Par (torque)
total_par_vel = 0  # Par-Velocidad
total_ctrl = 0  # Control articular
total_reward = 0  # Recompensa total
total_acumulation = 0
episode = 0

min_distance = np.inf

# Historial de recompensas y penalizaciones
desglose_recompensas = []

print("\n🚀 **Iniciando prueba del modelo SAC_4 en Ur-v3** 🚀\n")

for episode in range(EPS):
# while True:
    # episode += 1
    obs = env.reset()[0]  # Reiniciar el entorno al inicio de cada episodio
    done = False
    step_count = 0  # Contador de pasos en el episodio
    episodio_info = {
        "reward_total": 0,
        "reward_dist": 0,
        "rw_goal": 0,
        "rw_collition": 0,
        "reward_dispersion": 0,
        "reward_osc_obj": 0,
        "reward_vel": 0, 
        "reward_par": 0
    }
    action_eps = 0
    
    print(episode)
    base_env = env.unwrapped 
    prev_action = None

    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)  # Obtener la acción del modelo entrenado
        obs, reward, _, done, info = env.step(action)  # Aplicar la acción en el entorno

        env.render()  # Renderizar la simulación
        step_count += 1  # Contar los pasos en el episodio

        if done == True and step_count!=100:
            print(f"✅ Episodio {episode}: Plano alineado en {step_count} pasos!")
            success_count += 1
            break  # Terminar el episodio

    # time.sleep(CHECK_INTERVAL)
    
    # print(episodio_info)
    """
    if episode % 100 == 0:
        print(episode)
        print(f"📈 Tasa de éxito: {success_count / episode * 100:.2f}%\n")
    """
    if done == True and step_count!=100:
        total_osc_obj += info.get("reward_osc_obj", 0) / step_count
        total_vel += info.get("reward_vel", 0) / step_count
        total_disp += info.get("reward_dispersion", 0) / step_count
        total_dist += info.get("reward_dist", 0) / step_count
        total_par += info.get("reward_par", 0) / step_count


    if episode % 100 == 0:
        print("Oscilación media de: ", total_osc_obj/EPS)
        print("Velocidad media de: ", total_vel/EPS)
        print("Dispersión media de: ", total_disp/EPS)
        print("Distancia media de: ", total_dist/EPS)
        print("Par media de: ", total_par/EPS)
    
print("\n📊 **Resumen Final** 📊")
print(f"🎯 Objetivo alcanzado {success_count} veces en {EPS} intentos.")
print(f"📈 Tasa de éxito: {success_count / EPS * 100:.2f}%\n")

print("Oscilación media de: ", total_osc_obj/EPS)
print("Velocidad media de: ", total_vel/EPS)
print("Dispersión media de: ", total_disp/EPS)
print("Distancia media de: ", total_dist/EPS)
print("Par media de: ", total_par/EPS)

env.close()  # Cerrar el entorno después de la prueba
