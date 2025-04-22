import os
import gymnasium as gym
from stable_baselines3 import SAC
import cv2
import numpy as np
import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Intervalo de modelos a evaluar
modelo_inicio = 78
modelo_final = 82
EPS = 1000  # Número de episodios por modelo

resultados = {}

# Probar modelos SAC_N.zip dentro del rango indicado
for i in range(modelo_inicio, modelo_final + 2):
    nombre_modelo = f"SAC_{i}.zip"
    path_to_model = os.path.join("models", nombre_modelo)

    if not os.path.exists(path_to_model):
        print(f"❌ Modelo {nombre_modelo} no encontrado. Saltando...")
        continue

    print(f"\n🚀 Evaluando modelo: {nombre_modelo}...\n")

    # Crear entorno
    env = gym.make('Ur-v5', render_mode=None)
    model = SAC.load(path_to_model, env=env)

    success_count = 0

    for episode in range(EPS):
        obs = env.reset()[0]
        done = False
        step_count = 0

        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, done, info = env.step(action)
            step_count += 1

            if done and step_count != 100:
                success_count += 1
                break

    tasa_exito = success_count / EPS * 100
    resultados[nombre_modelo] = tasa_exito

    print(f"📊 {nombre_modelo}: {tasa_exito:.2f}% de éxito en {EPS} episodios")

    env.close()

# Mostrar resumen
print("\n📈 Resultados finales por modelo:")
for nombre, tasa in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
    print(f"{nombre}: {tasa:.2f}%")

# Mejor modelo
mejor = max(resultados.items(), key=lambda x: x[1])
print(f"\n🏆 Mejor modelo: {mejor[0]} con {mejor[1]:.2f}% de éxito")
