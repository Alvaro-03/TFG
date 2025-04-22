import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

MODELS_DIR = "models"
CONTROL_FASE_PATH = "control_fase.txt"
ULTIMO_MODELO_PATH = "ultimo_modelo_probado.txt"
EVAL_EPISODES = 4000
CHECK_INTERVAL = 30  # segundos
UMBRALES = [99.0,99.25,99.5,99.75,100]

def cargar_fase_actual():
    try:
        with open(CONTROL_FASE_PATH, "r") as f:
            return int(f.read().strip())
    except:
        return 0

def actualizar_fase(nueva_fase):
    with open(CONTROL_FASE_PATH, "w") as f:
        f.write(str(nueva_fase))

def cargar_ultimo_modelo_probado():
    if not os.path.exists(ULTIMO_MODELO_PATH):
        return None
    with open(ULTIMO_MODELO_PATH, "r") as f:
        return f.read().strip()

def guardar_ultimo_modelo_probado(nombre):
    with open(ULTIMO_MODELO_PATH, "w") as f:
        f.write(nombre)

def evaluar_modelo(model_path, fase_actual):
    print(f"\n🔍 Probando nuevo modelo: {model_path}")
    env = gym.make("Ur-v5", render_mode=None)
    model = SAC.load(model_path, env)
    success_count = 0

    for ep in range(EVAL_EPISODES):
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

    tasa_exito = success_count / EVAL_EPISODES * 100
    print(f"\n📈 Tasa de éxito del modelo: {tasa_exito:.2f}% ({success_count}/{EVAL_EPISODES})")

    env.close()

    if tasa_exito >= UMBRALES[fase_actual] and fase_actual < 4:
        nueva_fase = fase_actual + 1
        print(f"✅ ¡Modelo supera el {tasa_exito:.2f}% de éxito! Avanzando a la fase {nueva_fase}")
        actualizar_fase(nueva_fase)

def extraer_version(modelo_nombre):
    try:
        return int(modelo_nombre.replace("SAC_", "").replace(".zip", ""))
    except ValueError:
        return -1  # Por si hay archivos que no siguen el formato esperado

def main():
    print("🚀 Monitoreando nuevos modelos...")
    ultimo_probado = cargar_ultimo_modelo_probado()
    version_ultima = extraer_version(ultimo_probado) if ultimo_probado else -1

    while True:
        fase_actual = cargar_fase_actual()

        # Buscar modelos ordenados por versión (no por fecha)
        modelos = sorted(
            [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")],
            key=extraer_version,
            reverse=True
        )

        if modelos:
            modelo_reciente = modelos[0]
            version_reciente = extraer_version(modelo_reciente)

            if version_reciente > version_ultima:
                evaluar_modelo(os.path.join(MODELS_DIR, modelo_reciente), fase_actual)
                guardar_ultimo_modelo_probado(modelo_reciente)
                version_ultima = version_reciente

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
