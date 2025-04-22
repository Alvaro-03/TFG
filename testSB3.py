import gymnasium as gym
from stable_baselines3 import SAC
import cv2
import numpy as np

# Ruta correcta del modelo entrenado
path_to_model = "models/SAC_20.zip"

# Cargar el entorno correcto
env = gym.make('Ur-v3', render_mode='human')  # Asegurarse de que es 'Ur-v3'

# Cargar el modelo entrenado
model = SAC.load(path_to_model, env)

obs = env.reset()[0]
done = False

EPS = 200  # Número de episodios a probar
success_count = 0  # Contador de éxitos

print("\n🚀 **Iniciando prueba del modelo SAC_4 en Ur-v3** 🚀\n")

for episode in range(EPS):
    obs = env.reset()[0]  # Reiniciar el entorno al inicio de cada episodio
    done = False
    step_count = 0  # Contador de pasos en el episodio
    episodio_info = {
        "reward_total": 0,
        "reward_dist": 0,
        "reward_ctrl": 0,
        "reward_orientt": 0,
        "rw_goal": 0,
        "rw_collition": 0
    }
    
    print(episode)

    # while not done:  # Ejecutar hasta que el episodio termine
    for i in range(100):
        action, _ = model.predict(obs)  # Obtener la acción del modelo entrenado
        obs, reward, _, done, info = env.step(action)  # Aplicar la acción en el entorno
        
        env.render()  # Renderizar la simulación
        step_count += 1  # Contar los pasos en el episodio
        
        # Sumar los valores en el episodio actual
        episodio_info["reward_total"] += reward
        episodio_info["reward_dist"] += info.get("reward_dist", 0)
        episodio_info["reward_ctrl"] += info.get("reward_ctrl", 0)
        episodio_info["reward_orientt"] += info.get("reward_orientt", 0)
        episodio_info["rw_goal"] += info.get("rw_goal", 0)
        episodio_info["rw_collition"] += info.get("rw_collition", 0)

        base_env = env.unwrapped 
        dist = np.linalg.norm(base_env.get_body_com("fingertip") - base_env.get_body_com("target1"))
        

        # Verificar si se alcanzó el objetivo (asumiendo que hay una recompensa significativa)
        if dist < 0.005:  # Umbral para considerar que alcanzó el objetivo
            print(np.abs(base_env.angulo1 - base_env.angulo2))
            print(episodio_info["reward_orientt"])
            print(f"✅ Episodio {episode + 1}: Objetivo alcanzado en {step_count} pasos!")
            success_count += 1
            break  # Terminar el episodio al alcanzar el objetivo

    # print(episodio_info)

print("\n📊 **Resumen Final** 📊")
print(f"🎯 Objetivo alcanzado {success_count} veces en {EPS} intentos.")
print(f"📈 Tasa de éxito: {success_count / EPS * 100:.2f}%\n")

env.close()  # Cerrar el entorno después de la prueba