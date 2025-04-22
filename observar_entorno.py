import gymnasium as gym
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Crear el entorno con renderización activada
env = gym.make("Ur-v6", render_mode="human")
obs = env.reset()
base_env = env.unwrapped 

posEF = base_env.get_body_com("fingertip")
quatEF = base_env.data.body('fingertip').xquat
rotEF = Rotation.from_quat(quatEF)
euEF = rotEF.as_euler('xyz', degrees=False)


print("\n🚀 **Simulación iniciada.**")
print("🔹 El entorno está activo, pero sin control manual.")

print("Posición: ", posEF)
print("Orietnación: ", euEF)

# Crear la ventana antes del bucle
cv2.namedWindow("Vista de la Cámara del Robot", cv2.WINDOW_NORMAL)

running = True
while running:
    # Ejecutar un paso en el entorno sin aplicar ninguna acción
    # obs, reward, done, _, _ = env.step(np.zeros(6))
    
    # Renderizar solo la cámara del extremo del robot
    env_render, rgb_array  = env.render()

    if rgb_array is not None:
        rgb_array_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        cv2.imshow("Vista de la Cámara del Robot", rgb_array_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
    
# Cerrar ventanas al salir
cv2.destroyAllWindows()
env.close()

print("\n🚀 **Simulación finalizada.**\n")
