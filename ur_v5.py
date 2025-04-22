__credits__ = ["Luis_Ortiz"]

from typing import Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import cv2
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation
import math 

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}

class UrEnvF(MujocoEnv, utils.EzPickle):
        
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "ur1.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 0.01,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_control_weight,
            **kwargs,
        )

        self.pos_xy = np.array([ 0, 0, 0, 0, 0, 0, 0, 0])
        self.pos_xyT = np.array([ 0, 0, 0, 0, 0, 0, 0, 0])
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight
        self.rw_goal_C = 0
        self.disconut = 0
        self.radius = 48
        self.prev_action = None
        self.prev_pos = None
        self.fase_rotacion = 0
        self.num_puntos_detectados = 0 
        self.bool_detect = False
        self.episodic_reward_dist = 0.0
        self.episodic_reward_vel = 0.0
        self.episodic_reward_par = 0.0
        self.episodic_rw_goal = 0.0
        self.episodic_reward_dispersion = 0.0
        self.episodic_reward_osc_obj = 0.0
        self.episodic_rw_collition = 0.0
        self.episodic_reward = 0.0
        self.plano_actual = np.zeros((4, 3))
        self.probabilidad_perturbacion = 0.5

        self.action_scale = 0.25

        observation_space = Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def check_collision(self):
        # Obtener posiciones de los cuerpos relevantes
        ground_height = 0.08
        
        link_1 = self.get_body_com("forearm_link")
        link_2 = self.get_body_com("wrist_1_link")
        link_3 = self.get_body_com("wrist_2_link")
        link_4 = self.get_body_com("wrist_3_link")

        
        if link_1[2] < ground_height or link_2[2] < ground_height or link_3[2] < ground_height or link_4[2] < ground_height:
            return True  # Hay una colisión
        return False  # No hay colisión

    def leer_fase_externa(self):
        """
        Lee el número de fase desde un archivo externo.
        Si el archivo no existe o contiene un valor inválido, devuelve la fase actual.
        """
        try:
            with open("control_fase.txt", "r") as file:
            # with open("control_fase_2.txt", "r") as file:
                fase = int(file.read().strip())  # Leer y convertir a entero
                return max(0, min(fase, 4))  # Asegurar que el valor esté entre 0 y 4
        except Exception:
            return None  # Si hay un error, no cambiar la fase

    def step(self, action):
        done = False
        self.num_puntos_detectados = 0 

        qpos_actual = self.data.qpos[:6].copy()
        delta_q = action * self.action_scale
        q_target = qpos_actual + delta_q
        # print("Posición actual: ", qpos_actual)
        # print("Diferencia: ", delta_q)
        # print("Target: ", q_target)
        # print("Acción: ", action)
        self.do_simulation(q_target, self.frame_skip)

        self.pos_xy = np.array([ 0, 0, 0, 0, 0, 0, 0, 0])
        
        env_render, rgb_array = self.render()

        if rgb_array is not None:
            # Convertir la imagen de RGB a BGR para compatibilidad con OpenCV
            rgb_array_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # Convertir la imagen a espacio de color HSV para facilitar la detección de colores
            imagen_hsv = cv2.cvtColor(rgb_array_bgr, cv2.COLOR_BGR2HSV)

            # Definir los rangos de colores en HSV
            color_ranges = {
                "rojo": ([0, 100, 50], [10, 255, 255]),
                "verde": ([10, 100, 50], [75, 255, 255]),
                "azul": ([115, 140, 50], [130, 255, 255]),
                "lila": ([131, 70, 0], [170, 255, 255])
            }
            
            # Crear máscaras y encontrar contornos para cada color
            contornos = {}
            for color, (bajo, alto) in color_ranges.items():
                mascara = cv2.inRange(imagen_hsv, np.array(bajo), np.array(alto))
                contornos[color], _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Color y radio de los círculos de detección
            color_deteccion = (255, 255, 255)  # Blanco
            radio_circulo = 8
            
            # Detectar y marcar cada color en la imagen
            for i, (color, lista_contornos) in enumerate(contornos.items()):
                if lista_contornos:
                    x, y, w, h = cv2.boundingRect(lista_contornos[0])
                    centro = [int(x + w / 2), int(y + h / 2)]
                    self.pos_xy[i * 2 : (i + 1) * 2] = centro
                    cv2.circle(rgb_array_bgr, tuple(centro), radio_circulo, color_deteccion, 2)
                    self.num_puntos_detectados += 1

            # Definir posiciones de referencia de los puntos objetivo
            puntos_objetivo = {
                "rojo": [106, 347],
                "verde": [382, 74],
                "azul": [106, 74],
                "lila": [382, 347]
            }
            
            # Convertir los puntos objetivo en una lista
            self.pos_xyT = np.array(sum(puntos_objetivo.values(), []))
            
            # Ajustar radio de visualización
            self.disconut += 0.8
            self.radius = max(24, int(50 - self.disconut))
            
            # Dibujar los círculos de referencia en la imagen
            color_referencia = (255, 0, 120)  # Color rosa en formato BGR
            grosor = 2
            if self.render_mode == "human":
                for punto in puntos_objetivo.values():
                    cv2.circle(rgb_array_bgr, tuple(punto), self.radius, color_referencia, grosor)
                
                # Mostrar la imagen procesada con los puntos detectados y los puntos de referencia
                cv2.imshow('Entorno', rgb_array_bgr)
                cv2.waitKey(1)

        
        reward, reward_info, done = self._get_rew(action)
        self.perturbar_plano_durante_episode()
        observation = self._get_obs()
        info = reward_info
        
        return observation, reward, False, done, info

    def distancia_euclidiana(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _get_rew(self, action):
        reward = 0
        done = False

        # Pesos
        rw_goal = 0
        dist_rw = 0.625 # 0.625 1         
        dispersion_rw = 4.2 # 4.2 10     
        par_rw = 0.000005 # 0.000005 0.0000125     
        velocity_rw = 0.002 # 0.002 0.004      
        osc_obj_rw = 0.0005 # 0.0005 0.002    
        
        # Penalización por colisión
        rw_collition = -1 if self.check_collision() else 0

        # ------------------ DISTANCIA ENTRE PUNTOS ------------------
        distancia_eu = np.linalg.norm(self.pos_xyT.reshape(-1, 2) - self.pos_xy.reshape(-1, 2), axis=1)
        distancia_sub_normalizada = distancia_eu / 480.0 
        # print("Distancia: ", distancia_sub_normalizada) 
        reward_dist = -np.sum(distancia_sub_normalizada * dist_rw)
        # print("Distancia: ", reward_dist)

        # ------------------ DISPERSIÓN ENTRE DISTANCIAS ------------------
        desviacion_dist = np.std(distancia_sub_normalizada)
        # print("Dispersión: ", desviacion_dist)
        reward_dispersion = -desviacion_dist * dispersion_rw
        # print("Dispersión: ", reward_dispersion)

        # ------------------ PAR ------------------
        par = np.square(self.data.qfrc_actuator[:6]).sum()
        # print("Par: ", self.data.qfrc_actuator[:6])
        reward_par = -par * par_rw
        # print("Par: ", reward_par)

        # ------------------ VELOCIDAD ------------------
        velocity = np.square(self.data.qvel[:6]).sum()
        # print("Velocidad: ", self.data.qvel[:6])
        reward_vel = -velocity * velocity_rw
        # print("Velocidad: ", reward_vel)
        
        # ------------------ OSCILACIÓN ------------------
        oscillation_obj = 0 if self.prev_action is None else np.square(action - self.prev_action).sum()
        # print("Oscilación: ", oscillation_obj)
        reward_osc_obj = -oscillation_obj * osc_obj_rw
        # print("Oscilación: ", reward_osc_obj)
        self.prev_action = action.copy()

        # ------------------ RECOMPENSA ------------------
        reward_progresivo = 0
        puntos_dentro_rango = 0
        radio_normalizado = self.radius / 480.0

        for d in distancia_sub_normalizada:
            if d < radio_normalizado:
                reward_progresivo += (radio_normalizado - d)
                puntos_dentro_rango += 1

        rw_goal = reward_progresivo * puntos_dentro_rango
        """
        if puntos_dentro_rango == 4:
            done = True
        """
        # print("Recompensa: ", rw_goal)
        # print("\n")

        # Sumar todas las recompensas y penalizaciones
        reward += rw_goal + reward_dist + reward_dispersion + reward_osc_obj + rw_collition + reward_par + reward_vel

        # Guardamos la suma de las penalizaciones individualmente durante un episodio
        self.episodic_reward_dist += reward_dist
        self.episodic_reward_vel += reward_vel
        self.episodic_reward_par += reward_par
        self.episodic_rw_goal += rw_goal
        self.episodic_reward_dispersion += reward_dispersion
        self.episodic_reward_osc_obj += reward_osc_obj
        self.episodic_rw_collition += rw_collition
        self.episodic_reward += reward

        """
        if abs(reward_dist+reward_dispersion) < abs(reward_par+reward_vel+reward_osc_obj) and done==False:
            print("\n🚀 **PROBLEMAS** 🚀\n")
            print("Diferencia: ", (reward_dist+reward_dispersion)-(reward_par+reward_vel+reward_osc_obj))
            print("Penalización por distancia: ", reward_dist)
            print("Penalización por dispersión: ", reward_dispersion)
            print("Penalización por par: ", reward_par)
            print("Penalización por velocidad: ", reward_vel)
            print("Penalización por oscilación: ", reward_osc_obj)
            print("Distancia: ", np.sum(distancia_sub_normalizada))
            print("Dispersión: ", desviacion_dist)
            print("Par: ", par)
            print("Velocidad: ", velocity)
            print("Oscilación: ", oscillation_obj)
        """

        # Guardar los valores desglosados para análisis posterior
        reward_info = {
            "reward_total": self.episodic_reward,
            "reward_dist": self.episodic_reward_dist,
            "reward_dispersion": self.episodic_reward_dispersion,
            "rw_goal": self.episodic_rw_goal,
            "rw_collition": self.episodic_rw_collition,
            "reward_osc_obj": self.episodic_reward_osc_obj,
            "reward_par": self.episodic_reward_par,
            "reward_vel": self.episodic_reward_vel
        }
        
        return reward, reward_info, done

    def rotacion_eje_z(self, puntos):
        nueva_fase = self.leer_fase_externa()
        if nueva_fase is not None and nueva_fase != self.fase_rotacion:
            print(f"🚀 Fase de rotación cambiada a {nueva_fase} manualmente")
            self.fase_rotacion = nueva_fase

        if self.fase_rotacion <= 1:
            angZ = angX = angY = 0
        elif self.fase_rotacion == 2:
            angZ = np.deg2rad(np.random.uniform(-25, 25))
            angX = angY = 0
        elif self.fase_rotacion >= 3:
            angZ = np.deg2rad(np.random.uniform(-25, 25))
            angX = np.deg2rad(np.random.uniform(-10, 10))
            angY = np.deg2rad(np.random.uniform(-10, 10))

        Rx = np.array([[1, 0, 0], [0, np.cos(angX), -np.sin(angX)], [0, np.sin(angX), np.cos(angX)]])
        Ry = np.array([[np.cos(angY), 0, np.sin(angY)], [0, 1, 0], [-np.sin(angY), 0, np.cos(angY)]])
        Rz = np.array([[np.cos(angZ), -np.sin(angZ), 0], [np.sin(angZ), np.cos(angZ), 0], [0, 0, 1]])

        puntos_rotados = puntos @ Rz.T @ Rx.T @ Ry.T

        return puntos_rotados

    def relacion_espacial(self):
        pos_cam = np.array([0.27942594, 0.13399861, 0.76105142])
        euler_cam = np.array([-1.57079255, 3.67163886e-06, -1.54159265])
        rot_cam = Rotation.from_euler('xyz', euler_cam).as_matrix()

        T_cam = np.eye(4)
        T_cam[:3, :3] = rot_cam
        T_cam[:3, 3] = pos_cam

        pos_sphere = np.array([0.325, 0.127, 0.3])
        T_sphere = np.eye(4)
        T_sphere[:3, 3] = pos_sphere

        T_rel = np.linalg.inv(T_cam) @ T_sphere

        posEF = self.get_body_com("fingertip")
        quatEF = self.data.body('fingertip').xquat
        rotEF = Rotation.from_quat(quatEF)
        euEF = rotEF.as_euler('xyz', degrees=False)

        rot_cam_new = Rotation.from_euler('xyz', euEF).as_matrix()
        T_cam_new = np.eye(4)
        T_cam_new[:3, :3] = rot_cam_new
        T_cam_new[:3, 3] = posEF

        T_sphere_new = T_cam_new @ T_rel
        new_sphere_pos = T_sphere_new[:3, 3]

        return new_sphere_pos

    def generar_punto_en_esfera(self):
        done = False
        x = y = z = 0
        t_c = 0.1
        p_al = np.array([[x - t_c/2, y - t_c/2, z], [x + t_c/2, y - t_c/2, z], [x + t_c/2, y + t_c/2, z], [x - t_c/2, y + t_c/2, z]])
        plano_rotado = self.rotacion_eje_z(p_al)

        centro_sphere = self.relacion_espacial()

        # Traslaciones aleatorias
        for _ in range(1000): 
            traslacion = np.random.uniform(-0.3, 0.3, size=3)
            plano_trasladado = plano_rotado + centro_sphere + traslacion

            centro = plano_trasladado.mean(axis=0)
            dist_superficie_esfera = np.linalg.norm(centro - centro_sphere)
            dist_al_objetivo = np.linalg.norm(centro - np.array([0, 0, 0.163]))
            dist_perpendicular = np.linalg.norm(centro[:2] - np.array([0, 0]))

            if (dist_superficie_esfera < 0.23 and
                0.32 <= dist_al_objetivo < 0.58 and dist_perpendicular >= 0.32 and 
                centro[2] > 0.17 and centro[0] > 0.07):
                done = True
                break

        self.plano_actual = plano_trasladado
        return plano_trasladado.flatten(), done

    def perturbar_plano_durante_episode(self):
        if self.fase_rotacion != 4:
            return

        if not hasattr(self, 'plano_actual'):
            return
        
        if np.random.rand() > self.probabilidad_perturbacion:
            return

        traslacion = np.random.uniform(-0.002, 0.002, size=3)
        ang_peq = np.deg2rad(np.random.uniform(-0.3, 0.3, size=3))  # rotación pequeña en X, Y, Z

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ang_peq[0]), -np.sin(ang_peq[0])],
            [0, np.sin(ang_peq[0]), np.cos(ang_peq[0])]
        ])
        Ry = np.array([
            [np.cos(ang_peq[1]), 0, np.sin(ang_peq[1])],
            [0, 1, 0],
            [-np.sin(ang_peq[1]), 0, np.cos(ang_peq[1])]
        ])
        Rz = np.array([
            [np.cos(ang_peq[2]), -np.sin(ang_peq[2]), 0],
            [np.sin(ang_peq[2]),  np.cos(ang_peq[2]), 0],
            [0, 0, 1]
        ])

        R_total = Rz @ Rx @ Ry
        plano_transformado = (self.plano_actual @ R_total.T) + traslacion
        self.plano_actual = plano_transformado

        qpos = self.data.qpos.copy()
        qpos[-12:] = plano_transformado.flatten()
        self.set_state(qpos, self.data.qvel.copy())

    def reset_model(self):
        self.prev_action = None

        while True:
            qpos = np.array([-1.5708, -1.8, 1, -0.80, -1.5708, 0] + [0]*12)
            nueva_fase = self.leer_fase_externa()
            if nueva_fase is not None and nueva_fase != self.fase_rotacion:
                print(f"🚀 Fase de rotación cambiada a {nueva_fase} manualmente")
                self.fase_rotacion = nueva_fase

            if self.fase_rotacion >= 1:
                variabilidad = np.array([0.3, 0.2, 0.2, 0.2, 0.1, 0.0])
                qpos[:6] += np.random.uniform(-variabilidad, variabilidad)

            qvel = np.zeros(18)
            self.set_state(qpos, qvel)

            p_f, done = self.generar_punto_en_esfera()
            qpos[-12:] = p_f
            self.set_state(qpos, qvel)
            _, _, _, _, info = self.step(np.zeros(self.action_space.shape))

            if self.num_puntos_detectados == 4 and done == True:
                break
            else:
                print("Descartado")

        self.prev_action = np.zeros(self.action_space.shape)
        self.disconut = 0
        self.episodic_reward_dist = 0.0
        self.episodic_reward_vel = 0.0
        self.episodic_reward_par = 0.0
        self.episodic_rw_goal = 0.0
        self.episodic_reward_dispersion = 0.0
        self.episodic_reward_osc_obj = 0.0
        self.episodic_rw_collition = 0.0
        self.episodic_reward = 0.0

        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos.flatten()[:6]
        qvel = self.data.qvel.flatten()[:6]
        posEF = self.get_body_com("fingertip")
        quatEF = self.data.body('fingertip').xquat
        rotEF = Rotation.from_quat(quatEF)
        euEF = rotEF.as_euler('xyz', degrees=False)
        puntos_vector1 = self.pos_xyT.reshape(-1, 2)
        puntos_vector2 = self.pos_xy.reshape(-1, 2)
        dd = np.linalg.norm(puntos_vector1 - puntos_vector2, axis=1)
        action = self.prev_action
        obss = np.concatenate(
            [
                np.cos(qpos), np.sin(qpos),
                qvel,
                posEF,           
                euEF,                      
                self.pos_xy,                    
                self.pos_xyT, 
                dd,                      
                action,
            ])

        return obss
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return super().reset(seed=seed, options=options)
     
