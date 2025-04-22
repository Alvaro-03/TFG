from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np
import os
import torch


class RewardDetailLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        if not infos or not isinstance(infos[0], dict):
            return True

        reward_keys = {
            "reward_dist": "dist",
            "reward_dispersion": "dispersion",
            "rw_goal": "rw_goal",
            "rw_collition": "collition",
            "reward_osc_obj": "osc_obj",
            "reward_par": "par",
            "reward_vel": "vel",
            "reward_total": "total"
        }

        for key_raw, name in reward_keys.items():
            values = [info[key_raw] for info in infos if key_raw in info]
            if values:
                self.logger.record(f"rewards/{name}", np.mean(values))  

        if hasattr(self.model, "replay_buffer"):
            self.logger.record("train/replay_buffer_size", self.model.replay_buffer.size())

        if hasattr(self.model, "lr_schedule"):
            current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
            self.logger.record("train/learning_rate", current_lr)
            
        return True

class FaseLearningRateScheduler(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.lr_fases = {
            0: (2_000_000, 3e-4, 2e-4),
            1: (500_000, 2e-4, 1.5e-4),
            2: (1_000_000, 1.5e-4, 1e-4),
            3: (2_000_000, 1e-4, 5e-5),
            4: (2_000_000, 5e-5, 1e-5)
        }
        self.fase_anterior = -1
        self.inicio = 0 
        self.fin = 0

    def leer_fase_externa(self):
        try:
            with open("control_fase.txt", "r") as file:
                fase = int(file.read().strip())
                return max(0, min(fase, 4))  # limitar a rango válido
        except Exception:
            return None

    def _on_step(self) -> bool:
        total_steps = self.model.num_timesteps
        fase_actual = self.leer_fase_externa()
        if fase_actual is None:
            return True
        
        if self.fase_anterior != fase_actual:
            self.fase_anterior = fase_actual
            self.inicio = total_steps
            self.fin = self.inicio + self.lr_fases[fase_actual][0]
        
        _ , lr_max, lr_min = self.lr_fases[fase_actual]
        
        if total_steps >= self.fin:
            new_lr = lr_min
        else:
            # Interpolación lineal entre max y min en la fase actual
            progress = (total_steps - self.inicio) / (self.fin - self.inicio)
            new_lr = lr_max - progress * (lr_max - lr_min)

        # Aplicar nuevo learning rate
        for param_group in self.model.actor.optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.model.critic.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Para ver en TensorBoard
        self.logger.record("train/learning_rate", new_lr)

        return True

class FaseEntCoefScheduler(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ent_fases = {
            0: (2_000_000, 0.02, 0.01),
            1: (500_000, 0.01, 0.007),
            2: (1_000_000, 0.007, 0.005),
            3: (2_000_000, 0.005, 0.003),
            4: (2_000_000, 0.003, 0.001),
        }

        self.fase_anterior = -1
        self.start = 0 
        self.end = 0

    def leer_fase_externa(self):
        try:
            with open("control_fase.txt", "r") as file:
                fase = int(file.read().strip())
                return max(0, min(fase, 4))  # limitar entre 0 y 4
        except Exception:
            return None

    def _on_step(self) -> bool:
        total_steps = self.model.num_timesteps
        fase_actual = self.leer_fase_externa()
        if fase_actual is None:
            return True

        if self.fase_anterior != fase_actual:
            self.fase_anterior = fase_actual
            self.start = total_steps
            self.end = self.start + self.ent_fases[fase_actual][0]

        _ , ent_max, ent_min = self.ent_fases[fase_actual]

        if total_steps >= self.end:
            new_ent = ent_min
        else:
            # Escalado exponencial entre ent_max y ent_min
            normalized_progress = (total_steps - self.start) / (self.end - self.start)
            decay_factor = np.exp(-5 * normalized_progress)  # Ajusta el "5" para cambiar la curvatura
            new_ent = ent_min + (ent_max - ent_min) * decay_factor

        # Actualizar ent_coef en el modelo
        if hasattr(self.model, "log_ent_coef"):
            self.model.ent_coef_tensor = self.model.log_ent_coef.exp().detach().clone()
            self.model.log_ent_coef.data = new_ent.log() if hasattr(new_ent, 'log') else torch.tensor(np.log(new_ent), dtype=torch.float32, device=self.model.device)

        # Registrar en tensorboard
        self.logger.record("train/ent_coef", new_ent)
        return True