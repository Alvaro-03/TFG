import gymnasium as gym
import os
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from reward_logger import RewardDetailLogger, FaseLearningRateScheduler, FaseEntCoefScheduler
import random
import numpy as np
from torch import nn 
import torch
from datetime import datetime

callbacks = CallbackList([
    RewardDetailLogger(),
    FaseLearningRateScheduler(),
    FaseEntCoefScheduler()
])

# Configuración de carpetas
model_dir = "models"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("logs", f"run_{timestamp}")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Semilla para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================= ENTORNOS VECTORIALIZADOS =======================

NUM_ENVS = 4

def make_env(seed):
    def _init():
        env = gym.make('Ur-v5', render_mode=None)
        env = Monitor(env) 
        env.reset(seed=seed)
        return env
    return _init

# =========================================================================

# Hiperparámetros de red
policy_kwargs = dict(
    activation_fn=nn.ELU,
    net_arch=[256, 256]
)


if __name__ == "__main__":

    env = SubprocVecEnv([make_env(SEED + i) for i in range(NUM_ENVS)])

    # Algoritmo de entrenamiento
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        # learning_rate= 3e-4, # CONFIRMADO
        # ent_coef= 0.01, # CONFIRMADO
        gamma=0.99, # CONFIRMADO
        seed=SEED, # CONFIRMADO
        train_freq=4, # CONFIRMADO
        gradient_steps=4, # CONFIRMADO
        batch_size=1024, # CONFIRMADO
        learning_starts=50000, # CONFIRMADO
        buffer_size=1_500_000, # CONFIRMADO
        verbose=1,
        device="cuda",
        tensorboard_log=log_dir
    )


    """
    model = PPO(
        policy="MlpPolicy",
        env=env,
        # policy_kwargs=policy_kwargs,
        gamma=0.99,
        seed=SEED,
        n_steps=2048,       # 512 pasos por entorno si usas 4 entornos
        batch_size=1024,    # igual que SAC
        n_epochs=4,         # imita gradient_steps=4
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.01,      # mismo valor que en SAC
        verbose=1,
        device="cuda",
        tensorboard_log=log_dir
    )
    """

    print("Dispositivo usado:", model.device)
    print(model.policy)

    # Entrenamiento y guardado periódico
    TIMESTEPS = 50000
    SAVE_INTERVAL = 100000
    iters = 0

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=callbacks, progress_bar=True)

        if iters % (SAVE_INTERVAL // TIMESTEPS) == 0:
            model.save(f"{model_dir}/SAC_{iters}")
