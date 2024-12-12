from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def train_model(env, timesteps=10000, save_path="models/soccer_model"):
    model = PPO("MlpPolicy", env, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="models/",
                                             name_prefix="soccer_model_checkpoint")
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    model.save(save_path)
    print(f"Modelo guardado en {save_path}")
    return model