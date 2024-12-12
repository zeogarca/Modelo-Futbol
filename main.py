from soccer_env import SoccerEnv
from train_model import train_model

if __name__ == "__main__":
    env = SoccerEnv()
    model = train_model(env, timesteps=10000, save_path="models/soccer_model")
    obs = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render()
    env.close()