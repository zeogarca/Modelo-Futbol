import gym
from gym import spaces
import numpy as np

class SoccerEnv(gym.Env):
    def __init__(self):
        super(SoccerEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                            high=np.array([100, 100, 100, 100]),
                                            dtype=np.float32)
        self.state = None
        self.done = False

    def reset(self):
        self.state = np.random.uniform(0, 100, size=(4,))
        self.done = False
        return self.state

    def step(self, action):
        team_stats = self.state
        opponent_stats = np.random.uniform(0, 100, size=(4,))
        match_result = np.sum(team_stats) - np.sum(opponent_stats) + action * 5

        reward = 10 if match_result > 0 else 5 if match_result == 0 else -10
        self.state = np.random.uniform(0, 100, size=(4,))
        self.done = True
        return self.state, reward, self.done, {}

    def render(self, mode="human"):
        print(f"Estado actual del equipo: {self.state}")

    def close(self):
        print("Entorno cerrado.")