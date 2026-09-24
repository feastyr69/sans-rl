import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from env.sans_env import SansEnv

LOG_DIR = "./logs/"
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_DIR = "./trained_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(rank):
    def _init():
        # Set connect_remote=False for isolated envs
        # Make the first environment (rank 0) visible so you can watch it train!
        is_headless = (rank != 0)
        env = SansEnv(render_mode=None, connect_remote=False, headless=is_headless)
        return env
    return _init

from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0

    def _on_step(self):
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        self.current_reward += reward
        if done:
            print(f"Episode finished! Total Reward: {self.current_reward:.2f}")
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        return True

def main():
    # Use SubprocVecEnv to run 1 parallel environment for visual testing
    num_envs = 1
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecFrameStack(env, n_stack=4)

    # Initialize PPO model with CnnPolicy (since we are using frames)
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    
    print("Starting training...")
    # Train for 1,000 timesteps with our custom callback to show logs
    callback = CustomCallback()
    model.learn(total_timesteps=1000, callback=callback)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "ppo_sans")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()

if __name__ == "__main__":
    main()
