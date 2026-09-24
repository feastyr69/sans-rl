import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env.sans_env import SansEnv

LOG_DIR = "./logs/"
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_DIR = "./trained_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env():
    def _init():
        env = SansEnv(render_mode="rgb_array")
        return env
    return _init

def main():
    # Use DummyVecEnv and VecFrameStack to stack 4 frames
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)

    # Initialize PPO model with CnnPolicy (since we are using frames)
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    
    print("Starting training...")
    # Train for 1,000 timesteps as requested for the initial test
    model.learn(total_timesteps=1000)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, "ppo_sans")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()

if __name__ == "__main__":
    main()
