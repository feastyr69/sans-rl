import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env.sans_env import SansEnv

MODEL_PATH = "./trained_models/ppo_sans.zip"

def make_env():
    def _init():
        # Using headless=False for evaluation to watch the agent
        env = SansEnv(render_mode="human")
        return env
    return _init

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)
    
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
        return

    obs = env.reset()
    print("Starting evaluation...")
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # In dummy env, done is an array
        if done[0]:
            print("Episode finished. Resetting...")
            obs = env.reset()
            break
            
    print("Closing environment...")
    env.close()

if __name__ == "__main__":
    main()
