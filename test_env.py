import time
from env.sans_env import SansEnv

def main():
    print("Initializing environment...")
    # Setting render_mode="human" will launch Playwright with headless=False so you can see the game.
    env = SansEnv(render_mode="human")
    print("Environment initialized.")
    
    obs, info = env.reset()
    print(f"Reset complete. Initial observation shape: {obs.shape}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}, Obs Shape={obs.shape}")
        if done:
            print("Episode finished.")
            break
            
    print("Closing environment...")
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
