import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env.sans_env import SansEnv
from stable_baselines3.common.callbacks import BaseCallback

LOG_DIR = "./logs/"
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_DIR = "./trained_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.steps_alive = []
        self.current_reward = 0
        self.episodes = 0
        self.success_rate_window = []
        
    def _on_step(self):
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        info = self.locals['infos'][0]
        
        self.current_reward += reward
        if done:
            steps = info.get('steps_alive', 0)
            self.episodes += 1
            
            # An attack usually lasts a fixed duration (e.g. 150-300 frames)
            # If it survived > 200 frames, count it as a success for attack 1.
            is_success = steps > 150 
            self.success_rate_window.append(is_success)
            if len(self.success_rate_window) > 20:
                self.success_rate_window.pop(0)
                
            success_rate = sum(self.success_rate_window) / len(self.success_rate_window) * 100
            
            print(f"Ep {self.episodes} finished! Reward: {self.current_reward:.2f} | Frames Alive: {steps} | Success Rate (last 20): {success_rate:.1f}%")
            self.episode_rewards.append(self.current_reward)
            self.steps_alive.append(steps)
            self.current_reward = 0
        return True

import threading
import http.server
import socketserver
import urllib.request

def start_local_server():
    # Try to connect to see if it's already running
    try:
        urllib.request.urlopen("http://localhost:8081/", timeout=1)
        return # Already running
    except:
        pass
        
    def run_server():
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "c2-sans-fight"))
        Handler = http.server.SimpleHTTPRequestHandler
        # Suppress logging
        class QuietHandler(Handler):
            def log_message(self, format, *args):
                pass
        with socketserver.ThreadingTCPServer(("", 8081), QuietHandler) as httpd:
            httpd.allow_reuse_address = True
            httpd.serve_forever()
            
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2) # Give it time to start

def make_env(rank, attack_index, render=False):
    def _init():
        # If render is True, we run headful for the first environment (rank 0)
        is_headless = not render if rank == 0 else True
        env = SansEnv(render_mode=None, connect_remote=False, headless=is_headless, attack_index=attack_index)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train Sans RL Agent")
    parser.add_argument("--attack", type=int, default=1, help="Attack index for curriculum learning (1=First jump, 2=Blue bones...)")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train")
    parser.add_argument("--render", action="store_true", help="Show the browser to watch the agent learn")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments to run")
    args = parser.parse_args()

    start_local_server()

    num_envs = args.envs
    if num_envs > 1:
        env = SubprocVecEnv([make_env(i, args.attack, args.render) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.attack, args.render)])

    # Initialize PPO model with MlpPolicy (10x faster than CNN, uses coordinate vector)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0003, n_steps=2048, batch_size=64)
    
    print(f"Starting Curriculum Training on Attack {args.attack}...")
    
    callback = CurriculumCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    
    model_path = os.path.join(MODEL_DIR, f"ppo_sans_attack_{args.attack}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()

if __name__ == "__main__":
    main()
