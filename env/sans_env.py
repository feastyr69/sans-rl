import gymnasium as gym
import numpy as np
import cv2
from playwright.sync_api import sync_playwright
import time
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import URL, TARGET_WIDTH, TARGET_HEIGHT, ACTIONS, REWARD_SURVIVAL, REWARD_DEATH

class SansEnv(gym.Env):
    """Custom Environment that follows gym interface for Sans Boss Fight"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, mode="normal", attack_index=1, connect_remote=True, headless=True):
        super().__init__()
        self.render_mode = render_mode
        self.game_mode = mode
        self.attack_index = attack_index
        self.connect_remote = connect_remote
        self.headless = headless
        
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(TARGET_HEIGHT, TARGET_WIDTH, 1), dtype=np.uint8
        )

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        self.last_hp = 100
        self.death_counter = 0

        self._start_browser()

    def _start_browser(self):
        self.playwright = sync_playwright().start()
        
        if self.connect_remote:
            try:
                # Connect to the manually launched browser
                self.browser = self.playwright.chromium.connect_over_cdp("http://localhost:9222")
                self.context = self.browser.contexts[0]
                self.page = self.context.pages[0]
                print("Connected to manual browser instance!")
                return
            except Exception as e:
                print("No manual browser found on port 9222. Launching optimized headless instance...")
                
        # Local instance (either explicitly requested or fallback)
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-frame-rate-limit", "--disable-gpu-vsync"]
        )
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.page.goto(URL)
        self.page.wait_for_selector('canvas')
        
        print("Browser instance ready! Waiting 10 seconds for game to load...")
        time.sleep(10)



    def _get_frame_and_hp(self):
        try:
            # Force wait for WebGL to render to prevent visual glitches (black frames)
            self.page.evaluate('() => new Promise(requestAnimationFrame)')
            canvas = self.page.locator('canvas')
            screenshot = canvas.screenshot(timeout=1000)
            np_img = np.frombuffer(screenshot, dtype=np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            
            # Check for the red heart presence (B<50, G<50, R>200)
            red_mask = (img[:, :, 2] > 200) & (img[:, :, 1] < 50) & (img[:, :, 0] < 50)
            has_red_heart = np.sum(red_mask) > 10
            
            # HP is a yellow bar at the bottom. We count the yellow pixels.
            bottom_half = img[240:, :, :]
            yellow_mask = (bottom_half[:, :, 2] > 200) & (bottom_half[:, :, 1] > 200) & (bottom_half[:, :, 0] < 50)
            current_hp = np.sum(yellow_mask)
            
            # Safeguard against WebGL flickering (black frames)
            # If HP drops from >1000 to 0 instantly, it's a visual glitch.
            if current_hp == 0 and getattr(self, 'last_hp', 0) > 1000:
                current_hp = self.last_hp
            
            # Convert to grayscale for the agent
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            obs = np.expand_dims(img_resized, axis=-1)
            return obs, current_hp, has_red_heart
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 1), dtype=np.uint8), self.last_hp if hasattr(self, 'last_hp') else 2310, False

    def step(self, action):
        keys = ACTIONS[action]
        
        for key in keys:
            self.page.keyboard.down(key)
        
        # Action duration
        time.sleep(1/30.0)
        
        for key in keys:
            self.page.keyboard.up(key)

        obs, current_hp, has_red_heart = self._get_frame_and_hp()
        
        # Calculate HP loss
        hp_diff = current_hp - self.last_hp
        self.last_hp = current_hp
        
        # Death is triggered if HP drops to 0 for a few consecutive frames
        if current_hp == 0:
            self.death_counter += 1
        else:
            self.death_counter = 0
            
        done = self.death_counter >= 3
        
        # Reward function
        reward = 0.0
        if has_red_heart:
            reward += REWARD_SURVIVAL
            
        if hp_diff < 0:
            # Penalize losing HP (scaling pixel loss to reward penalty)
            reward += hp_diff * 0.05
            
        if done:
            reward = REWARD_DEATH
            
        info = {'hp': current_hp}
        truncated = False
        
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # We do NOT reload the page anymore since the browser is manually managed
        
        # Spam 'z' until the battle starts and a reward is achievable (red heart appears)
        self.page.locator("canvas").click()
        
        obs, current_hp, has_red_heart = self._get_frame_and_hp()
        
        # Spam 'z' until the combat UI appears (indicated by a visible HP bar, current_hp > 0)
        # This completely skips the Game Over screen, Main Menu, and Dialogue, 
        # preventing the agent from gaining control on the Main Menu and switching modes.
        print("Agent died. Skipping menus and dialogue to restart combat...")
        while current_hp == 0:
            self.page.keyboard.down("z")
            time.sleep(0.05)
            self.page.keyboard.up("z")
            time.sleep(0.05)
            
            obs, current_hp, has_red_heart = self._get_frame_and_hp()
            
        print("Combat started! Agent taking control.")
            
        self.last_hp = current_hp
        self.death_counter = 0
        info = {'hp': current_hp}
        return obs, info

    def render(self):
        if self.render_mode == 'rgb_array':
            obs, _ = self._get_frame_and_hp()
            return obs

    def close(self):
        # Do not close the page, context, or browser here.
        # They are managed externally by launch_game.py.
        # Closing them over CDP shuts down the remote Chromium instance, 
        # which can cause GPU driver crashes (laptop crashes) and leaves 
        # launch_game.py as a zombie process.
        if self.playwright:
            self.playwright.stop()
