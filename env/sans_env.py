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

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(9)
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
        headless = self.render_mode != 'human'
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.page.goto(URL)
        self.page.wait_for_selector('canvas')

    def _navigate_menu(self):
        # Allow game to load completely (loading bar can be red and trigger false positives)
        time.sleep(6)
        
        # Title screen -> Menu
        self.page.keyboard.press("Enter")
        time.sleep(2)
        
        # Menu -> Select "Normal" mode
        self.page.keyboard.press("Enter")
        time.sleep(2)
        
        # Mash 'x' (skip) and 'Enter' (advance) to get through the dialogue until the red heart appears
        for i in range(150):
            self.page.keyboard.press("x")
            self.page.keyboard.press("Enter")
            
            # Start checking for the heart only after a few seconds of dialogue 
            # to ensure we aren't seeing red artifacts from the menu.
            if i > 15:
                try:
                    canvas = self.page.locator('canvas')
                    screenshot = canvas.screenshot(timeout=1000)
                    np_img = np.frombuffer(screenshot, dtype=np.uint8)
                    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR) # Read as BGR
                    
                    # Check for pure red (heart is red, mostly black/white otherwise)
                    # BGR format: B<50, G<50, R>200
                    red_mask = (img[:, :, 2] > 200) & (img[:, :, 1] < 50) & (img[:, :, 0] < 50)
                    if np.sum(red_mask) > 50:
                        break
                except Exception:
                    pass
                    
            time.sleep(0.2)
            
        # Give it a moment to fully transition
        time.sleep(0.5)

    def _get_frame_and_hp(self):
        try:
            canvas = self.page.locator('canvas')
            screenshot = canvas.screenshot(timeout=1000)
            np_img = np.frombuffer(screenshot, dtype=np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            
            # HP is a yellow bar at the bottom. We count the yellow pixels.
            bottom_half = img[240:, :, :]
            yellow_mask = (bottom_half[:, :, 2] > 200) & (bottom_half[:, :, 1] > 200) & (bottom_half[:, :, 0] < 50)
            current_hp = np.sum(yellow_mask)
            
            # Convert to grayscale for the agent
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            obs = np.expand_dims(img_resized, axis=-1)
            return obs, current_hp
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 1), dtype=np.uint8), self.last_hp

    def step(self, action):
        keys = ACTIONS[action]
        
        for key in keys:
            self.page.keyboard.down(key)
        
        # Action duration
        time.sleep(1/30.0)
        
        for key in keys:
            self.page.keyboard.up(key)

        obs, current_hp = self._get_frame_and_hp()
        
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
        reward = REWARD_SURVIVAL
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
        self.page.reload()
        self.page.wait_for_selector('canvas')
        self._navigate_menu()
        obs, current_hp = self._get_frame_and_hp()
        self.last_hp = current_hp
        self.death_counter = 0
        info = {'hp': current_hp}
        return obs, info

    def render(self):
        if self.render_mode == 'rgb_array':
            obs, _ = self._get_frame_and_hp()
            return obs

    def close(self):
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
