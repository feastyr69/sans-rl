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
        # Allow game to load
        time.sleep(2)
        
        # Press Enter to go from title screen to mode selection
        self.page.keyboard.press("Enter")
        time.sleep(1)
        
        # Press Enter to select the default "Normal" mode
        self.page.keyboard.press("Enter")
        time.sleep(1)
        
        # Mash Enter until the red heart appears (or max 100 tries)
        for _ in range(100):
            self.page.keyboard.press("Enter")
            
            try:
                canvas = self.page.locator('canvas')
                screenshot = canvas.screenshot(timeout=1000)
                np_img = np.frombuffer(screenshot, dtype=np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR) # Read as BGR
                
                # Check for pure red (heart is red, mostly black/white otherwise)
                # BGR format: B<50, G<50, R>200
                red_mask = (img[:, :, 2] > 200) & (img[:, :, 1] < 50) & (img[:, :, 0] < 50)
                if np.any(red_mask):
                    break
            except Exception:
                pass
                
            time.sleep(0.2)
            
        # Give it a moment to fully transition
        time.sleep(0.5)

    def _get_frame(self):
        try:
            canvas = self.page.locator('canvas')
            screenshot = canvas.screenshot(timeout=1000)
            np_img = np.frombuffer(screenshot, dtype=np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            return np.expand_dims(img_resized, axis=-1)
        except Exception as e:
            # Fallback if screenshot fails
            print(f"Failed to get frame: {e}")
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 1), dtype=np.uint8)

    def _check_death(self):
        # A simple placeholder. Since we don't have exact HP hooks yet,
        # we can assume death if the "GAME OVER" screen or certain pixels are present.
        # Returning False for now.
        return False

    def step(self, action):
        keys = ACTIONS[action]
        
        for key in keys:
            self.page.keyboard.down(key)
        
        # Action duration
        time.sleep(1/30.0)
        
        for key in keys:
            self.page.keyboard.up(key)

        obs = self._get_frame()
        done = self._check_death()
        
        reward = REWARD_DEATH if done else REWARD_SURVIVAL
        
        info = {}
        truncated = False
        
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.page.reload()
        self.page.wait_for_selector('canvas')
        self._navigate_menu()
        obs = self._get_frame()
        info = {}
        return obs, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._get_frame()

    def close(self):
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
