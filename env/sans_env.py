import gymnasium as gym
import numpy as np
from playwright.sync_api import sync_playwright
import time
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import URL, ACTIONS, REWARD_SURVIVAL, REWARD_DEATH

MAX_HAZARDS = 20
NUM_FEATURES = 2 + (MAX_HAZARDS * 4)

class SansEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None, mode="normal", attack_index=1, connect_remote=False, headless=True):
        super().__init__()
        self.render_mode = render_mode
        self.game_mode = mode
        self.attack_index = attack_index
        self.connect_remote = connect_remote
        self.headless = headless
        
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(
            low=-10000.0, high=10000.0, shape=(NUM_FEATURES,), dtype=np.float32
        )

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        self.death_counter = 0
        self.steps_alive = 0
        self._start_browser()

    def _start_browser(self):
        self.playwright = sync_playwright().start()
        
        # Local instance optimized for headless speed
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-frame-rate-limit", 
                "--disable-gpu-vsync", 
                "--autoplay-policy=no-user-gesture-required",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding"
            ]
        )
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        
        # Spoof visibility so Construct 2 never pauses the game in background/headless mode
        self.page.add_init_script("""
            Object.defineProperty(document, 'visibilityState', { get: () => 'visible' });
            Object.defineProperty(document, 'hidden', { get: () => false });
            window.addEventListener('visibilitychange', (e) => e.stopImmediatePropagation(), true);
        """)
        
        self.page.goto(URL)
        self.page.wait_for_selector('canvas')
        
        print(f"Browser instance ready! Targeting attack {self.attack_index}.")
        time.sleep(3)

    def _get_state(self):
        js_code = """
        () => {
            try {
                const rt = cr_getC2Runtime();
                if (!rt) return {alive: false};
                
        let soul = null;
        let hazards = [];
        let combatZoneExists = false;
        
        let seen = new Set();
        function findObjects(obj, depth) {
            if (depth > 3) return;
            if (!obj || typeof obj !== 'object') return;
            if (seen.has(obj)) return;
            seen.add(obj);
            
            if (Array.isArray(obj)) {
                if (obj.length > 0 && obj[0] && typeof obj[0] === 'object' && 'x' in obj[0] && 'y' in obj[0] && 'width' in obj[0]) {
                    for (let inst of obj) {
                        if (inst && typeof inst === 'object' && typeof inst.x === 'number') {
                            let typeName = inst.type ? inst.type.name : "";
                            
                            // Combat Zone Border (t15) or Combat Zone (t14)
                            if (typeName === 't15' || typeName === 't14') {
                                combatZoneExists = true;
                            }
                            
                            // Soul (t54)
                            if (inst.width === 16 && inst.height === 16 && typeName === 't54' && !soul) {
                                soul = {x: inst.x, y: inst.y};
                            } 
                            // Hazards (bones, blasters)
                            else if (['t29', 't30', 't32', 't33', 't35', 't36', 't42'].includes(typeName)) {
                                hazards.push({
                                    x: inst.x,
                                    y: inst.y,
                                    w: inst.width,
                                    h: inst.height
                                });
                            }
                        }
                    }
                } else {
                    for (let i = 0; i < obj.length; i++) {
                        findObjects(obj[i], depth + 1);
                    }
                }
            } else {
                for (let key in obj) {
                    try { findObjects(obj[key], depth + 1); } catch(e) {}
                }
            }
        }
        findObjects(rt, 0);
        
        return {
            alive: soul !== null && combatZoneExists,
            soul: soul || {x: 0, y: 0},
            hazards: hazards
        };
    } catch(e) {
        return {alive: false};
    }
        }
        """
        try:
            result = self.page.evaluate(js_code)
            alive = result.get('alive', False)
            soul = result.get('soul', {'x': 0, 'y': 0})
            hazards = result.get('hazards', [])
            
            obs = np.zeros(NUM_FEATURES, dtype=np.float32)
            obs[0] = soul['x']
            obs[1] = soul['y']
            
            # Sort hazards by distance to soul
            hazards.sort(key=lambda h: (h['x'] - soul['x'])**2 + (h['y'] - soul['y'])**2)
            
            idx = 2
            for h in hazards[:MAX_HAZARDS]:
                obs[idx] = h['x'] - soul['x']
                obs[idx+1] = h['y'] - soul['y']
                obs[idx+2] = h['w']
                obs[idx+3] = h['h']
                idx += 4
                
            return obs, alive
        except Exception as e:
            return np.zeros(NUM_FEATURES, dtype=np.float32), False

    def step(self, action):
        keys = ACTIONS[action]
        for key in keys:
            self.page.keyboard.down(key)
        
        # Uncapped speed: very short sleep just for browser thread to register
        time.sleep(1/60.0) 
        
        for key in keys:
            self.page.keyboard.up(key)

        obs, alive = self._get_state()
        
        if not alive:
            self.death_counter += 1
        else:
            self.death_counter = 0
            self.steps_alive += 1
            
        done = self.death_counter >= 2
        
        # Reward is survival + a bonus if it survives a long time
        reward = REWARD_SURVIVAL if alive else 0.0
        if done:
            reward = REWARD_DEATH
            
        info = {'steps_alive': self.steps_alive}
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.page.reload()
        self.page.wait_for_selector('canvas')
        time.sleep(1) # wait for game to init
        
        self.page.locator("canvas").click()
        
        # Wait for the main menu to load by checking for the cursor
        js_cursor = """
        () => {
            const rt = cr_getC2Runtime();
            let soul = null;
            let seen = new Set();
            function findObjects(obj, depth) {
                if (depth > 3 || soul) return;
                if (!obj || typeof obj !== 'object') return;
                if (seen.has(obj)) return;
                seen.add(obj);
                if (Array.isArray(obj)) {
                    if (obj.length > 0 && obj[0] && typeof obj[0] === 'object' && 'x' in obj[0] && 'y' in obj[0] && 'width' in obj[0]) {
                        for (let inst of obj) {
                            if (inst && typeof inst === 'object' && typeof inst.x === 'number') {
                                let typeName = inst.type ? inst.type.name : "";
                                if (inst.width === 16 && inst.height === 16 && typeName === 't54') {
                                    soul = true;
                                }
                            }
                        }
                    } else {
                        for (let i = 0; i < obj.length; i++) findObjects(obj[i], depth + 1);
                    }
                } else {
                    for (let key in obj) {
                        try { findObjects(obj[key], depth + 1); } catch(e) {}
                    }
                }
            }
            findObjects(rt, 0);
            return soul;
        }
        """
        
        attempts = 0
        while attempts < 100:
            if self.page.evaluate(js_cursor):
                break
            time.sleep(0.1)
            attempts += 1
            
        # Navigate Main Menu -> Single Attack
        for _ in range(3):
            self.page.keyboard.press("ArrowDown", delay=50)
            time.sleep(0.1)
        self.page.keyboard.press("z", delay=50)
        time.sleep(0.5)
        
        # Select attack in Single Attack menu
        for _ in range(self.attack_index - 1):
            self.page.keyboard.press("ArrowDown", delay=50)
            time.sleep(0.1)
            
        self.page.keyboard.press("z", delay=50)
        time.sleep(0.5)
        
        # Skip any dialogue until soul spawns AND combat zone is present
        obs, alive = self._get_state()
        attempts = 0
        while not alive and attempts < 100:
            self.page.keyboard.press("z", delay=50)
            time.sleep(0.1)
            obs, alive = self._get_state()
            attempts += 1
            
        self.death_counter = 0
        self.steps_alive = 0
        return obs, {}

    def render(self):
        pass

    def close(self):
        if self.playwright:
            self.playwright.stop()
