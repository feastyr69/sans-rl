# config.py

# Game URL
URL = "https://jcw87.github.io/c2-sans-fight/"

# Canvas dimensions and crop
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
TARGET_WIDTH = 84
TARGET_HEIGHT = 84

# Action mappings (9 discrete actions: No-op, U, D, L, R, UL, UR, DL, DR)
ACTIONS = {
    0: [],
    1: ["ArrowUp"],
    2: ["ArrowDown"],
    3: ["ArrowLeft"],
    4: ["ArrowRight"],
    5: ["ArrowUp", "ArrowLeft"],
    6: ["ArrowUp", "ArrowRight"],
    7: ["ArrowDown", "ArrowLeft"],
    8: ["ArrowDown", "ArrowRight"],
    9: ["z"],
}

# Reward scaling
REWARD_SURVIVAL = 0.1
REWARD_DEATH = -20.0
