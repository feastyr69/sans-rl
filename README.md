# Sans RL

A Reinforcement Learning pipeline to defeat the HTML5 *Bad Time Simulator* (Sans boss fight). 

The project uses a Playwright-driven headless browser to interface with the game and trains an agent using Stable-Baselines3 (PPO) via a custom Gymnasium environment.

## Setup
1. Create a virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Install Playwright binaries: `playwright install chromium`

## Run
* Test environment: `python test_env.py`
* Train model: `python train.py`
* Evaluate model: `python evaluate.py`
