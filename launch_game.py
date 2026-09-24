import time
from playwright.sync_api import sync_playwright

def main():
    print("Launching browser with remote debugging on port 9222...")
    with sync_playwright() as p:
        # Launch chromium with remote debugging enabled and framerate unlocked
        browser = p.chromium.launch(headless=False, args=[
            '--remote-debugging-port=9222',
            '--disable-frame-rate-limit', 
            '--disable-gpu-vsync'
        ])
        page = browser.new_page()
        
        print("Loading Bad Time Simulator...")
        page.goto("https://jcw87.github.io/c2-sans-fight/")
        
        print("\n=======================================================")
        print("Browser is ready!")
        print("1. Click into the game window.")
        print("2. Navigate the menu manually to your desired practice attack.")
        print("3. Once the fight is about to start (or you are on the Game Over screen),")
        print("   leave this running and open a new terminal to run 'python test_env.py' or 'python train.py'.")
        print("=======================================================\n")
        
        try:
            # Keep the script alive so the browser stays open
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nClosing browser...")

if __name__ == "__main__":
    main()
